import numpy as np
import os, re, subprocess, math, io
from lammps import PyLammps, lammps
from numpy.linalg import norm, svd
from math import degrees, radians
from collections.abc import Iterable
# from lib.recipes.alkane import Alkane
# from compound import Compound, compload, Port
from copy import deepcopy

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def pairs_in_rcut(compound, pairs, rcut):
    tmp = set()
    tpairs = []
    for p in pairs:
        _, dist = compound.neighbs([p[0]], [p[1]], rcut=3.6)
        if dist and not [x for x in p if x in tmp]:
            tpairs.append(p)
            [tmp.add(x) for x in p]
    return tpairs

def transform_mat(A, B):
    ''' two sets of 3D points, gives the matrix that rigid transforms one into the other.
     for details see http://nghiaho.com/?page_id=671'''
    A, B = map(np.array, [A, B])

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centroid_A.shape = (1, 3)
    centroid_B.shape = (1, 3)

    H = np.zeros((3, 3), dtype=float)
    for i in range(A.shape[0]):
        H = H + (A[i, :] - centroid_A).reshape(3,1) @ (B[i, :] - centroid_B)
    U, _, V = svd(H)
    R = np.transpose(U @ V)

    t = (centroid_B.reshape(3,1) - R @ centroid_A.reshape(3,1))
    return np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])

def apply_transform(T, points):
    points = np.array(points)
    one_row = np.ones([1,points.shape[0]])
    return (T @ np.vstack([points.T, one_row]))[0:3,:].T


def alkane(n):
    from compound import Compound, Port
    ch2 = Compound(name='ch2')
    ch2.add(Compound(name='C',pos=[0,0.5935,0]))
    ch2.add(Compound(name='H',pos=[0,1.2513,0.8857]))
    ch2.add(Compound(name='H',pos=[ 0,1.2513,-0.8857]))

    dr1, dr2 = [-0.64315, -0.42845, 0], [ 0.64315, -0.42845, 0]
    ch2.add(Port(anchor=ch2['C'], loc_vec=dr1),expand=0)
    ch2.add(Port(anchor=ch2['C'], loc_vec=dr2),expand=0)
    [ch2.add_bond(x) for x in [[ch2[0], ch2[1]],[ch2[0], ch2[2]]]]
    alk = Compound(name='alkane')
    alk.add(deepcopy(ch2))
    for i in range(n-1):
        c_ch2 = deepcopy(ch2)
        alk.add(c_ch2)
        alk.force_overlap(c_ch2, c_ch2[-2], alk[-6], flip=1)

    return alk

def dimer(nchain):
    from compound import Compound, compload, Port
    from funcs import alkane
    alksil = Compound(name='alksil')

    sil = compload('/home/ali/ongoing_research/polymer4/22_lammps_py_fitting/sil.mol2', compound=Compound(name='sil'),
                   infer_hierarchy=0)  # type: Compound
    c2, c1 = [6.3001, 4.1639, 6.0570], [8.5253, 8.0180, 6.0570]

    tmp1, tmp2 = c1 - sil['Si1'].pos, c2 - sil['Si2'].pos
    sil.add(Port(sil['Si1'], loc_vec=tmp1, orientation=[0, 0, 1], name='c1'), expand=0)
    sil.add(Port(sil['Si2'], loc_vec=tmp2, orientation=[0, 0, 1], name='c2'), expand=0)

    alksil.add(sil, expand=0)
    alkane = alkane(nchain)
    alkane['port'][0].translate_to(alkane['C'][0].pos)  # translate port to carbon
    alkane['port'][0].rotate(-90, [0, 0, 1])
    # alkane.energy_minimize()

    alksil.add(deepcopy(alkane), expand=0)
    alksil.add(alkane, expand=0)

    alksil.force_overlap(alksil['alkane'][0], alksil['alkane'][0]['port'][0], sil['c1'], flip=1)
    alksil.force_overlap(alksil['alkane'][1], alksil['alkane'][1]['port'][0], sil['c2'], flip=1)
    alksil.remove([x.parent for x in alksil.particles_by_name('_p')])
    return alksil

def mod_com(commands, intervals, tt, opts, pstr: 'partial string', par_lst, par_lst2):
    a, b = [eval(str(tt['f'+x])) for x in opts[:2]]
    temp = (opts[2] + pstr + ('* ' if opts[2] == 'pair' else '') + '{} {}\n')\
        .format('{}' if a else tt[opts[0]], '{}' if b else tt[opts[1]])
    intervals.extend([x for x in (a, b) if x])
    if '{' not in temp:
        return commands
    commands += temp
    def get_str(x, y):
        if 'bond' in opts:
            return f"{opts[2]} ~ {tt['type1']}-{tt['type2']} ~ {y} ~ {str(x)} "
        elif 'pair' in opts:
            return f"{opts[2]} ~ {tt['type']} ~ {y} ~ {str(x)} "
        elif 'angle' in opts:
            return f"{opts[2]} ~ {tt['type1']}-{tt['type2']}-{tt['type3']} ~ {y} ~ {str(x)} "
    if a:
        par_lst.append(get_str(a, opts[0]))
        par_lst2.append((dict(tt), opts[0]))
    if b:
        par_lst.append(get_str(b, opts[1]))
        par_lst2.append((dict(tt), opts[1]))
    return commands


def get_lmps(configs, inp_fle='runfile'):
    '''
    :param configs: nframes*natoms*3
    :return: list of lammps objects for each config, forces for these configurations
    '''
    lmps = []
    frc_lmps = np.empty(configs.shape, dtype=float)
    for i, pos in enumerate(configs):
        lmp = lammps(cmdargs=['-echo', 'none', '-screen', 'none'])
        lmp.file(inp_fle)
        x = lmp.extract_atom("x", 3)
        for j, p in enumerate(pos):
        	for k, c in enumerate(p):
        		x[j][k] = c
        lmps.append(lmp)
        lmp.command('run 0')
        frc_lmps[i] = np.ctypeslib.as_array(lmp.extract_atom("f", 3).contents, shape=configs[0].shape)
    return lmps[0] if len(lmps)==1 else lmps, frc_lmps

def ev_to_kcalpmol(input):
    from parmed import unit as u
    return u.AVOGADRO_CONSTANT_NA._value * (
		input * u.elementary_charge * u.volts).in_units_of(u.kilocalorie)

def get_file_num(outdir):
    files = next(os.walk(f'./{outdir}'))[-1]
    try:
        return np.sort(np.array([int(re.match('[^_a-zA-Z]*', x).group())
                            for x in files if re.match('[^_a-zA-Z]*', x).group() != '']))[-1] + 1
    except:
	    return 1

def coords_forces_from_outcar(direc, nump,save_npy=0):
    '''direc: outcar directory
       nump: number of particles'''

    dft_pos_frc = subprocess.check_output(
        f'''awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg{{print $1, $2, $3, $4, $5, $6}}' {direc}/OUTCAR ''', shell=1)
    dft_pos_frc = np.vstack([np.fromstring(x, np.float, sep=' ')
                        for x in dft_pos_frc.decode('utf-8').split('\n')[:-1]])
    dft_pos_frc = cut_array2d(dft_pos_frc, [len(dft_pos_frc)//nump, 1])

    return [x[:, :3] for x in dft_pos_frc], [x[:, 3:] for x in dft_pos_frc]

def cut_array2d(array, shape):
    xstep = array.shape[0]//shape[0]
    ystep = array.shape[1]//shape[1]

    if 1 in shape:
        blocks = np.repeat(None, np.max(shape))
    else:
        blocks = np.tile(None, shape)

    cnt = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = array[i*xstep:(i+1)*xstep, j*ystep:(j+1)*ystep]
            if 1 in shape:
                blocks[cnt] = val
                cnt += 1
            else:
                blocks[i, j] = val
    return blocks

def get_vasp_hessian(direc, n):
    hessian = subprocess.check_output(
        f''' awk '/^  1X/{{flg=1}};/^ Eigen/{{flg=0}}flg{{$1="";print $0}}' {direc}/OUTCAR ''',
        shell=1)
    return np.float_(hessian.decode('utf-8').split()).reshape(n * 3, n * 3)

def  unit_perps(ang, coords):
    #This gives the vector in the plane A,B,C and perpendicular to A to B

    diff_AB = coords[ang[1], :] - coords[ang[0], :]
    u_AB = diff_AB / norm(diff_AB)

    diff_CB = coords[ang[1], :] - coords[ang[2], :]
    u_CB = diff_CB / norm(diff_CB)

    cross_product = np.cross(u_CB, u_AB)
    u_N = cross_product / norm(cross_product)

    u_PA = np.cross(u_N, u_AB)
    u_PC = np.cross(u_N, u_CB)

    return u_PA / norm(u_PA), u_PC / norm(u_PC), math.degrees(math.acos(np.dot(u_AB, u_CB)))

def equivalence_classes(objects, func):
    ''' func should take two arguments of objects and return true or fals'''
    objects = list(objects)
    out = []
    while objects:
        eq_class = [x for x in objects if func(x, objects[0])]
        out.append(eq_class)
        objects = [x for x in objects if x not in eq_class]
    return out

def find_file(name, paths):
    results = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            if name in files:
                results.append(os.path.join(root, name))
    return results

def get_vasp_freqs(direc):
    ''' units of output (each column): THz - 2PiTHz - cm-1  - meV'''

    command = f"awk '/2PiTHz/{{if (\"f\"==$2) {{print $4, $6, $8, $10}} else {{print $3, $5, $7, $9}}}}' {os.path.join(direc,'')}OUTCAR"
    file = subprocess.check_output(command, shell=1)
    return np.genfromtxt(io.BytesIO(file))

def get_gulp_freqs(direc):
    command = f"awk '/Frequencies/{{getline;getline;flg=1}};NF==0&&flg{{print \"sep\";flg=0}};flg' {direc}"
    file = subprocess.check_output(command, shell=1).decode('utf-8').split('sep')[-2].replace('\n',' ')
    return np.fromstring(file, sep=' ')

def angle_between_vecs(v1, v2, in_degrees=1):
    v1, v2 = map(np.array,[v1, v2])
    ang = np.arccos(v1 @ v2/(norm(v1) * norm(v2)))

    return degrees(ang) if in_degrees else ang


