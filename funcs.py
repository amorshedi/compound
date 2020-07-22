import numpy as np
import os, re, subprocess, math
from lammps import PyLammps, lammps
from numpy.linalg import norm

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


def get_lmps(configs):
    '''
    :param configs: nframes*natoms*3
    :return: list of lammps objects for each config, forces for these configurations
    '''
    lmps = []
    frc_lmps = np.empty(configs.shape, dtype=float)
    for i, pos in enumerate(configs):
        lmp = lammps(cmdargs=['-echo', 'none', '-screen', 'none'])  # type: PyLammps
        lmp.file("runfile")
        x = lmp.extract_atom("x", 3)
        for j, p in enumerate(pos):
        	for k, c in enumerate(p):
        		x[j][k] = c
        lmps.append(lmp)
        lmp.command('run 0')
        frc_lmps[i] = np.ctypeslib.as_array(lmp.extract_atom("f", 3).contents, shape=configs[0].shape)
    return lmps, frc_lmps

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

def coords_forces_from_outcar(direc, nump):
    '''direc: outcar directory
       nump: number of particles'''

    dft_pos_frc = subprocess.check_output(
        f'''awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg{{print $1, $2, $3, $4, $5, $6}}' {direc}/OUTCAR ''', shell=1)
    dft_pos_frc = np.vstack([np.fromstring(x, np.float, sep=' ')
                        for x in dft_pos_frc.decode('utf-8').split('\n')[:-1]])
    dft_pos_frc = cut_array2d(dft_pos_frc, [len(dft_pos_frc)//nump, 1])

    [x[:,:3] for x in dft_pos_frc]

    # os.system(f"awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg' {direc} > out")
    # out = np.loadtxt('out').reshape((-1, nump, 6), order='C')
    # os.remove('out')
    return [x[:,:3] for x in dft_pos_frc], [x[:,3:] for x in dft_pos_frc]

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

