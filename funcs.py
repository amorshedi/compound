import numpy as np
import sys
from itertools import islice
import os, subprocess, math, io, fnmatch
from glob import glob
# from lammps import lammps
from numpy import dot, cross, sqrt, einsum
from numpy.linalg import norm, svd, eig
from math import degrees, radians
from collections.abc import Iterable
from collections import Counter, UserDict
from pathlib import Path
from parmed import unit as u
from subprocess import check_output
# from molmod.ic import bond_length, bend_angle, dihed_angle
from itertools import combinations, chain, product
# from scipy.spatial.transform import Rotation as R
from warnings import warn
from operator import itemgetter
# from lib.recipes.alkane import Alkane
# from compound import Compound, compload, Port
from copy import deepcopy
import regex as re
import importlib.util
import dill

def module_from_file(file_path):
    pth = re.findall('.*/', file_path)[0]
    cwd = os.getcwd()
    os.chdir(pth)
    
    spec = importlib.util.spec_from_file_location('', file_path.replace(pth, ''))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.chdir(cwd)
    return module

def skip_blanks(f):
    ''' f: iterator, skip the blank lines '''
    for x in f:
        if x.strip():
            return x


def expand_box_pair(comp, rcut, idx=[]):
    ''' expand the box such that within rcut gives unique pair distances '''
    from compound import Compound
    b = np.linalg.inv(comp.latmat)

    if not idx:
        idx = np.ceil(rcut * np.apply_along_axis(norm, 1, b))
    idx = np.array(idx, dtype=int) + 1

    comp2 = Compound()
    comp2.latmat = comp.latmat
    xx, yy, zz = np.meshgrid(range(-idx[0] + 1, idx[0]), range(-idx[1] + 1, idx[1]), range(1, idx[2]))

    for x, y, z in zip(xx.flatten(), yy.flatten(), zz.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, z])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
    xx, yy = np.meshgrid(range(0, idx[0]), range(0, idx[1]))
    for x, y in zip(xx.flatten(), yy.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, 0])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
        if [x, y] == [0, 0]:
            porig = c.particles()
    xx, yy = np.meshgrid(range(-idx[0] + 1, 0), range(1, idx[1]))
    for x, y in zip(xx.flatten(), yy.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, 0])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
    comp2.latmat *= (idx[:, np.newaxis]-1)*2+1

    # nlst1 = porig
    # nlst2 = comp2.particles()
    # n1 = len(nlst1)
    # n2 = len(nlst2)
    # cnt = 0
    # if comp2.bond_graph:
    #     for x in comp2.bond_graph.edges:
    #         cnt += 1
    #         equivalent_parts = [nlst2[i] for i in range(nlst2.index(x[1]) % n1, n2, n1)]
    #         p = comp2.closest_img(x[0], equivalent_parts)
    #         if x[1] is not p:
    #             comp2.bond_graph.remove_edge(*x)
    #             comp2.bond_graph.add_edge(x[0], p)
        # comp2.gen_angs_and_diheds()

    return comp2, porig


def rm_bonds(a, b, p):
    '''a:typed b:types'''
    if not isinstance(p, Iterable):
        p = [p]
        
    typed = dict()
    types = []
    for y in a:
        if np.any([(x in y) for x in p]):
            typed[y] = b[list(a).index(y)]
            types.append(b[list(a).index(y)])
    return typed, types


def points_in_sphere():
    d = .12
    tmp = np.arange(-d, d, .0266*d/.1)
    x, y, z = np.meshgrid(tmp, tmp, tmp);
    idx = x**2 + y**2 + z**2 < d**2;
    return x[idx], y[idx], z[idx]


def dist_sphere(num_pts):
    from numpy import pi, cos, sin, arccos, arange

    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2 * indices / num_pts)
    theta = pi * (1 + 5 ** 0.5) * indices

    return cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);#x, y, z =

    import plotly.graph_objects as go
    go.Figure(go.Scatter3d(x=x,y=y,z=z,mode='markers')).show()



def gulp_confs_to_xyz(direc='gulp.in'):
    out = shell(f'''awk '/cart/{{getline;flg=1}}NF<3{{flg=0}}flg{{print $1, $2, $3, $4}}' {direc}''').decode().split('\n')
    n = shell(f'''awk '/cart/{{flg=1;getline;n=NR}} flg && NF<3 {{print NR-n;flg=0}}' {direc}''').decode().split()
    with open('out.xyz', 'w') as f:
        for i, x in enumerate(n):
            f.write('\n'+x+'\n')
            f.write('comment\n')
            f.write('\n'.join(out[i*int(x):(i+1)*int(x)]))

def get_gulp_configs(direc='gulp.in'):
    f = iter(open(direc))
    flg = 0
    out = ''
    ret = []
    for x in f:
        if 'vector' in x:
            flg = 1
        if 'end' in x and flg:
            flg = 0
            ret.append(out+x)
            out = ''
        if flg:
            out += x
    return ret
    

def normalize(x):
    return x/norm(x)

def hybrid_struct():
    '''create bonding for relaxed structure coming from vasp'''
    from compound import compload
    comp = hybrid_silane_portlandite()
    comp.xyz_label_sorted = compload('/home/ali/ongoing_research/polymer4/45_hybrid_hexane/2_opt_allfree_prec/CONTCAR').xyz_label_sorted
    
    comp.generate_bonds(comp.particles_by_name('C'), comp.particles_by_name('Si'), 1.5, 1.9)
    comp.generate_bonds(comp.particles_by_name('Ca'), comp.particles_by_name('O'), 2.23, 2.54)
    comp.create_bonding_all(kb=0, ka=0, kd=0, acfpath='/home/ali/ongoing_research/polymer4/45_hybrid_hexane/4_charge/')
    rm = []
    for x in comp.propers_typed.keys():
        if [y for y in x if y.name == 'Ca']:
            rm.append(x)
    
    for x in rm:
        comp.ff.proper_types.remove(comp.propers_typed[x])
        comp.propers_typed.pop(x)
    return comp


def rot_ax_ang(ax, ang, vec, deg=1):
    from scipy.spatial.transform import Rotation as R
    ax = normalize(ax)
    if deg:
        ang = math.radians(ang)
    rot = R.from_rotvec(ax * ang)
    return rot.apply(vec)

def get_fitted(comp, direc='.'):
    pth = os.path.join(direc, 'gulp.in')
    tmp = check_output("""awk '/harm/||/three/{getline;flg=1}NF==0{flg=0}flg' """+pth+
                       """| awk '/harm/||/three/{getline;flg=1}NF==0{flg=0}flg' """+pth+""" |
                          grep -Po '(?<=\s)[0-9].*(?=\#)'""", shell=1)
    qlist = [*comp.bonds_typed, *comp.angles_typed, *comp.propers_typed]
    k, rest, flags = [],[],[]
    rest = deepcopy(k)
    for x, y in zip(tmp.decode().replace('\t','').split('\n')[:-1], qlist):
        x = x.split()
        k.append(x[0])
        rest.append(x[1])
        flags.append(x[-2:])

    pth = os.path.join(direc, 'gulp.out')
    tmp = check_output("""awk '/Parameter        P/{getline;getline;getline;flg=1}/-----/{flg=0}flg{print $3}'  """+pth, shell=1)
    # vals = np.genfromtxt(io.BytesIO(tmp))
    
    return k, rest, np.array(flags, dtype=int), tmp.decode().split('\n')[:-1]
    
def shell(comm, inp=''):
    '''comm: command to run
    inp: a byte string to give as input'''
    try:
        o = check_output(comm, shell=1, input=inp, executable='/bin/bash') if inp else check_output(comm, shell=1, executable='/bin/bash')
    except subprocess.CalledProcessError as ex:
        o = ex.output

    return o

def amass(nme):
    ''' nme: a string with the name of the element '''
    import pymatgen.core.periodic_table as pt
    return pt.Element(nme).atomic_mass.real

def dist(c1, c2):
    c1, c2 = map(np.array, [c1, c2])
    return norm(c1 - c2)


def modsem(comp, outcar_pth, vibrational_scaling=1, ff_xml_pth=None):
    ''' initial version is in /home/ali/ongoing_research/my_python_codes/modsem'''
    from indexed import IndexedOrderedDict
    # faulthandler.enable()

    vibrational_scaling_squared = vibrational_scaling ** 2 # Square the vibrational scaling used for frequencies
    parts = comp.particles_label_sorted()
    eigenvectors, eigenvalues = dict(), dict()
    hessian = -get_vasp_hessian(outcar_pth)
    n = comp.n_particles()
    hessian_partial = cut_array2d(hessian, [n, n])
    for i, p1 in enumerate(parts):
        for j, p2 in enumerate(parts):
            eigenvalues[p1, p2], eigenvectors[p1, p2] = eig(hessian_partial[i, j])

    bonds_list, angles_list, diheds_list = map(list, (comp.bonds_typed.keys(), comp.angles_typed.keys(), comp.propers_typed))
    bonds_length_list = IndexedOrderedDict()

    for x in bonds_list:
        bonds_length_list[frozenset(x)] = dist(*comp.closest_img_bond(*x))
    # atom_names = [x.type['name'] for x in comp.particles_label_sorted()]

    def force_constant_bond(atom_A, atom_B, eigenvalues, eigenvectors, c0, c1):
        # Force Constant - Equation 10 of Seminario paper - gives force constant for bond

        diff_AB = c1 - c0
        unit_vector_AB = diff_AB / norm(diff_AB)  # Vector along bond

        return eigenvalues[atom_A, atom_B] @ np.abs(
            unit_vector_AB @ eigenvectors[atom_A, atom_B]) / 2  # divide by 2 cause k/2 = K

    k_b = np.zeros(len(bonds_list))
    for i, bnd in enumerate(bonds_list):
        c0, c1 = comp.closest_img_bond(bnd[0], bnd[1])
        AB = force_constant_bond(bnd[0], bnd[1], eigenvalues, eigenvectors, c0, c1)
        BA = force_constant_bond(bnd[1], bnd[0], eigenvalues, eigenvectors, c0, c1)

        k_b[i] = np.real((AB + BA) / 2) * vibrational_scaling_squared # Order of bonds sometimes causes slight differences, find the mean

    val = [0, 2]
    k_a, theta = [np.zeros(len(angles_list)) for _ in range(2)]

    for iang, ang in enumerate(angles_list):
        same_centers = [x for x in angles_list if x[1] == ang[1]]
        c0, c1, c2 = comp.closest_img_angle(*ang)
        upa, upc, theta[iang] = unit_perps(c0,c1,c2)
        upac = upa, upc

        invk = 0
        for i, bond in enumerate([ang[0: 2], ang[2:0:-1]]):
            tmp = np.abs(upac[i] @ eigenvectors[bond[0], bond[1]])
            ki = tmp @ eigenvalues[bond[0], bond[1]]

            coeff = cnt = 0
            for x in [y for y in same_centers if bond[0] in y and set(y) != set(ang)]: #among bonds with the same center as ang, the ones share bond with ang
                c0, c1, c2 = comp.closest_img_angle(*bond, *[i for i in x if i not in bond])
                up = unit_perps(c0, c1, c2)
                coeff += np.abs(upac[i] @ up[0]) ** 2
                cnt += 1
            fact = 1 + coeff / cnt if cnt else 1
            invk += fact / (bonds_length_list[frozenset(bond)] ** 2 * ki)
        k_a[iang] = 1 / np.real(invk) / 2

    k_d, phi = [np.zeros(len(diheds_list)) for _ in range(2)]
    for i, dihed in enumerate(diheds_list):
        points = comp.closest_img_dihed(*dihed[0:4])
        normals = plane_normal(*points[0:3]), plane_normal(*points[1:])
        phi[i] = np.degrees(np.arccos(normals[0] @ normals[1]))

        invk = 0
        for j in range(2): # loop over each dihedral arm (AB or CD)
            blen = bonds_length_list[frozenset(dihed[2*j:2*j+2])]
            sint_sq = np.sin(angle_between_vecs(points[j]-points[j+1], points[j+2]-points[j+1], degrees=0))**2
            tmp = np.abs(normals[j] @ eigenvectors[dihed[2*j], dihed[2*j+1]])
            tmp = tmp @ eigenvalues[dihed[2*j], dihed[2*j+1]]

            invk += 1/blen/sint_sq/tmp
        k_d[i] = 1/invk.real

    return np.array([*k_b, *k_a, *k_d])


def parms_from_labels(lbls, comp):
    '''you have connections in label form, you want them in compound tuple'''
    n = len(lbls)
    v = [comp.bonds_typed, comp.angles_typed, comp.propers_typed][n - 2]
    gg = tuple([z for z in comp.particles() if z.type['name'] == qq][0] for qq in lbls)
    if gg not in v:
        gg = gg[-1::-1]
    return float(v[gg]['k']), abs(float(v[gg]['rest']))

def swap_rows_cols(mat, idx):
    idx = list(map(list, idx))
    for i in range(2):
        for x in idx:
            if len(set(x)) == 1:
                continue
            mat[x] = mat[np.flip(x)]
        mat = mat.T

    return mat

def rm_dependent_rows(mat):
    mat = np.array(mat)
    
    out = []
    idx = []
    for cnt, x in enumerate(mat):
        if np.linalg.matrix_rank([*out, x]) == len(out) + 1:
            out.append(x)
            idx.append(cnt)
    return idx, np.array(out)[:, idx]


def zet(s):
    return (s[0] == s[1]) - (s[0] == s[2])

def dihed_derivs2(p1,p2,p3,p4):

    u = p1 - p2
    w = p3 - p2
    v = p4 - p3
    nu,nv,nw = np.array([norm(x) for x in [u,v,w]])
    u,v,w = u/nu, v/nv, w/nw

    cphi_u = dot(u,w)
    sphi_u = sqrt(1-cphi_u**2)
    cphi_v = -dot(v,w)
    sphi_v = sqrt(1-cphi_v**2)
    cuw = cross(u,w)
    cvw = cross(v,w)

    sphi_u4 = sphi_u**4
    sphi_v4 = sphi_v**4

    t1 = einsum('i,j', cuw, w*cphi_u-u)
    e1 = (t1 + t1.T)/sphi_u4/nu**2

    t2 = einsum('i,j', cvw, w*cphi_v-v)
    e2 = (t2+t2.T)/sphi_v4/nv**2

    t3 = einsum('i,j', cuw, w - 2*u*cphi_u + w*cphi_u**2)
    e3 = (t3+t3.T)/sphi_u4/2/nu/nw

    t4 = einsum('i,j', cvw, w + 2*u*cphi_v + w*cphi_v**2)
    e4 = (t4+t4.T)/sphi_v4/2/nv/nw

    t5 = einsum('i,j', cuw, u + u*cphi_u**2 - 3*w*cphi_u + w*cphi_u**3)
    e5 = (t5+t5.T)/sphi_u4/2/nw**2

    t6 = einsum('i,j', cvw, v + v*cphi_v**2 + 3*w*cphi_v - w*cphi_v**3)
    e6 = (t6+t6.T)/sphi_v4/2/nw**2
    
    e7 = np.zeros([3,3])
    e8 = deepcopy(e7)
    for i in range(3):
        for j in [x for x in range(3) if x!=i]:
            k = [x for x in range(3) if x not in [i,j]][0]
            e7[i,j] = (j-i) * (-1/2)**np.abs(j-i) * (w[k]*cphi_u - u[k])/nu/nw/sphi_u
            e8[i,j] = (j-i) * (-1/2)**np.abs(j-i) * (w[k]*cphi_v - v[k])/nv/nw/sphi_v

    val = np.zeros([12,12])
    for cnt1,a in enumerate('mopn'):
        for cnt2,b in enumerate('mopn'):
           val[3*cnt1:3*(cnt1+1),3*cnt2:3*(cnt2+1)] = \
            zet(a+'mo')*zet(b+'mo')*e1 + zet(a+'np')*zet(b+'np')*e2+\
            (zet(a+'mo')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e3 +\
            (zet(a+'np')*zet(b+'po') + zet(a+'po')*zet(b+'np'))*e4+\
            zet(a+'op')*zet(b+'po')*e5+\
            zet(a+'op')*zet(b+'op')*e6+\
                  (1- (a==b))*(zet(a+'mo')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e7+\
                  (1- (a==b))*(zet(a+'no')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e8
    return val

def angle_derivs(rs, n=0):
    p1,p2,p3 = rs
    u = p1 - p2
    v = p3 - p2
    nu,nv = norm(u), norm(v)
    u, v = u/nu, v/nv
    ang = np.arccos(dot(u, v))
    if n == 0:
        return ang
        
    w = cross(u, v)
    if np.allclose(w, 0):
        # w = cross(u, [1, -1, 1])
        u = u + np.array([1, -1, 1])*.0001
        w = cross(u, v)
    if np.allclose(w, 0):
        w = cross(u, [-1, 1, 1])
    nw = norm(w)
    w = w/nw

    val = np.zeros(9)
    if n > 0:
        for cnt, a in enumerate('mon'):
            val[3*cnt:3*(cnt+1)] = zet(a+'mo')*cross(u, w)/nu + zet(a+'no')*cross(w, v)/nv
        if n == 1:
            return ang, val

    cqa = dot(u, v)
    sqa = sqrt(1-cqa**2)

    sval = np.zeros([9, 9])
    if n==2:
        try:
            for cnt1, a in enumerate('mon'):
                for cnt2, b in enumerate('mon'):
                    sval[3*cnt1:3*(cnt1+1),3*cnt2:3*(cnt2+1)] =\
                        zet(a+'mo')*zet(b+'mo')*(einsum('i,j', u, v) + einsum('j,i', u, v) - 3*einsum('i,j', u, u)*cqa + np.eye(3)*cqa)/nu**2/sqa +\
                        zet(a+'no')*zet(b+'no')*(einsum('i,j', v, u) + einsum('j,i', v, u) - 3*einsum('i,j', v, v)*cqa + np.eye(3)*cqa)/nu**2/sqa +\
                        zet(a+'mo')*zet(b+'no')*(einsum('i,j', u, u) + einsum('j,i', v, v) - einsum('i,j', u, v)*cqa - np.eye(3))/nu/nv/sqa +\
                        zet(a+'no')*zet(b+'mo')*(einsum('i,j', v, v) + einsum('j,i', u, u) - einsum('i,j', v, u)*cqa - np.eye(3))/nu/nv/sqa -\
                        cqa/sqa*einsum('i,j', val[3*cnt1:3*(cnt1+1)], val[3*cnt2:3*(cnt2+1)])
        except FloatingPointError:
            sval = np.zeros([9, 9])

        return ang, val, sval
        


def plt_eigenvecs(coords, ev):
    import plotly.graph_objects as go

    # eigenvec = eigenvec.reshape([-1, 3])

    fig = go.Figure(data=go.Cone(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], u=ev[0::3], v=ev[1::3], w=ev[2::3], anchor='tip', hoverinfo='u+v+w'))
    # fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode='markers'))
    # fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))

    fig.show()


def sub_list(lst, idx):
    return [x for x in lst if lst.index(x) in idx]


def mat_prt(mat, frm = '{:12.5f} ', prt=1):
    tmp = '\n'.join([(len(x)*frm).format(*x) for x in mat])
    if prt:
        print(tmp)
    return tmp
    
def brace_to_curly(lst):
    return str(lst).replace('[','{').replace(']', '}')


def gulp_hessian(direc='gulp.out', massed=1):
    '''extract hessian from gulp output'''
    n = shell('''awk '/Number of irred/ {print $6}' ''' + direc).decode('utf-8')
    n = int(n)*3

    with  open(direc, 'r') as f:
        for cnt, x in enumerate(f):
            if 'Real Dynamical' in x:
                f.readline()
                vals = [None] * n**2
                for i in range(n**2):
                    tmp = f.read(11)
                    if '\n' in tmp:
                        tmp = tmp.replace('\n', '') + f.read(1)
                    vals[i] = tmp
                break
    
    hessian = np.array(vals, dtype=float).reshape([n, n])
    if massed:
        return hessian
    
    # dyn_mat = check_output('''awk '/Real Dynamical/{getline;getline;flg=1}NF<2{flg=0}flg' '''+direc, shell=1)
    # dyn_mat = np.array(dyn_mat.decode('utf-8').split(), dtype=float)[-n*n:].reshape([n, -1])
    #
    # hessian = np.zeros([n, n])
    parts = comp.particles_label_sorted()
    masses = np.repeat(np.array([x.type['mass'] for x in parts], dtype=float), 3)
    for i in range(n):
        for j in range(n):
            hessian[i, j] = np.sqrt(masses[i]*masses[j]) * hessian[i, j]
    return ev_to_kcalpmol(hessian)._value


def nearestr(r, latmat):
    '''nearest for vector r
    rv = latmat
    '''
    lst = np.zeros([27, 3])
    for i, n in enumerate(product([-1,0,1], [-1,0,1], [-1,0,1])):
        lst[i, :] = r + n @ latmat
    return lst[np.argmin(norm(lst, axis=1))]
    # rv = rv.T
    # rmin = 10000.0
    # xcdi = xdc - 2.0 * rv[0, 0]
    # ycdi = ydc - 2.0 * rv[1, 0]
    # zcdi = zdc - 2.0 * rv[2, 0]
    # # !
    # # !  Loop over unit cells
    # # !
    # for ii in range(-1, 2):
    #     xcdi = xcdi + rv[0, 0]
    #     ycdi = ycdi + rv[1, 0]
    #     zcdi = zcdi + rv[2, 0]
    #     xcdj = xcdi - 2.0 * rv[0, 1]
    #     ycdj = ycdi - 2.0 * rv[1, 1]
    #     zcdj = zcdi - 2.0 * rv[2, 1]
    #     for jj in range(-1, 2):
    #         xcdj = xcdj + rv[0, 1]
    #         ycdj = ycdj + rv[1, 1]
    #         zcdj = zcdj + rv[2, 1]
    #         xcrd = xcdj - 2.0 * rv[0, 2]
    #         ycrd = ycdj - 2.0 * rv[1, 2]
    #         zcrd = zcdj - 2.0 * rv[2, 2]
    #         for kk in range(-1, 2):
    #             xcrd = xcrd + rv[0, 2]
    #             ycrd = ycrd + rv[1, 2]
    #             zcrd = zcrd + rv[2, 2]
    #             r = xcrd * xcrd + ycrd * ycrd + zcrd * zcrd
    #             if r <= rmin:
    #                 rmin = r
    #                 xdc = xcrd
    #                 ydc = ycrd
    #                 zdc = zcrd
    # return xdc, ydc, zdc
               

def join_silane(portsil, alk, port, tpairs, bpairs):
    from compound import Port
    
    for j, x in enumerate([tpairs[1], bpairs[1]]):
        pos1 = np.mean([i.pos for i in x], axis=0)
        p = Port(pos=pos1, orientation=x[1].pos-x[0].pos)
        port.add(p, expand=0)
        port.remove(x)
        calk = deepcopy(alk)
        portsil.add(calk)
        portsil.force_overlap(calk, calk[-1], p)
        if j:
            calk.reflect(pos1, [0, 0, 1])
            tmp1 = calk.particles()
            for z in tmp1:
                if z.name == 'C':
                    tmp = list(portsil.bond_graph.neighbors(z))
                    while tmp:
                        y = tmp[0]
                        if y.name == 'H':
                            portsil.remove(y)
                        tmp.remove(y)
                portsil.remove(z)


def hybrid_silane_portlandite():
    from compound import compload, Port, Compound
    port = compload('/home/ali/ongoing_research/polymer4/14_mbuild/port.cif')
    port.add_bond([port[4], port[2]])
    port.add_bond([port[1], port[3]])
    
    port.supercell([[2, 0, 0], [1, 2, 0], [0, 0, 1]])
    port.xyz += np.array([0, 0, 2.2])
    port.wrap_atoms()
    
    topo = [x for x in port.particles(0) if x.name == 'O' and x.pos[2] > 2.2]
    boto = [x for x in port.particles(0) if x.name == 'O' and x.pos[2] < 2.2]
    
    tpairs = pairs_in_rcut(port, combinations(topo, 2), rcut=3.6)
    bpairs = pairs_in_rcut(port, combinations(boto, 2), rcut=3.6)
    
    # we remove for a single pair
    hparts1, _ = port.neighbs(tpairs[1], port.particles_by_name('H'), 1.2)
    hparts2, _ = port.neighbs(bpairs[1], port.particles_by_name('H'), 1.2)
    port.remove(flatten([hparts1, hparts2]))
    
    alk = dimer(6)
    alk.latmat = 50 * np.eye(3)
    
    
    pos1 = np.mean(alk['sil'].xyz[[6, 7]], 0)
    pos2 = alk['sil'][8].pos
    alk.add(Port(pos=pos1, orientation=alk['sil'].xyz[6] - alk['sil'].xyz[7]), expand=0)
    
    portsil = Compound()
    portsil.latmat = port.latmat
    portsil.latmat[2, 2] = 13
    portsil.add(port, expand=0)
    
    join_silane(portsil, alk, port, tpairs, bpairs)
    
    portsil.xyz_with_ports += [0, 0, 4]
    for x in portsil.particles(0):
        x.name = re.sub('[0-9]', '', x.name)    
    return portsil
                
def hessian_to_dynmat(comp, hess):
    '''multiply by 1/sqrt(m_im_j)'''
    n = comp.n_particles() * 3
    dynmat = np.zeros([n, n])
    parts = comp.particles_label_sorted()
    masses = np.repeat(np.array([x.type['mass'] for x in parts], dtype=float), 3)
    for i in range(n):
        for j in range(n):
            dynmat[i, j] = 1/np.sqrt(masses[i] * masses[j]) * hess[i, j]
    return dynmat

def bond_derivs(rs, deriv=1):
    ''' seeing bond length as generalized coordinate, this gives dq/dx_i or d^2q/dx_i*dx_j'''
    rs = np.array(rs)
    del_r = rs[0] - rs[1]
    ndr = norm(del_r)
    ndrv = del_r/ndr #normalized dr vector
    if deriv == 1:
        return np.vstack([ndrv, -ndrv])

    hessian = np.zeros([6, 6])
    if deriv == 2:
        tmp = (np.outer(ndrv, ndrv) - np.eye(3))/ndr
        for i in range(2):
            for j in range(2):
                hessian[3*i:3*(i+1), 3*j:3*(j+1)] = (-1)**(i == j) * tmp
        return hessian

def dihed_derivs(p1,p2,p3,p4,d='dum'):
    '''from Karplus' paper'''
    p1,p2,p3,p4 = map(np.array,[p1,p2,p3,p4])

    F = p1 - p2
    G = p2 - p3
    H = p4 - p3
    A = cross(F, G)
    B = cross(H, G)

    phi = np.arccos(dot(A,B)/norm(A)/norm(B))
    if d==0:
        return phi

    nG = norm(G); nB = norm(B)
    nAsq = norm(A)**2
    nBsq = norm(B)**2

    dfg = dot(F,G)
    dhg = dot(H,G)

    dphi_dr1 = -nG/nAsq*A
    dphi_dr2 = nG/nAsq*A + dfg/nAsq/nG*A - dhg/nBsq/nG*B
    dphi_dr3 = dhg/nBsq/nG*B - dfg/nAsq/nG*A - nG/nBsq*B
    dphi_dr4 = nG/nBsq*B

    dphi_dr = np.array([*dphi_dr1, *dphi_dr2, *dphi_dr3, *dphi_dr4])

    s1 = range(0,3); s2 = range(3,6); s3 = range(6,9);

    dT_dr=np.array([[1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1]])
    d2phi_dT2 = np.zeros([9, 9])
    cga = cross(G,A)
    cfa = cross(F,A)
    chb = cross(H,B)
    cgb = cross(G,B)
    t1 = einsum('i,j', A, cga) + einsum('i,j', cga, A)
    t2 = einsum('i,j', A, cfa) + einsum('i,j', cfa, A)
    t3 = einsum('i,j', B, cgb) + einsum('i,j', cgb, B)
    t4 = einsum('i,j', B, chb) + einsum('i,j', chb, B)

    d2phi_dT2[np.ix_(s1, s1)] = nG/nAsq**2*t1 #eq 32

    d2phi_dT2[np.ix_(s2, s2)] = 1/2/nG**3/nAsq*t1 + dfg/nG/nAsq**2*t2 -  1/2/nG**3/nBsq*t3 - dhg/nG/nBsq**2*t4 #eq 44

    d2phi_dT2[np.ix_(s3, s3)] = -nG/nBsq**2*t3 #eq 33

    d2phi_dT2[np.ix_(s1, s2)] = -1/nG/nAsq**2*(nG**2*einsum('i,j', cfa, A) + dfg*einsum('i,j', A, cga)) #eq 38
    d2phi_dT2[np.ix_(s2, s1)] = d2phi_dT2[np.ix_(s1, s2)]

    d2phi_dT2[np.ix_(s2, s3)] = 1/nG/nBsq**2*(nG**2*einsum('i,j', chb, B) + dhg*einsum('i,j', B, cgb)) #eq 39
    d2phi_dT2[np.ix_(s3, s2)] = d2phi_dT2[np.ix_(s2, s3)]

    # kk = np.zeros([12,12])
    # for i in range(12):
    #     for j in range(12):
    #         tmp = 0
    #         for k in range(9):
    #             for l in range(9):
    #                 tmp += dT_dr[k, i]*dT_dr[l, j]*d2phi_dT2[k, l]
    #         kk[i, j] = tmp
    tmp = einsum('ij,kl', dT_dr, dT_dr)
    return phi, dphi_dr, einsum('ij,ikjl',d2phi_dT2,tmp)

def make_unique(original_list):
    unique_list, idx = [], []
    for i, obj in enumerate(original_list):
        if obj not in unique_list:
            unique_list.append(obj)
            idx.append(i)
    return unique_list, idx

def gulp_out_coords(direc='gulp.out'):
    coords = shell('''awk '/Fractional c/{for (i=0;i<6;i++) getline; flg=1}/---/{flg=0}flg{print $4, $5, $6}' '''+direc)
    lat = shell('''awk '/Cartesian lattice/{getline;getline; for (i=0;i<3;i++) {print;getline}}' '''+direc)
    n = shell('''awk '/Number of irred/ {print $6}' '''+direc).decode('utf-8')
    n = int(n)
    return (np.genfromtxt(io.BytesIO(coords)) @ (lat:=np.genfromtxt(io.BytesIO(lat)))) [-n:], lat 

def plane_normal(p1, p2, p3):
    ''' normal to plane from three points '''
    p1, p2, p3 = map(np.array, [p1, p2, p3])
    v1 = p1 - p2
    v2 = p3 - p2
    vec = np.cross(v1, v2)
    return vec/norm(vec)


def ab_to_es(A, B, reverse=0):
    '''
  #          A       B                            /  / s \^a    / s \^b \
  #  U(r) = ---  -  ---           =       4 * e * |  |---|   -  |---|   |
  #         r^a     r^b                           \  \ r /      \ r /   /

  #    A =  4*e*s^a
  #    B = -4*e*s^b

  For A,B>0:
  #    s = (A/B)^(1/(a-b))
  #    e = -B / (4*s^b)
  #  Setting B=0.0001, and A=1.0   (for sigma=2^(-1/6))
  #  ---> e=2.5e-09, s=4.641588833612778
  #  This is good enough for us.  (The well depth is 2.5e-9 which is almost 0)'''
    if reverse:
        e, s = A, B
        A = 4*e*s**12
        B = 4*e*s**6
    else:
        s = (A/B)**(1/6)
        e = B/(4*s**6)

    return (A, B) if reverse else (e, s)

def get_2d_mat(inpstr):
    aa = inpstr.strip().split('\n')
    return np.array([x.split() for x in aa], dtype=float)

def b_to_mat(inp, dt=float):
    return np.genfromtxt(io.BytesIO(inp), dtype=dt)

def txt_to_mat(inp, dt=float):
    return np.genfromtxt(io.StringIO(inp), dtype=dt)

def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
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


def alkane(n, num_caps=0):
    from compound import Compound, Port
    from scipy.spatial.transform import Rotation as R

    ch2 = Compound(name='ch', names=['C', 'H', 'H'], pos=[[0,0.5935,0],[0,1.2513,0.8857],[ 0,1.2513,-0.8857]])

    dr1, dr2 = [-0.64315, -0.42845, 0], [ 0.64315, -0.42845, 0]
    ch2.add(Port(anchor=ch2['C0'], orientation=dr1, pos=ch2['C0'].pos+dr1))
    ch2.add(Port(anchor=ch2['C0'], orientation=-np.array(dr2), pos=ch2['C0'].pos+dr2))
    ch2.add_bond([[ch2['C0'], ch2['h0']],[ch2['c0'], ch2['h1']]])
    alk = Compound(name='alkane')
    alk.add(deepcopy(ch2), expand=0)
    for i in range(n-1):
        c_ch2 = deepcopy(ch2)
        alk.add(c_ch2, expand=0)
        alk.force_overlap(c_ch2, c_ch2[f'p!{2*i+2}'], alk.sel(f'p!{2*i+1}')[0], rotate_ang=0 if np.mod(i, 2) else 180)


    ch3 = Compound(names=['C', 'H', 'H', 'H'],
                         pos=[[-0.0000,   0.0000,   0.0000],
                              [ 0.6252,   0.6252,   0.6252],
                              [-0.6252,  -0.6252,   0.6252],
                              [-0.6252,   0.6252,  -0.6252]])
    
    [ch3.add_bond(x) for x in product([ch3[0]], ch3.particles()[1:])]

    dr = np.array([ 0.6252,  -0.6252,  -0.6252])/2
    ch3.add(Port(anchor=ch3['C'],pos=dr, orientation=-dr), expand=0)
    if n == 0:
        alk = Compound(name='alkane')
        cp1, cp2 = deepcopy(ch3), deepcopy(ch3)
        alk.add([cp1, cp2])
        alk.force_overlap(cp1, cp1['p!'], cp2['p!'], flip=0)
        return alk
    for i in range(num_caps):
        cp = deepcopy(ch3)
        alk.add(cp)
        alk.force_overlap(cp, cp['p!'], alk['p!'][-2], flip=0)

    alk.unique_names()
    return alk

def dimer(nchain):
    from compound import Compound, compload, Port
    from funcs import alkane
    alksil = Compound(name='alksil', latmat=np.eye(3)*50)

    sil = compload('/home/ali/ongoing_research/polymer4/22_lammps_py_fitting/sil.mol2')  # type: Compound
    sil.name = 'sil0'
    c2, c1 = [6.3001, 4.1639, 6.0570], [8.5253, 8.0180, 6.0570]
    tmp1, tmp2 = c1 - sil['Si0'].pos, c2 - sil['Si1'].pos
    sil.add(Port(sil['Si0'], pos=sil['Si0'].pos+tmp1/2, orientation=[0,0,1], name='p!1'), expand=0)
    sil.add(Port(sil['Si1'], pos=sil['Si1'].pos+tmp2/2, orientation=[0,0,1], name='p!2'), expand=0)
    alksil.add(sil, expand=0)

    alkane = alkane(nchain)
    alkane['p!0'].orientation = alkane['C2'].pos - alkane['C0'].pos
    alksil.add(deepcopy(alkane), expand=0)
    alksil.add(deepcopy(alkane), expand=0)

    alksil.force_overlap(alksil['alkane0'], alksil['p!2'], alksil['p!0'], rotate_ang=90)
    alksil.force_overlap(alksil['alkane1'], alksil['p!4'], alksil['p!1'], rotate_ang=90)
    alksil.remove([x for x in alksil.particles(1) if '!' in x.name])
    alksil.unique_names()
    return alksil

def trimer(nchain):
    from compound import Compound, compload, Port
    alksil = dimer(nchain)
    sel = alksil.sel(['alkane0', 'o4', 'o3', 'o0', 'si0'], 0)
    alksil.add((sel:=deepcopy(sel)))
    alksil.add(Port(alksil['o4']))
    alksil.add(Port(alksil['o0']))
    alksil.force_overlap(sel, alksil['p!0'], alksil['p!1'])
    sel.rotate_around([0, 0, 1], 60, pnt=sel['o5'])
    alksil.remove(['o5', 'h0'])
    # sil['sil'].remove(sil['sil']['H2'])
    # sil['sil'].remove(sil['sil']['O2'])
    # new = deepcopy(sil) #type: Compound
    # # new.rotate_around(sil['sil']['Si2'].pos - sil['sil']['O3'].pos, 120, pnt=sil['sil']['Si2'].pos)
    # new.rotate_around([0, 0, 1], 120, pnt=sil['sil']['Si2'].pos)

    # new.remove(new['alkane'][1])
    # tmp, _ = new['sil'].neighbs(sil.particles(), rcut=.5)
    # sil.remove(flatten(tmp))

    # sil.add(new)
    return alksil

def regng(a, b):
    from regex_engine import generator as rgen
    return rgen().numerical_range(a, b).replace('^', '').replace('$', '')

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
        # lmp = lammps(cmdargs=['-echo', 'none', '-screen', 'none'])
        lmp = lammps()
        lmp.file(inp_fle)
        x = lmp.extract_atom("x", 3)
        for j, p in enumerate(pos):
        	for k, c in enumerate(p):
        		x[j][k] = c
        lmps.append(lmp)
        lmp.command('run 0')
        frc_lmps[i] = np.ctypeslib.as_array(lmp.extract_atom("f", 3).contents, shape=configs[0].shape)
    return lmps[0] if len(lmps)==1 else lmps, frc_lmps

def ev_to_kcalpmol(input, reverse=0):
    from parmed import unit as u
    return u.AVOGADRO_CONSTANT_NA._value**-1 * (
		input * u.kilocalorie).in_units_of(u.elementary_charge * u.volts) if reverse else u.AVOGADRO_CONSTANT_NA._value * (
		input * u.elementary_charge * u.volts).in_units_of(u.kilocalorie)

def get_file_num(outdir):
    ''' if the file name is sth like out23, it returns 23'''
    files = next(os.walk(f'./{outdir}'))[-1]
    try:
        return np.sort(np.array([int(re.match('[^_a-zA-Z]*', x).group())
                            for x in files if re.match('[^_a-zA-Z]*', x).group() != '']))[-1] + 1
    except:
	    return 1

def vasp_output(direc, save_npy=0, get_freqs=1, get_eigvecs=1):
    '''direc: outcar directory
    latmat, coords, frcs, derv2, freqs, eigr'''

    pth = os.path.join(direc,"OUTCAR")

    with open(pth) as f:
        latmat, coords, frcs, types = [], [], [], []
        fcnt = 0
        for ln in f:
            if 'direct lattice vectors' in ln:
                latmat.append([])
                for i in range(3):
                    latmat[-1].append(get_cols(next(f), range(1, 4)))

            if 'position of ions in cartesian coordinates' in ln:
                tmp = ''
                for ln in f:
                    if not ln.strip():
                        break
                    tmp += ln
                coords.append(get_2d_mat(tmp))
                n = len(coords[-1])
                freqs = [None] * 3*n
                eigr = np.zeros([3*n, 3*n])

            if 'TOTAL-FORCE' in ln:
                next(f)
                tmp = ''
                for i in range(n):
                    tmp += next(f)
                tmp = get_2d_mat(tmp)
                coords.append(tmp[:, :3])
                frcs.append(tmp[:, 3:])

            if 'cm-1' in ln:
                freqs[fcnt] = re.search(r'[0-9\.\-]*\s+(?=cm-1)', ln).group()
                if 'f/i=' in ln:
                    freqs[fcnt] = '-' + freqs[fcnt]
                next(f)
                tmp = ''
                for i in range(n):
                    tmp += next(f)
                eigr[:, fcnt] = get_2d_mat(tmp)[:, 3:].flatten()
                fcnt += 1

            if 'SECOND DERIVATIVES' in ln:
                f = islice(f, 2, None)
                tmp = ''
                for x in f:
                    if not x.strip():
                        break
                    tmp += x[6:]
                derv2 = get_2d_mat(tmp)

            if 'ions per type' in ln:
                nt = np.array(re.search(r'\d.*', ln).group().split(), dtype=int)
            if 'VRHFIN' in ln:
                types.append(re.search(r'(?<==).*?(?=:)', ln).group())

    tm = []
    for x in types:
        tm.append(amass(x))
    elems = []
    masses = []
    for i, x in enumerate(types):
        for j in range(nt[i]):
            elems.append(x)
            masses.append(tm[i])
    return np.array(latmat, dtype=float), coords, frcs, derv2, np.array(freqs, dtype=float), eigr, elems, masses

    # n = shell(f' grep -oP -m 1 \'(?<=NIONS =)\\s*[0-9]*\' {os.path.join(direc,"OUTCAR")} ')
    # n = int(n.decode())
    # dft_pos_frc = shell(
    #     f'''awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg{{print $1, $2, $3, $4, $5, $6}}' {direc}/OUTCAR ''')
    # dft_pos_frc = np.vstack([np.fromstring(x, np.float, sep=' ')
    #                     for x in dft_pos_frc.decode().split('\n')[:-1]])
    # dft_pos_frc = cut_array2d(dft_pos_frc, [len(dft_pos_frc) // n, 1])
    #
    # return np.array([x[:, :3] for x in dft_pos_frc]), np.array([x[:, 3:] for x in dft_pos_frc])

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

def get_vasp_hessian(direc):
    n = check_output(f' grep -oP -m 1 \'(?<=NIONS =)\\s*[0-9]*\' {os.path.join(direc,"OUTCAR")} ', shell=1)
    n = int(n.decode('utf-8'))
    hessian = check_output(f''' awk '/^  1X/{{flg=1}};/^ Eigen/{{flg=0}}flg{{$1="";print $0}}' {direc}/OUTCAR ''', shell=1)
    return -(u.AVOGADRO_CONSTANT_NA._value *(np.genfromtxt(io.BytesIO(hessian)) *
                                            u.elementary_charge * u.volts).in_units_of(u.kilocalorie))._value

def  unit_perps(c0, c1, c2):
    #This gives the vector in the plane A,B,C and perpendicular to A to B

    diff_AB = c1 - c0
    u_AB = diff_AB / norm(diff_AB)

    diff_CB = c1 - c2
    u_CB = diff_CB / norm(diff_CB)

    cross_product = np.cross(u_CB, u_AB)
    u_N = cross_product / norm(cross_product)

    u_PA = np.cross(u_N, u_AB)
    u_PC = -np.cross(u_N, u_CB)

    return u_PA / norm(u_PA), u_PC / norm(u_PC), math.degrees(math.acos(np.dot(u_AB, u_CB)))

def repnth(s_in, n, sub, sep='(\s+)'):
    '''replace nth col of string with sth elase
    sub: a string or a list of strings'''
    if not isinstance(n, Iterable):
        sub = [sub]
        n = [n]
    s_in = re.sub('^\s+', '', s_in)
    res = re.split(sep, s_in)
    for i, x in zip(n, sub):
        res[2*(i-1)] = str(x)
    return  ''.join(res)

def gulp_elem_names(ln):
    tmp = re.findall('^.*?\s+(?=[\d-])', ln)[0].split()
    n = len(tmp)
    if n == 3:
        tmp = list(np.roll(tmp, 1))
    return tmp, n

def direc_list(direc, depth):
    files = []
    dir2 = os.path.join(direc, "/".join("*" * (depth+1)))
    return glob(dir2)
    
def get_cols(s, n, sep='\s+'):
    s = re.sub('^\s+', '', s)
    if not isinstance(n, Iterable):
        n = [n]
    n = [x-1 for x in n]
    return itemgetter(*n)(re.split(sep, s))
    
    
def write_pprofile(titles='', pp=None):
    pp.disable()
    pp.dump_stats('pprofile_tmp')
    txt = open('pprofile_tmp').read()
    if isinstance(titles, str):
        titles = [titles]
    out = ''
    for x in titles:
        out += re.findall(f'(?s){x}.*?(?=File:)', txt)[0] + '\n\n next func\n'
        
    open('pprofile', 'w').write(out)
    # Path('pprofile_tmp').unlink(missing_ok=1)
    
    
def equivalence_classes(objects, func):
    ''' func should take two arguments of objects and return true or fals'''
    objects = list(objects)
    out = []
    used = np.zeros(len(objects))

    for i, obj in enumerate(objects):
        if not used[i]:
            eq_class = [(i,x) for i, x in enumerate(objects) if func(x, obj)]
            used[[x[0] for x in eq_class]] = 1
            out.append(eq_class)
    return [[y[1] for y in x] for x in out], [[y[0] for y in x] for x in out]


def gulp_output_params(f, read_current=0):
    nparms,flg = [],0
    types = []
    for x in f:
        if 'Variables :' in x:
            [next(f) for _ in range(4)]
            for y in f:
                if '---' in y: break
                nparms.append([get_cols(y, 2)])

        if 'Parameter        P' in x:
            [next(f) for _ in range(2)]
            for i, y in enumerate(f):
                if '---' in y: break
                nparms[i] = [nparms[i][0], get_cols(y, 3)]
                types.append(' '.join(re.findall('[A-Za-z].*?(?=\s)', y)))

        if read_current and 'Current' in x:
            s = len(nparms)
            for i, y in enumerate(f):
                if flg:
                    nparms[i][1] = get_cols(y, 2)
                else:
                    nparms[i].append(get_cols(y, 2))
                if i+1 == s: break
            flg = 1

    return np.array(nparms, dtype=float), types


class cdict(UserDict):
    ''' you give it something like ['Ca', 'Si'] and it treats it as Counter'''
    def __setitem__(self, key, value):
        if isinstance(key, list):
            self.data[frozenset(Counter(key).items())] = value
        else:
            self.data[key] = value

    def __getitem__(self, key):
        if isinstance(key, list):
            return self.data[frozenset(Counter(key).items())]
        else:
            return self.data[key]

def gen_grimme():
    # indices = [20, 1, 8, 14, 6]
    # conversion = 10.364425449557316
    atoms = ['Ca', 'H', 'O', 'Si', 'C']

    cc = dict(zip(atoms, [10.8, 0.14, 0.7, 9.23, 1.75]))
    r0 = dict(zip(atoms, [1.474, 1.001, 1.342, 1.716, 1.452]))

    for x in atoms:
        cc[x] = (cc[x] * u.joule * u.nanometer ** 6).in_units_of(u.angstrom ** 6 * u.kilocalorie)._value

    n = len(atoms)
    rr0, ccc, ats = [], [], []
    for i in range(n):
        for j in range(i, n):
            ats.append(f'{atoms[i]} {atoms[j]}')

            rr0.append(r0[t1] + r0[t2])
            ccc.append(np.sqrt(cc[t1] * cc[t2]))
    return ats, rr0, ccc


def get_flags(ln):
    return re.findall(r'\d\s+\d(?=\s+#)', ln)[0].split()


def update_gulp_fit(direc, jj, skip=0, sub_flag='1 0 ', repneg='', bnd='',read_current=0, create=0, prt_fle=0):
    ''' direc: path/to/{gulp.in,gulp[n].out} -- only path/to/gulp[n]
    creates a new file gulp[n+1].in with the updated params if gulp[n].out exists. If not, leaves with a warning.
    skip: if it's only gulp3.in and no gulp3.out there, just return
    if jj=='', then reads gulp.in and gulp.out and makes gulp1.in
    repneg: replace negative stiffness with value
    bnd: the bond that is okey if it's fixed
    read_current: try to read current params from out even if the run is not completed
    prt_fle: print messages into the file 'fle'
    '''

    try:
        f = open(os.path.join(direc,f'gulp{jj}.out'))
    except FileNotFoundError:
        if skip: 
            print(os.path.join(direc,f'gulp{jj}.out'), ' does not exist.')
            return
        if jj:  #example: you have gulp1.in but no gulp1.out and you want to replace gulp1.in based on data from gulp.in and gulp.out
            jj = int(jj)
            jj = jj - 1 if jj>1 else ''
            f = open(os.path.join(direc, f'gulp{jj}.out'))
        else:
            return  #there is just gulp.in in the folder -- no update needed

    nparms, _ = gulp_output_params(f, read_current)
    if len(nparms) == 0 or nparms.shape[1] == 1:
        print(f'{direc}/gulp{jj}.out seems to be incomplete')
        return

    f1 = open(os.path.join(direc,f'gulp{jj}.in'), 'r')
    pth = os.path.join(direc,f'gulp{jj+1 if jj else 1}.in')
    flg1,cnt,flg2,flg3 = 1,0,0,0 #flg2:0:remove the new gulp\d.in, 1:keep cause negative, 2:keep cause the run had fixed params
    out = ''
    sv_bnd = []
    for x in f1:
        if 'tol' in x:
            x=''
        if 'print 5' in x:
            flg1 = 0
        if 'r0' in x:
            oflags = re.findall('\d\s+\d\s+(?=#)', x)[0]
            flags = oflags
            tmp, n = gulp_elem_names(x)
            tmp = tuple(tmp)
            if Counter(tmp) == Counter(bnd):
                flags = '0 0 '
                x = re.sub('\d\s+\d\s+(?=#)', flags, x)
                out += x
                continue
            tol = .2 if n==2 else 8 # tolerance for rejection or acceptance of r0
            for iflg in range(2):  #loop for k and r0
                if get_cols(oflags, iflg+1) == '1':
                    nparms[cnt, :] *= 1 if iflg else 23.06054972536769
                    if repneg and nparms[cnt, 1] < 0:
                        nparms[cnt, 1] = repneg
                        flags = '0 1 '
                        flg2 = 1
                    if iflg:
                        r0 = float(re.findall('(?<=r0:).*?(?=[\s□])',x)[0])
                        if not np.isclose(r0, nparms[cnt, 1], atol=tol, rtol=0):
                            nparms[cnt, 1] = nparms[cnt, 0]
                            sv_bnd.append(tmp)
                            if nparms[cnt, 1] < r0:
                                nparms[cnt, 1] = r0 - tol - .01
                            else:
                                nparms[cnt, 1] = r0 + tol + .01
                        if not bnd:
                            flags = '1 0 '
                    elif not bnd:
                        flags = '0 1 '

                    x = repnth(x, n+iflg+1, nparms[cnt, 1])
                    x = re.sub('\n', (' r0' if iflg else ' k') + f':{nparms[cnt, 0]:.2f}\n', x)
                    cnt += 1
                elif iflg == 0:
                    flags = '1 0 '
                    flg3 += 1


            if sub_flag:
                flags = sub_flag
            x = re.sub('\d\s+\d\s+(?=#)', flags, x)

        if 'shift ' in x:
            x = repnth(x, 2, nparms[-1, 1])
        # out = f'{out}{x}'
        out = ''.join([out, x])
    if flg1:
        out += '\nprint 5'

    if (flg2 or flg3) and create:
        open(pth, 'w').write(out)
    msg = ''
    if (prt := flg2 or flg3 or sv_bnd):
        msg += f'{direc}/gulp{jj}.out '
    msg += (f' has negative stiffness' if flg2 else '') +\
          (f' & was ran with fixed params' if flg3 else '') + \
          (f' & has problematic r0 for {sv_bnd}' if sv_bnd else '') + ('\n' if prt else '')
    if prt_fle:
        open('fle', 'w').write(msg)
    else:
        if msg:
            print(msg,end="")

    return out
        

def max_file_num(direc, pat = '(?<=gulp)\d+(?=.in)'):
    direc = str(direc)
    files = os.listdir(direc) #  next(os.walk(direc))[-1]
    nums = []
    for x in files:
        y = re.findall(pat, x)
        if y:
            nums.append(int(y[0]))
    if not nums:
        return ''
    else:
        return np.max(nums)


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
    file = shell(command)
    return np.genfromtxt(io.BytesIO(file))

def get_gulp_freqs_eigs(direc='gulp.out'):
    n = shell('''awk '/Number of irred/ {print $6;exit}' ''' + direc).decode()
    n = int(n)
    f = open(direc)
    reg = re.compile('^\s*Frequency\s*(?=-?\d)')
    reg2 = re.compile('(?<=[xyz]).*')
    freqs = ''
    flg = 0
    eigstr = ['' for _ in range(3*n)]
    for x in f:
        if reg.match(x):
            freqs += re.findall(r'(?<=Frequency).*', x)[0]
            f = islice(f, 6, None)
            tmp = ''
            for i, y in enumerate(f):
                if y == '\n':
                    break
                eigstr[i] += reg2.findall(y)[0]

    return txt_to_mat(freqs), txt_to_mat('\n'.join(eigstr))

def angle_between_vecs(v1, v2, degrees=1):
    v1, v2 = map(np.array,[v1, v2])
    ang = np.arccos(v1 @ v2/(norm(v1) * norm(v2)))

    return np.degrees(ang) if degrees else ang


