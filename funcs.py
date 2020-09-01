import numpy as np
import os, re, subprocess, math, io
from lammps import lammps
from numpy import dot, cross, sqrt, einsum
from numpy.linalg import norm, svd
from math import degrees, radians
from collections.abc import Iterable
from parmed import unit as u
from subprocess import check_output
from molmod.ic import bond_length, bend_angle, dihed_angle
from itertools import combinations, chain

# from lib.recipes.alkane import Alkane
# from compound import Compound, compload, Port
from copy import deepcopy


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


def mat_prt(mat, frm = '{:12.5f} '):
    print(*[(len(x)*frm).format(*x) for x in mat], sep='\n')
    
def brace_to_curly(lst):
    return str(lst).replace('[','{').replace(']', '}')


def gulp_hessian(comp, direc='gulp.out'):
    '''extract hessian from gulp output'''
    n = comp.n_particles()*3

    with  open('gulp.out', 'r') as f:
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
    A = np.cross(F, G)
    B = np.cross(H, G)

    phi = np.arccos(dot(A,B)/norm(A)/norm(B))
    if d==0:
        return phi

    nG = norm(G); nB = norm(B)
    nAsq = norm(A)**2
    nBsq = norm(B)**2

    dfg = dot(F,G)
    dhg = dot(H,G)
    dfg = dot(F,G)

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
    t1 = np.einsum('i,j', A, cga) + np.einsum('i,j', cga,A)
    t2 = np.einsum('i,j', A, cfa) + np.einsum('i,j', cfa,A)
    t3 = np.einsum('i,j', B, cgb) + np.einsum('i,j', cgb,B)
    t4 = np.einsum('i,j', B, chb) + np.einsum('i,j', chb,B)

    d2phi_dT2[np.ix_(s1, s1)] = nG/nAsq**2*t1 #eq 32

    d2phi_dT2[np.ix_(s2, s2)] = 1/2/nG**3/nAsq*t1 + dfg/nG/nAsq**2*t2 -  1/2/nG**3/nB**2*t3 - dhg/nG/nB**4*t4 #eq 44

    d2phi_dT2[np.ix_(s3, s3)] = -nG/nBsq**2*t3 #eq 33

    d2phi_dT2[np.ix_(s1, s2)] = -1/nG/nAsq**2*(nG**2*np.einsum('i,j', cfa, A) + dfg*np.einsum('i,j', A, cga)) #eq 38
    d2phi_dT2[np.ix_(s2, s1)] = d2phi_dT2[np.ix_(s1, s2)]

    d2phi_dT2[np.ix_(s2, s3)] = 1/nG/nBsq**2*(nG**2*np.einsum('i,j', chb, B) + dhg*np.einsum('i,j', B, cgb)) #eq 39
    d2phi_dT2[np.ix_(s3, s2)] = d2phi_dT2[np.ix_(s2, s3)]

    # kk = np.zeros([12,12])
    # for i in range(12):
    #     for j in range(12):
    #         tmp = 0
    #         for k in range(9):
    #             for l in range(9):
    #                 tmp += dT_dr[k, i]*dT_dr[l, j]*d2phi_dT2[k, l]
    #         kk[i, j] = tmp
    tmp = np.einsum('ij,kl', dT_dr, dT_dr)
    return phi, dphi_dr, np.einsum('ij,ikjl',d2phi_dT2,tmp)

def gulp_out_coords(direc='gulp.out'):
    coords = check_output('''awk '/Final fractional/{for (i=0;i<6;i++) getline; flg=1}/---/{flg=0}flg{print $4, $5, $6}' '''+direc, shell=1)
    lat = check_output('''awk '/Cartesian lattice/{getline;getline; for (i=0;i<3;i++) {print;getline}}' '''+direc, shell=1)
    n = check_output('''awk '/Number of irred/ {print $6}' '''+direc, shell=1).decode('utf-8')
    n = int(n)
    return (np.genfromtxt(io.BytesIO(coords)) @ np.genfromtxt(io.BytesIO(lat))) [-n:]

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

def get_2d_arr(inpstr):
    aa = inpstr.split('\n')
    return np.array([x.split() for x in aa], dtype=float)

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


def alkane(n, num_caps=0):
    from compound import Compound, Port
    from scipy.spatial.transform import Rotation as R

    ch2 = Compound(name='ch2', names=['C', 'H', 'H'], pos=[[0,0.5935,0],[0,1.2513,0.8857],[ 0,1.2513,-0.8857]])

    dr1, dr2 = [-0.64315, -0.42845, 0], [ 0.64315, -0.42845, 0]
    ch2.add(Port(anchor=ch2['C'], orientation=dr1, pos=ch2['C'].pos+dr1),expand=0)
    ch2.add(Port(anchor=ch2['C'], orientation=dr2, pos=ch2['C'].pos+dr2),expand=0)
    [ch2.add_bond(x) for x in [[ch2[0], ch2[1]],[ch2[0], ch2[2]]]]
    alk = Compound(name='alkane')
    alk.add(deepcopy(ch2))
    for i in range(n-1):
        c_ch2 = deepcopy(ch2)
        alk.force_overlap(c_ch2, c_ch2['p!'][0], alk[-1], flip=1)
        alk.add(c_ch2)

    ch3 = Compound(names=['C', 'H', 'H', 'H'],
                         pos=[[-0.0000,   0.0000,   0.0000],
                              [ 0.6252,   0.6252,   0.6252],
                              [-0.6252,  -0.6252,   0.6252],
                              [-0.6252,   0.6252,  -0.6252]])
    from itertools import product
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


    return alk

def dimer(nchain):
    from compound import Compound, compload, Port
    from funcs import alkane
    alksil = Compound(name='alksil')

    sil = compload('/home/ali/ongoing_research/polymer4/22_lammps_py_fitting/sil.mol2', compound=Compound(name='sil'),
                   infer_hierarchy=0)  # type: Compound
    c2, c1 = [6.3001, 4.1639, 6.0570], [8.5253, 8.0180, 6.0570]
    sil[['O3', 'Si2']]
    tmp1, tmp2 = c1 - sil['Si1'].pos, c2 - sil['Si2'].pos
    sil.add(Port(sil['Si1'], pos=sil['Si1'].pos+tmp1/2, orientation=sil['Si2'].pos-sil['O3'].pos, name='p1!'), expand=0)
    sil.add(Port(sil['Si2'], pos=sil['Si2'].pos+tmp2/2, orientation=sil['Si2'].pos-sil['O3'].pos, name='p2!'), expand=0)

    alksil.add(sil, expand=0)
    alkane = alkane(nchain)
    alkane['p!'][-1].orientation = alkane['C'][-3].pos - alkane['C'][-1].pos
    alk1, alk2 = deepcopy(alkane), deepcopy(alkane)
    alksil.add([alk1, alk2], expand=0)
    alksil.force_overlap(alk1, alk1['p!'][-1], sil['p1!'], rotate_ang=90)
    alksil.force_overlap(alk2, alk2['p!'][-1], sil['p2!'], rotate_ang=90)

    alksil.remove([x for x in alksil.particles(1) if '!' in x.name])
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
    files = next(os.walk(f'./{outdir}'))[-1]
    try:
        return np.sort(np.array([int(re.match('[^_a-zA-Z]*', x).group())
                            for x in files if re.match('[^_a-zA-Z]*', x).group() != '']))[-1] + 1
    except:
	    return 1

def coords_forces_from_outcar(direc, save_npy=0):
    '''direc: outcar directory
       nump: number of particles'''

    n = check_output(f' grep -oP -m 1 \'(?<=NIONS =)\\s*[0-9]*\' {os.path.join(direc,"OUTCAR")} ', shell=1)
    n = int(n.decode('utf-8'))
    dft_pos_frc = check_output(
        f'''awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg{{print $1, $2, $3, $4, $5, $6}}' {direc}/OUTCAR ''', shell=1)
    dft_pos_frc = np.vstack([np.fromstring(x, np.float, sep=' ')
                        for x in dft_pos_frc.decode('utf-8').split('\n')[:-1]])
    dft_pos_frc = cut_array2d(dft_pos_frc, [len(dft_pos_frc) // n, 1])

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
    file = check_output(command, shell=1)
    return np.genfromtxt(io.BytesIO(file))

def get_gulp_freqs(direc='gulp.out'):
    command = f"awk '/Frequencies/{{getline;getline;flg=1}};NF==0&&flg{{print \"sep\";flg=0}};flg' {direc}"
    file = check_output(command, shell=1).decode('utf-8').split('sep')[-2].replace('\n',' ')
    return np.fromstring(file, sep=' ')

def angle_between_vecs(v1, v2, degrees=1):
    v1, v2 = map(np.array,[v1, v2])
    ang = np.arccos(v1 @ v2/(norm(v1) * norm(v2)))

    return np.degrees(ang) if degrees else ang


