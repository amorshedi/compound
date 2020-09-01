import numpy as np
from collections.abc import Iterable
from collections import OrderedDict, defaultdict
from ismember import ismember
from copy import deepcopy
import os, math#, sys, tempfile, importlib
# from warnings import warn
from itertools import chain, compress, combinations
#
#
import parmed as pmd
from scipy.spatial.transform import Rotation as R
from parmed.periodic_table import AtomicNum, element_by_name, Mass, Element
from funcs import *
from openbabel import pybel,openbabel
import networkx as nx
from molmod.ic import bond_length, bend_angle, dihed_angle

# from pymatgen.core import Structure
#
# import mdtraj.geometry as geom
# from pymatgen import Lattice, Structure
# # from openbabel import pybel
#
from numpy.linalg import inv, norm, det, matrix_rank
# from coordinate_transform import _translate, _rotate,AxisTransform,RigidTransform, unit_vector, angle
# from collections import defaultdict
from orderedset import OrderedSet

#to do list:
# ForceField from foyer should be in this file somehow
# nonbond types should have somethig similar to bonds_typed (maybe add 'nonbond' to each atom)

def from_pymatgen(structure):
    """ convert pymatgen structure to compound """

    comp = Compound(names=[x.specie.name for x in structure.sites], pos=structure.cart_coords)
    comp.latmat = structure.lattice.matrix
    comp.latmat.setflags(write=1)

    return comp


def compload(filename_or_object, relative_to_module=None, compound=None, coords_only=False,
         use_parmed=False, smiles=False,
         infer_hierarchy=True, **kwargs):
    """Load a file or an existing topology into an mbuild compound.

    Files are read using the MDTraj package unless the `use_parmed` argument is
    specified as True. Please refer to http://mdtraj.org/1.8.0/load_functions.html
    for formats supported by MDTraj and https://parmed.github.io/ParmEd/html/
    readwrite.html for formats supported by ParmEd.

    Parameters
    ----------
    filename_or_object : str, mdtraj.Trajectory, parmed.Structure, mbuild.Compound,
            pybel.Molecule
        Name of the file or topology from which to load atom and bond information.
    compound : mb.Compound, optional, default=None
        Existing compound to load atom and bond information into.
    coords_only : bool, optional, default=False
        Only load the coordinates into an existing compound.
    use_parmed : bool, optional, default=False
        Use readers from ParmEd instead of MDTraj.
    smiles: bool, optional, default=False
        Use Open Babel to parse filename as a SMILES string
        or file containing a SMILES string.
    infer_hierarchy : bool, optional, default=True
        If True, infer hierarchy from chains and residues
    **kwargs : keyword arguments
        Key word arguments passed to mdTraj for loading.

    Returns
    -------
    compound : mb.Compound

    """
    import mdtraj as md
    from pymatgen.core import Structure
    # If compound doesn't exist, we will initialize one
    if compound is None:
        compound = Compound()

    # First check if we are loading from an existing parmed or trajectory structure
    type_dict = {
        pmd.Structure:compound.from_parmed,
        md.Trajectory:compound.from_trajectory,
    }#pybel.Molecule:compound.from_pybel

    if isinstance(filename_or_object, Compound):
        return filename_or_object
    for type in type_dict:
        if isinstance(filename_or_object, type):
            type_dict[type](filename_or_object,coords_only=coords_only,
                    infer_hierarchy=infer_hierarchy, **kwargs)

    # Handle mbuild2 *.py files containing a class that wraps a structure file
    # in its own folder. E.g., you build a system from ~/foo.py and it imports
    # from ~/bar/baz.py where baz.py loads ~/bar/baz.pdb.
    script_path = relative_to_module
    if relative_to_module:
        # script_path = os.path.realpath(
        #     sys.modules[relative_to_module].__file__)
        file_dir = os.path.dirname(script_path)
        filename_or_object = os.path.join(file_dir, filename_or_object)

    # Handle the case of a xyz and json file, which must use an internal reader
    extension = os.path.splitext(filename_or_object)[-1]
    if extension == '.json':
        compound = compound_from_json(filename_or_object)

    if extension == '.xyz' and not 'top' in kwargs:
        if coords_only:
            tmp = read_xyz(filename_or_object)
            if tmp.n_particles != compound.n_particles:
                raise ValueError('Number of atoms in {filename_or_object} does not match'
                                 ' {compound}'.format(**locals()))

            ref_and_compound = zip(tmp._particles(include_ports=False),
                                   compound.particles(include_ports=False))
            for ref_particle, particle in ref_and_compound:
                particle.pos = ref_particle.pos
        else:
            compound = compound.read_xyz(filename_or_object)

    if extension == '.sdf':
        pybel_mol = pybel.readfile('sdf', filename_or_object)
        # pybel returns a generator, so we grab the first molecule of a list of len 1
        # Raise ValueError user if there are more molecules
        pybel_mol = [i for i in pybel_mol]
        compound.from_pybel(pybel_mol[0])

    if 'POSCAR' in filename_or_object or 'CONTCAR' in filename_or_object:
        return from_pymatgen(Structure.from_file(filename_or_object))

    if extension == '.cif':
        return from_pymatgen(Structure.from_file(filename_or_object)) #type: Compound

    if use_parmed:
        # warn(
        #     "use_parmed set to True.  Bonds may be inferred from inter-particle "
        #     "distances and standard residue templates!")
        structure = pmd.load_file(filename_or_object, structure=True, **kwargs)
        compound.from_parmed(structure, coords_only=coords_only,
                infer_hierarchy=infer_hierarchy)

    elif smiles:
        # First we try treating filename_or_object as a SMILES string
        try:
            mymol = pybel.readstring("smi", filename_or_object)
        # Now we treat it as a filename
        except(OSError, IOError):
            # For now, we only support reading in a single smiles molecule,
            # but pybel returns a generator, so we get the first molecule
            # and warn the user if there is more

            mymol_generator = pybel.readfile("smi", filename_or_object)
            mymol_list = list(mymol_generator)
            if len(mymol_list) == 1:
                mymol = mymol_list[0]
            else:
                mymol = mymol_list[0]
                warn("More than one SMILES string in file, more than one SMILES "
                     "string is not supported, using {}".format(mymol.write("smi")))

        # We create a temporary directory and mol2 file that will created from the smiles string
        # A ParmEd structure and subsequenc mbuild2 compound will be created from this mol2 file
        tmp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(tmp_dir, 'smiles_to_mol2_intermediate.mol2')
        mymol.make3D()
        compound = Compound()
        compound.from_pybel(mymol, infer_hierarchy=infer_hierarchy)

    else:
        traj = md.load(filename_or_object, **kwargs)
        # traj.xyz*=10
        if extension in ['.pdb', '.mol2']:
            traj.xyz*=10

        compound.from_trajectory(traj, frame=-1, infer_hierarchy=infer_hierarchy)

    return compound


def clone(existing_compound, clone_of=None, root_container=None):
    """A faster alternative to deepcopying.

    Does not resolve circular dependencies. This should be safe provided
    you never try to add the top of a Compound hierarchy to a
    sub-Compound.

    Parameters
    ----------
    existing_compound : mb.Compound
        Existing Compound that will be copied

    Other Parameters
    ----------------
    clone_of : dict, optional
    root_container : mb.Compound, optional

    """
    if clone_of is None:
        clone_of = dict()

    newone = existing_compound._clone(clone_of=clone_of,
                                      root_container=root_container)
    existing_compound._clone_bonds(clone_of=clone_of)
    return newone

def from_lmps_data(direc='data.dat'):
    from pymatgen.io.lammps.data import LammpsData
    aa = LammpsData.from_file(direc)
    comp = from_pymatgen(aa.structure)

    _,_,bb=aa.disassemble(guess_element=1)

    for x in bb[0].topologies['Bonds']:
        comp.add_bond(comp[x])
    
    return comp
    

class Compound(object):
    """A building block in the mbuild2 hierarchy.

    The design of Compound follows the Composite design pattern (Gamma, Erich; Richard Helm; Ralph Johnson; John
    M. Vlissides (1995). Design Patterns: Elements of Reusable Object-Oriented
    Software. Addison-Wesley. p. 395. ISBN 0-201-63361-2.), with Compound being
    the composite, and Particle playing the role of the primitive (leaf) part,
    where Particle is in fact simply an alias to the Compound class.

    Compound maintains a list of children (other Compounds contained within).
    Compound has built-in support for copying
    and deepcopying Compound hierarchies, enumerating particles or bonds in the
    hierarchy, proximity based searches, visualization, I/O operations, and a
    number of other convenience methods.

    Parameters
    ----------
    name : str, optional, default=self.__class__.__name__
        The type of Compound.
    pos : np.ndarray, shape=(3,), dtype=float, optional, default=[0, 0, 0]
        The position of the Compound in Cartestian space
    """

    def __init__(self, name='comp', names=None, pos=None):

        self.name = name
        pos = np.array(pos)
        if not np.array_equal(pos, None) and (pos.ndim == 1 or pos.shape[0] == 1):
            self._pos = pos.flatten()
        else:
            self._pos = np.zeros(3)

        self.parent = None
        self.children = OrderedSet()
        self.referrers = set()
        self.type = []

        self.bond_graph = None

        self._latmat = np.eye(3)
        self.box = Box(self)
        
        if names:
            for x, y in zip(names, pos):
                self.add(Compound(name=x, pos=y))
            
    
    @property
    def latmat(self):
        return self._latmat
    
    @latmat.setter
    def latmat(self, vec):
        tmp = np.array(vec).flatten()
        if tmp.size == 1:
            self._latmat = tmp * np.eye(3)
        elif tmp.size == 3:
            self._latmat = np.diag(tmp)
        else:
            self._latmat = vec
        
    # @staticmethod
    def reflect(self, pp, pvec):
        """ image of a point with respect to a plane
        p: point
        pp: point in the plane
        pvect: normal to the plane"""

        pp, pvec = map(np.array, [pp, pvec])
        n = pvec / norm(pvec)
        t = np.sum((pp - self.xyz_with_ports) * n, axis=1)
        self.xyz_with_ports += 2 * np.array([x * n for x in t])

        for x in [z for z in self.particles(1) if '!' in z.name]:
            tmp = Compound(pos=x.orientation)
            tmp.reflect(pp, pvec)
            x.orientation = tmp.pos
    
    def particles(self,ports=0):
        return list(self._particles(ports))
            

    def _particles(self,ports=0):
        """Return all Particles of the Compound.

        Parameters
        ----------
        include_ports : bool, optional, default=False
            Include port particles

        Yields
        -------
        mb.Compound
            The next Particle in the Compound

        """
        if not self.children:
            if '!' in self.name and not ports:
                yield from []
            else:
                yield self

        for child in self.children:
            for leaf in child.particles(ports):
                yield leaf

    def successors(self):
        """Yield Compounds below self in the hierarchy.

        Yields
        -------
        mb.Compound
            The next Particle below self in the hierarchy

        """
        if not self.children:
            return
        for part in self.children:
            # Parts local to the current Compound.
            yield part
            # Parts further down the hierarchy.
            for subpart in part.successors():
                yield subpart

    def bonds_angles_index(self, sorted=1):
        bond_list = []
        angle_list = []
        nlst = self.particles_label_sorted() if sorted else self.particles()
        for i, bond in enumerate(self.bonds_typed, 1):
            bond_list.append([nlst.index(j) for j in bond])
        for i, angle in enumerate(self.angles_typed, 1):
            angle_list.append([nlst.index(j) for j in angle])
        return np.array(bond_list), np.array(angle_list)

    def n_particles(self,ports=0):
        """Return the number of Particles in the Compound.

        Returns
        -------
        int
            The number of Particles in the Compound

        """
        if not self.children:
            return 1
        else:
            return sum(1 for _ in self.particles(ports))

    def _contains_only_ports(self):
        for part in self.children:
            if not part.port_particle:
                return False
        return True

    def ancestors(self):
        """Generate all ancestors of the Compound recursively.

        Yields
        ------
        mb.Compound
            The next Compound above self in the hierarchy

        """
        if self.parent is not None:
            yield self.parent
            for ancestor in self.parent.ancestors():
                yield ancestor

    @property
    def root(self):
        """The Compound at the top of self's hierarchy.

        Returns
        -------
        mb.Compound
            The Compound at the top of self's hierarchy

        """
        parent = None
        for parent in self.ancestors():
            pass
        if parent is None:
            return self
        return parent
    
    
    def prt_bonding(self):
        print('number of bonds: ', len(self.bonds_typed))
        print('number of angles: ', len(self.angles_typed))
        print('number of diheds: ', len(self.propers_typed), '\n')

    def analytic_hessian(self, qlist=None):
        nlst = self.particles_label_sorted()
        n = len(nlst)
        hessian = np.zeros([3 * n, 3 * n])
        qlist = qlist if qlist else [*self.bonds_typed, *self.angles_typed, *self.propers_typed]
        tjac = np.zeros([len(qlist), 3 * self.n_particles()])
        for cnt, x in enumerate(qlist):
            idx = []
            for v in x:
                tmp = nlst.index(v)
                idx.extend([3 * tmp, 3 * tmp + 1, 3 * tmp + 2])
            # idx = np.sort(idx)
            if len(x) == 2:
                fun, img = bond_length, self.closest_img_bond
                k = self.bonds_typed[x]['k']
                eq = float(self.bonds_typed[x]['length'])
            elif len(x) == 3:
                fun, img = bend_angle, self.closest_img_angle
                k = self.angles_typed[x]['k']
                eq = math.radians(float(self.angles_typed[x]['angle']))
                # aa = angle_derivs(*img(*x),2)
                fun = angle_derivs
                #
                # mag2, fderiv2, sderiv2 = angle_derivs(*img(*x), 2)
            else:
                fun, img = dihed_angle, self.closest_img_dihed
                k = self.propers_typed[x]['k']
                eq = math.radians(float(self.propers_typed[x]['phi']))
                k = float(k)
                coords = img(*x)
                v1 = coords[0] - coords[1]
                v2 = coords[2] - coords[1]
                v3 = coords[3] - coords[2]
                cv1 = cross(v1, v2)
                cv2 = cross(v2, v3)
                # mag2, fderiv2, sderiv2 = dihed_derivs(*[v.pos for v in x])
                # sderiv3 = dihed_derivs2(*[v.pos for v in x])
            k = float(k)


            mag, fderiv, sderiv = fun(img(*x), 2)
            if mag < 0:
                mag *= -1
                sderiv *= -1

            sderiv = sderiv.reshape([-1, 3 * len(x)])
            fderiv = fderiv.flatten()
            if len(x) == 4 and (np.isclose(mag, 0, atol=1e-15, rtol=0) or
                                np.isclose(mag, np.pi, atol=1e-15, rtol=0) or np.allclose(cv1, 0, atol=.0001) or np.allclose(cv2, 0, atol=.0001)):
                fderiv = np.zeros(fderiv.shape)
                sderiv = np.zeros(sderiv.shape)
            hessian[np.ix_(idx, idx)] += 2 * k * (np.einsum('i,j', fderiv, fderiv) + (mag - eq) * sderiv)
        return hessian

    def particles_by_name(self, name):
        """Return all Particles of the Compound with a specific name

        Parameters
        ----------
        name : str
            Only particles with this name are returned

        Yields
        ------
        mb.Compound
            The next Particle in the Compound with the user-specified name

        """
        for particle in self.particles(1):
            if particle.name == name:
                yield particle

    def add(self, new_child, expand=True,name=None,replace=False,
            inherit_periodicity=True, reset_rigid_ids=True):
        """Add a part to the Compound.

        Parameters
        ----------
        new_child : mb.Compound or list-like of mb.Compound
            The object(s) to be added to this Compound.

        """
        # Support batch add via lists, tuples and sets.
        if (isinstance(new_child, Iterable) and
                not isinstance(new_child, str)):
            for child in new_child:
                self.add(child, reset_rigid_ids=reset_rigid_ids)
            return

        if not isinstance(new_child, Compound):
            raise ValueError('Only objects that inherit from Compound '
                             'can be added to Compounds. You tried to add '
                             '"{}".'.format(new_child))
        if name:
            self.name = name

        # Create children and labels on the first add operation
        if self.children is None:
            self.children = OrderedSet()

        if expand and new_child.children:
            # if new_child.parent is not None:
            #     raise MBuildError('Part {} already has a parent: {}'.format(
            #         new_child, new_child.parent))
            for x in new_child.children:
                x.parent = self
            self.children |= new_child.children
        else:
            self.children.add(new_child)
            new_child.parent = self

        if self.root.bond_graph is None:
            self.root.bond_graph = new_child.bond_graph
        elif new_child.bond_graph:
            self.root.bond_graph = nx.compose(self.root.bond_graph, new_child.bond_graph)

        new_child.bond_graph = None

    def remove(self, objs_to_remove):
        """ Cleanly remove children from the Compound.

        Parameters
        ----------
        objs_to_remove : mb.Compound or list of mb.Compound
            The Compound(s) to be removed from self

        """
        # Preprocessing and validating input type
        if not hasattr(objs_to_remove, '__iter__'):
            if objs_to_remove not in self:
                return
            objs_to_remove = [objs_to_remove]
        objs_to_remove = set(objs_to_remove)

        for obj in objs_to_remove:
            if obj.root.bond_graph: #take care of removing bonds
                for p in obj.particles(0):
                    if obj.root.bond_graph.has_node(p):
                        obj.root.bond_graph.remove_node(p)
            obj.parent.children.remove(obj)
            if not obj.parent.children and obj.parent.parent: #all of its children are removed. Remove itself
                obj.parent.parent.children.remove(obj.parent)

    def add_bond(self, bonds):
        """Add a bond between two Particles.

        Parameters
        ----------
        bonds : indexable object, length=2, dtype=mb.Compound
            The pair of Particles to add a bond between

        """
        if self.root.bond_graph is None:
            self.root.bond_graph = nx.Graph()
        
        if isinstance(bonds[0], Iterable):
            for x in bonds:
                self.add_bond(x)
            return

        self.root.bond_graph.add_edge(bonds[0], bonds[1])

    def generate_bonds(self, atoms_a, atoms_b, dmin, dmax):
        """Add Bonds between all pairs of types a/b within [dmin, dmax].

        Parameters
        ----------
        dmin : float
            The minimum distance between Particles for considering a bond
        dmax : float
            The maximum distance between Particles for considering a bond

        """
        atoms_a, atoms_b = map(list, [atoms_a, atoms_b])
        neighbs, dists = self.neighbs(atoms_a, atoms_b, rcut=dmax)

        for i, p in enumerate(atoms_a):
            for x, d in zip(neighbs[i], dists[i]):
                if d > dmin:
                    self.add_bond([p, x])


    def remove_bond(self, particle_pair):
        """Deletes a bond between a pair of Particles

        Parameters
        ----------
        particle_pair : indexable object, length=2, dtype=mb.Compound
            The pair of Particles to remove the bond between

        """
        from port import Port
        if self.root.bond_graph is None or not self.root.bond_graph.has_edge(
                *particle_pair):
            warn("Bond between {} and {} doesn't exist!".format(*particle_pair))
            return
        self.root.bond_graph.remove_edge(*particle_pair)
        bond_vector = particle_pair[0].pos - particle_pair[1].pos
        if np.allclose(bond_vector, np.zeros(3)):
            warn("Particles {} and {} overlap! Ports will not be added."
                 "".format(*particle_pair))
            return

    @property
    def pos(self):
        if not self.children:
            return self._pos
        else:
            return self.center()

    @pos.setter
    def pos(self, value):
        value = np.array(value)
        if not self.children:
            self._pos = value
        else:
            raise Exception('Cannot set position on a Compound that has'
                              ' children.')

    @property
    def xyz(self):
        """Return all particle coordinates in this compound.

        Returns
        -------
        pos : np.ndarray, shape=(n, 3), dtype=float
            Array with the positions of all particles.
        """
        if not self.children:
            pos = np.expand_dims(self._pos, axis=0)
        else:
            arr = np.fromiter(chain.from_iterable(
                particle.pos for particle in self.particles(0)), dtype=float)
            pos = arr.reshape((-1, 3))
        return pos

    def particles_label_sorted(self, ports=0):
        atom_names = OrderedSet([atom.name for atom in self.particles(ports)])
        return [x for label in atom_names for x in self.particles(ports) if x.name == label]

    @property
    def xyz_label_sorted(self):
        return np.array([x.pos for x in self.particles_label_sorted(0)])

    @xyz_label_sorted.setter
    def xyz_label_sorted(self, value):
        for i, p in enumerate(self.particles_label_sorted(0)):
            p._pos = value[i]

    @property
    def xyz_with_ports(self):
        """Return all particle coordinates in this compound including ports.

        Returns
        -------
        pos : np.ndarray, shape=(n, 3), dtype=float
            Array with the positions of all particles and ports.

        """
        return np.array([x.pos for x in self.particles(1)])

    @xyz.setter
    def xyz(self, arrnx3):
        """Set the positions of the particles in the Compound, excluding the Ports.

        This function does not set the position of the ports.

        Parameters
        ----------
        arrnx3 : np.ndarray, shape=(n,3), dtype=float
            The new particle positions

        """

        for atom, coords in zip(self.particles(0), arrnx3):
            atom.pos = coords

    @xyz_with_ports.setter
    def xyz_with_ports(self, arrnx3):
        """Set the positions of the particles in the Compound, including the Ports.

        Parameters
        ----------
        arrnx3 : np.ndarray, shape=(n,3), dtype=float
            The new particle positions

        """

        for atom, coords in zip(self.particles(1), arrnx3):
            atom.pos = coords

    def center(self,include_ports=0):
        """The cartesian center of the Compound based on its Particles.

        Returns
        -------
        np.ndarray, shape=(3,), dtype=float
            The cartesian center of the Compound based on its Particles

        """

        if np.all(np.isfinite(self.xyz )):
            if include_ports == 1:
                return np.mean(self.xyz_with_ports, axis=0)
            else:
                return np.mean(self.xyz, axis=0)

    def lmps_minimize(self, path = None, fixed_atoms=None, movie_freq=100, buff=0):
        fixed_atoms = list(fixed_atoms) if fixed_atoms else None
        self.write_lammpsdata(path if path else 'data.dat',buff=buff)
        from lammps import lammps
        lmps = lammps()
        lmps.file('runfile')
        # lmps,_ = get_lmps(np.array([self.xyz_label_sorted]))
        commands = []
        if fixed_atoms:
            commands.append('group fixed id '+' '.join([str(self.particles_label_sorted().index(x)+1) for x in fixed_atoms]))
            commands.append('fix fx fixed setforce 0 0 0 ')
        commands.extend([f'dump 10 all custom {movie_freq} movie.lammpstrj element x y z',
                            'dump_modify 10 element ' +' '.join([x['name'] for x in self.ff.atom_types])+' sort id',
                            'minimize 1e-8 1e-8 10000 100000'])
        lmps.commands_list(commands)
        os.remove('log.lammps')
        return np.array(lmps.gather_atoms('x', 1,3)).reshape([-1, 3])#np.ctypeslib.as_array(lmps.extract_atom("x", 3).contents, shape=self.xyz.shape)


    def vmd(self,ports=1,atoms=None,label_sorted=1, mol2=0, types=0, commands='pbc box\n '):
        """ see the system in vmd """
        nme = 'out.mol2' if mol2 else 'out.lammpstrj'
        f = open('txt', 'w')
        # os.system('>txt;# echo "mol new {out.lammpstrj} type {lammpstrj} first 0 last -1 step 1 waitfor -1" >> txt')
        if atoms is not None:
            if not hasattr(atoms, '__iter__'):
                atoms = [atoms]
            else:
                atoms = list(atoms)
            idx = np.in1d(self.particles() if mol2 else self.particles_label_sorted(),atoms)
            f.write('mol addrep 0\n'
                    'mol modstyle 1 0 VDW 0.3 30.\n'
                    'mol modselect 1 0 index '+(np.sum(idx)*'{} ').format(*np.nonzero(idx)[0])+'\n')
        
        f.write(f'pbc set {{{self.box.a} {self.box.b} {self.box.c} {self.box.alph} {self.box.bet} {self.box.gam}}} -all\n')
        f.write(commands)
        f.close()
        self.save(nme) if mol2 else self.write_lammpstrj()
        os.system(f'/home/ali/software/vmd-1.9.4a43/bin2/vmd {nme} -e txt')
        os.system(f'rm {nme} txt')

    def write_lammpstrj(self, fle='out', unit_set='real',ports=1, label_sorted=1, types=0):
        """Write one or more frames of data to a lammpstrj file.
           types: if the forcefield types were assigned
        """
        fle=open(fle+'.lammpstrj','w')
        fle.write('ITEM: TIMESTEP\n'
                  '0\n'
                  'ITEM: NUMBER OF ATOMS\n'
                  f'{self.n_particles(ports)}\n'
                  'ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
        b = self.box
        fle.write(f'{b.xlo_bound} {b.xhi_bound} {b.xy}\n'
                  f'{b.ylo_bound} {b.yhi_bound} {b.xz}\n'
                  f'{b.zlo_bound} {b.zhi_bound} {b.yz}\n\n')
        # --- begin body ---
        fle.write('ITEM: ATOMS element x y z\n')
        for part in self.particles_label_sorted(ports) if label_sorted else self.particles(ports):
            c = part.pos
            fle.write(f"{part.type['name'] if types else part.name} {c[0]} {c[1]} {c[2]}\n")


    def write_poscar(self, path='.', lattice_const = 1,
                     fixed_atoms=[],coord='cartesian'):
        """
        Parameters
        ----------
        filename: str
            Path of the output file
        coord: str, default = 'cartesian', other option = 'direct'
            Coordinate style of atom positions
        """
        atom_names = OrderedSet([atom.name for atom in self.particles(0)])

        if coord == 'direct':
            for atom in structure.atoms:
                atom.xx = atom.xx / lattice_constant
                atom.xy = atom.xy / lattice_constant
                atom.xz = atom.xz / lattice_constant

        path.replace('POSCAR', '')
        with open(os.path.join(path, 'POSCAR'), 'w') as data:
            data.write(f' - created by mBuild\n'
                                f'{lattice_const}\n')
            for x in self.latmat:
                data.write(f'{x[0]} {x[1]} {x[2]} \n')
            data.write((len(atom_names)*'{} ').format(*atom_names)+'\n')
            part_names = [x.name for x in self.particles(0)]
            data.write((len(atom_names)*'{} ').format(*[part_names.count(y) for y in atom_names])+'\n')

            if fixed_atoms:
                data.write('Selective Dyn\n')
            data.write(coord+'\n')

            for p in self.particles_label_sorted():
                data.write('{0:.13f} {1:.13f} {2:.13f} '.format(*p.pos))
                if fixed_atoms:
                    data.write(' '.join(3*['F' if p in fixed_atoms else 'T'])+'\n')
                else:
                    data.write('\n')

    def particles_in_range(
            self,
            compound,
            dmax,
            max_particles=20,
            particle_kdtree=None,
            particle_array=None):
        """Find particles within a specified range of another particle.

        Parameters
        ----------
        compound : mb.Compound
            Reference particle to find other particles in range of
        dmax : float
            Maximum distance from 'compound' to look for Particles
        max_particles : int, optional, default=20
            Maximum number of Particles to return
        particle_kdtree : mb.PeriodicCKDTree, optional
            KD-tree for looking up nearest neighbors. If not provided, a KD-
            tree will be generated from all Particles in self
        particle_array : np.ndarray, shape=(n,), dtype=mb.Compound, optional
            Array of possible particles to consider for return. If not
            provided, this defaults to all Particles in self

        Returns
        -------
        np.ndarray, shape=(n,), dtype=mb.Compound
            Particles in range of compound according to user-defined limits

        See Also
        --------
        periodic_kdtree.PerioidicCKDTree : mBuild implementation of kd-trees
        scipy.spatial.ckdtree : Further details on kd-trees

        """
        if particle_kdtree is None:
            particle_kdtree = PeriodicCKDTree(
                data=self.xyz, bounds=self.periodicity)
        _, idxs = particle_kdtree.query(
            compound.pos, k=max_particles, distance_upper_bound=dmax)
        idxs = idxs[idxs != self.n_particles]
        if particle_array is None:
            particle_array = np.array(list(self.particles()))
        return particle_array[idxs]

    def closest_img_bond(self, p1, p2):
        ''' input two particles, get back two coordinates '''
        c0 = p1.pos
        c1 = self.neighbs(c0, p2, closest_img=1)[0]
        return c0, c1

    def closest_img_angle(self, p1, p2, p3):
        ''' input three particles, get back three coordinates '''
        c1, c2 = self.closest_img_bond(p1, p2)
        c3 = self.neighbs(c2, p3, closest_img=1)[0]
        return c1, c2, c3

    def closest_img_dihed(self, p1, p2, p3, p4):
        ''' input three particles, get back three coordinates '''
        c1, c2, c3 = self.closest_img_angle(p1, p2, p3)
        c4 = self.neighbs(c3, p4, closest_img=1)[0]
        return c1, c2, c3, c4


    def closest_img(self, atom, atoms):
        ''' among atoms and their images, who is closest to atom'''
        if isinstance(atoms, Iterable):
            atoms = list(atoms)
        else:
            atoms = [atoms]

        coords_set2 = np.vstack([x.xyz for x in atoms])

        r = (atom.pos - coords_set2) @ inv(self.latmat)
        rfrac = r - np.round(r)
        dists = norm(rfrac @ self.latmat,axis=1)

        return atoms[np.argmin(dists)]

    def neighbs(self,set1,set2=None,rcut=2,slf=0, closest_img=0):
        ''' set1: either a list of atoms or a list of coordinates'''
        if isinstance(set1, Iterable):  #this way to be able to handle generators coming for set1
            set1 = list(set1)
            if not isinstance(set1[0], Compound):
                set1 = [set1]
        else:
            set1 = [set1]

        if set2:
            if isinstance(set2, Iterable):
                set2 = list(set2)
            else:
                set2 = [set2]
        else:
            set2 = self.particles()

        coords_set1 = np.vstack([x.xyz for x in set1]) if isinstance(set1[0], Compound) else set1
        coords_set2 = np.vstack([x.xyz for x in set2]) if set2 else self.xyz  #self.xyz[[x in set2 for x in self.particles()]]

        idx = [[] for i in range(set1.__len__())]
        odists = deepcopy(idx)
        for cnt,coord in enumerate(coords_set1):
            r = (coord - coords_set2) @ inv(self.latmat)
            rfrac = r - np.round(r)
            dists = norm(rfrac @ self.latmat,axis=1)
            if closest_img:
                return coord - rfrac @ self.latmat

            tmp = dists<rcut
            for x, y in zip(compress(set2, tmp), compress(dists, tmp)):
                if slf or set1[cnt] != x:
                    idx[cnt].append(x)
                    odists[cnt].append(y)
            # [x for x in  if slf or set1[cnt]!=x])
            # odists.extend([dists[i] for i, x in enumerate(compress(set2, tmp)) ])

        return idx,odists

    def _update_port_locations(self, initial_coordinates):
        """Adjust port locations after particles have moved

        Compares the locations of Particles between 'self' and an array of
        reference coordinates.  Shifts Ports in accordance with how far anchors
        have been moved.  This conserves the location of Ports with respect to
        their anchor Particles, but does not conserve the orientation of Ports
        with respect to the molecule as a whole.

        Parameters
        ----------
        initial_coordinates : np.ndarray, shape=(n, 3), dtype=float
            Reference coordinates to use for comparing how far anchor Particles
            have shifted.

        """
        particles = list(self.particles(0))
        for port in self.all_ports():
            if port.anchor:
                idx = particles.index(port.anchor)
                shift = particles[idx].pos - initial_coordinates[idx]
                port.translate(shift)

    def _kick(self):
        """Slightly adjust all coordinates in a Compound

        Provides a slight adjustment to coordinates to kick them out of local
        energy minima.
        """
        xyz_init = self.xyz
        for particle in self.particles(0):
            particle.pos += (np.random.rand(3,) - 0.5) / 100
        self._update_port_locations(xyz_init)

    def energy_minimize(self, forcefield='UFF', steps=1000, **kwargs):
        """Perform an energy minimization on a Compound

        Default behavior utilizes Open Babel (http://openbabel.org/docs/dev/)
        to perform an energy minimization/geometry optimization on a
        Compound by applying a generic force field

        Can also utilize OpenMM (http://openmm.org/) to energy minimize
        after atomtyping a Compound using
        Foyer (https://github.com/mosdef-hub/foyer) to apply a forcefield
        XML file that contains valid SMARTS strings.

        This function is primarily intended to be used on smaller components,
        with sizes on the order of 10's to 100's of particles, as the energy
        minimization scales poorly with the number of particles.

        Parameters
        ----------
        steps : int, optional, default=1000
            The number of optimization iterations
        forcefield : str, optional, default='UFF'
            The generic force field to apply to the Compound for minimization.
            Valid options are 'MMFF94', 'MMFF94s', ''UFF', 'GAFF', and 'Ghemical'.
            Please refer to the Open Babel documentation (http://open-babel.
            readthedocs.io/en/latest/Forcefields/Overview.html) when considering
            your choice of force field.
            Utilizing OpenMM for energy minimization requires a forcefield
            XML file with valid SMARTS strings. Please refer to (http://docs.
            openmm.org/7.0.0/userguide/application.html#creating-force-fields)
            for more information.


        Keyword Arguments
        ------------
        algorithm : str, optional, default='cg'
            The energy minimization algorithm.  Valid options are 'steep',
            'cg', and 'md', corresponding to steepest descent, conjugate
            gradient, and equilibrium molecular dynamics respectively.
            For _energy_minimize_openbabel
        scale_bonds : float, optional, default=1
            Scales the bond force constant (1 is completely on).
            For _energy_minimize_openmm
        scale_angles : float, optional, default=1
            Scales the angle force constant (1 is completely on)
            For _energy_minimize_openmm
        scale_torsions : float, optional, default=1
            Scales the torsional force constants (1 is completely on)
            For _energy_minimize_openmm
            Note: Only Ryckaert-Bellemans style torsions are currently supported
        scale_nonbonded : float, optional, default=1
            Scales epsilon (1 is completely on)
            For _energy_minimize_openmm

        References
        ----------
        If using _energy_minimize_openmm(), please cite:
        .. [1] P. Eastman, M. S. Friedrichs, J. D. Chodera, R. J. Radmer,
               C. M. Bruns, J. P. Ku, K. A. Beauchamp, T. J. Lane,
               L.-P. Wang, D. Shukla, T. Tye, M. Houston, T. Stich,
               C. Klein, M. R. Shirts, and V. S. Pande.
               "OpenMM 4: A Reusable, Extensible, Hardware Independent
               Library for High Performance Molecular Simulation."
               J. Chem. Theor. Comput. 9(1): 461-469. (2013).


        If using _energy_minimize_openbabel(), please cite:
        .. [1] O'Boyle, N.M.; Banck, M.; James, C.A.; Morley, C.;
               Vandermeersch, T.; Hutchison, G.R. "Open Babel: An open
               chemical toolbox." (2011) J. Cheminf. 3, 33

        .. [2] Open Babel, version X.X.X http://openbabel.org, (installed
               Month Year)

        If using the 'MMFF94' force field please also cite the following:
        .. [3] T.A. Halgren, "Merck molecular force field. I. Basis, form,
               scope, parameterization, and performance of MMFF94." (1996)
               J. Comput. Chem. 17, 490-519
        .. [4] T.A. Halgren, "Merck molecular force field. II. MMFF94 van der
               Waals and electrostatic parameters for intermolecular
               interactions." (1996) J. Comput. Chem. 17, 520-552
        .. [5] T.A. Halgren, "Merck molecular force field. III. Molecular
               geometries and vibrational frequencies for MMFF94." (1996)
               J. Comput. Chem. 17, 553-586
        .. [6] T.A. Halgren and R.B. Nachbar, "Merck molecular force field.
               IV. Conformational energies and geometries for MMFF94." (1996)
               J. Comput. Chem. 17, 587-615
        .. [7] T.A. Halgren, "Merck molecular force field. V. Extension of
               MMFF94 using experimental data, additional computational data,
               and empirical rules." (1996) J. Comput. Chem. 17, 616-641

        If using the 'MMFF94s' force field please cite the above along with:
        .. [8] T.A. Halgren, "MMFF VI. MMFF94s option for energy minimization
               studies." (1999) J. Comput. Chem. 20, 720-729

        If using the 'UFF' force field please cite the following:
        .. [3] Rappe, A.K., Casewit, C.J., Colwell, K.S., Goddard, W.A. III,
               Skiff, W.M. "UFF, a full periodic table force field for
               molecular mechanics and molecular dynamics simulations." (1992)
               J. Am. Chem. Soc. 114, 10024-10039

        If using the 'GAFF' force field please cite the following:
        .. [3] Wang, J., Wolf, R.M., Caldwell, J.W., Kollman, P.A., Case, D.A.
               "Development and testing of a general AMBER force field" (2004)
               J. Comput. Chem. 25, 1157-1174

        If using the 'Ghemical' force field please cite the following:
        .. [3] T. Hassinen and M. Perakyla, "New energy terms for reduced
               protein models implemented in an off-lattice force field" (2001)
               J. Comput. Chem. 22, 1229-1242



        """
        tmp_dir = tempfile.mkdtemp()
        original = deepcopy(self)
        self._kick()
        self.save(os.path.join(tmp_dir, 'un-minimized.mol2'))
        extension = os.path.splitext(forcefield)[-1]
        openbabel_ffs = ['MMFF94', 'MMFF94s', 'UFF', 'GAFF', 'Ghemical']
        if forcefield in openbabel_ffs:
            self._energy_minimize_openbabel(tmp_dir, forcefield=forcefield,
                                            steps=steps, **kwargs)
        elif extension == '.xml':
            self._energy_minimize_openmm(tmp_dir, forcefield_files=forcefield,
                                         forcefield_name=None,
                                         steps=steps, **kwargs)
        else:
            self._energy_minimize_openmm(tmp_dir, forcefield_files=None,
                                         forcefield_name=forcefield,
                                         steps=steps, **kwargs)

        self.update_coordinates(os.path.join(tmp_dir, 'minimized.pdb'))

    def update_coordinates(self, filename, update_port_locations=0):
        """Update the coordinates of this Compound from a file.
        Parameters
        ----------
        filename : str
            Name of file from which to load coordinates. Supported file types
            are the same as those supported by load()
        update_port_locations : bool, optional, default=True
            Update the locations of Ports so that they are shifted along with
            their anchor particles.  Note: This conserves the location of
            Ports with respect to the anchor Particle, but does not conserve
            the orientation of Ports with respect to the molecule as a whole.
        See Also
        --------
        load : Load coordinates from a file
        """
        if update_port_locations:
            xyz_init = self.xyz
            self = compload(filename)
            self._update_port_locations(xyz_init)
        else:
            tmp = compload(filename)
            self.xyz = tmp.xyz

    def _energy_minimize_openmm(
            self,
            tmp_dir,
            forcefield_files=None,
            forcefield_name=None,
            steps=1000,
            scale_bonds=1,
            scale_angles=1,
            scale_torsions=1,
            scale_nonbonded=1):
        """ Perform energy minimization using OpenMM

        Converts an mBuild Compound to a ParmEd Structure,
        applies a forcefield using Foyer, and creates an OpenMM System.

        Parameters
        ----------
        forcefield_files : str or list of str, optional, default=None
            Forcefield files to load
        forcefield_name : str, optional, default=None
            Apply a named forcefield to the output file using the `foyer`
            package, e.g. 'oplsaa'. Forcefields listed here:
            https://github.com/mosdef-hub/foyer/tree/master/foyer/forcefields
        steps : int, optional, default=1000
            Number of energy minimization iterations
        scale_bonds : float, optional, default=1
            Scales the bond force constant (1 is completely on)
        scale_angles : float, optiona, default=1
            Scales the angle force constant (1 is completely on)
        scale_torsions : float, optional, default=1
            Scales the torsional force constants (1 is completely on)
        scale_nonbonded : float, optional, default=1
            Scales epsilon (1 is completely on)


        Notes
        -----
        Assumes a particular organization for the force groups
        (HarmonicBondForce, HarmonicAngleForce, RBTorsionForce, NonBondedForce)

        References
        ----------

        .. [1] P. Eastman, M. S. Friedrichs, J. D. Chodera, R. J. Radmer,
               C. M. Bruns, J. P. Ku, K. A. Beauchamp, T. J. Lane,
               L.-P. Wang, D. Shukla, T. Tye, M. Houston, T. Stich,
               C. Klein, M. R. Shirts, and V. S. Pande.
               "OpenMM 4: A Reusable, Extensible, Hardware Independent
               Library for High Performance Molecular Simulation."
               J. Chem. Theor. Comput. 9(1): 461-469. (2013).



        """
        foyer = import_('foyer')

        to_parmed = self.to_parmed()
        ff = foyer.Forcefield(forcefield_files=forcefield_files, name=forcefield_name)
        to_parmed = ff.apply(to_parmed)

        from simtk.openmm.app.simulation import Simulation
        from simtk.openmm.app.pdbreporter import PDBReporter
        from simtk.openmm.openmm import LangevinIntegrator
        import simtk.unit as u

        system = to_parmed.createSystem() # Create an OpenMM System
        # Create a Langenvin Integrator in OpenMM
        integrator = LangevinIntegrator(298 * u.kelvin, 1 / u.picosecond,
                                        0.002 * u.picoseconds)
        # Create Simulation object in OpenMM
        simulation = Simulation(to_parmed.topology, system, integrator)

        # Loop through forces in OpenMM System and set parameters
        for force in system.getForces():
            if type(force).__name__ == "HarmonicBondForce":
                for bond_index in range(force.getNumBonds()):
                    atom1, atom2, r0, k = force.getBondParameters(bond_index)
                    force.setBondParameters(bond_index,
                                            atom1, atom2,
                                            r0, k * scale_bonds)
                force.updateParametersInContext(simulation.context)

            elif type(force).__name__ == "HarmonicAngleForce":
                for angle_index in range(force.getNumAngles()):
                    atom1, atom2, atom3, r0, k = force.getAngleParameters(
                        angle_index)
                    force.setAngleParameters(angle_index,
                                             atom1, atom2, atom3,
                                             r0, k * scale_angles)
                force.updateParametersInContext(simulation.context)

            elif type(force).__name__ == "RBTorsionForce":
                for torsion_index in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(
                        torsion_index)
                    force.setTorsionParameters(
                        torsion_index,
                        atom1,
                        atom2,
                        atom3,
                        atom4,
                        c0 * scale_torsions,
                        c1 * scale_torsions,
                        c2 * scale_torsions,
                        c3 * scale_torsions,
                        c4 * scale_torsions,
                        c5 * scale_torsions)
                force.updateParametersInContext(simulation.context)

            elif type(force).__name__ == "NonbondedForce":
                for nb_index in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(
                        nb_index)
                    force.setParticleParameters(nb_index,
                                                charge, sigma,
                                                epsilon * scale_nonbonded)
                force.updateParametersInContext(simulation.context)

            elif type(force).__name__ == "CMMotionRemover":
                pass

            else:
                warn(
                    'OpenMM Force {} is '
                    'not currently supported in _energy_minimize_openmm. '
                    'This Force will not be updated!'.format(
                        type(force).__name__))

        simulation.context.setPositions(to_parmed.positions)
        # Run energy minimization through OpenMM
        simulation.minimizeEnergy(maxIterations=steps)
        reporter = PDBReporter(os.path.join(tmp_dir, 'minimized.pdb'), 1)
        reporter.report(
            simulation,
            simulation.context.getState(
                getPositions=True))

    def _energy_minimize_openbabel(self, tmp_dir, steps=1000, algorithm='cg',
                                   forcefield='UFF'):
        """Perform an energy minimization on a Compound

        Utilizes Open Babel (http://openbabel.org/docs/dev/) to perform an
        energy minimization/geometry optimization on a Compound by applying
        a generic force field.

        This function is primarily intended to be used on smaller components,
        with sizes on the order of 10's to 100's of particles, as the energy
        minimization scales poorly with the number of particles.

        Parameters
        ----------
        steps : int, optionl, default=1000
            The number of optimization iterations
        algorithm : str, optional, default='cg'
            The energy minimization algorithm.  Valid options are 'steep',
            'cg', and 'md', corresponding to steepest descent, conjugate
            gradient, and equilibrium molecular dynamics respectively.
        forcefield : str, optional, default='UFF'
            The generic force field to apply to the Compound for minimization.
            Valid options are 'MMFF94', 'MMFF94s', ''UFF', 'GAFF', and 'Ghemical'.
            Please refer to the Open Babel documentation (http://open-babel.
            readthedocs.io/en/latest/Forcefields/Overview.html) when considering
            your choice of force field.

        References
        ----------
        .. [1] O'Boyle, N.M.; Banck, M.; James, C.A.; Morley, C.;
               Vandermeersch, T.; Hutchison, G.R. "Open Babel: An open
               chemical toolbox." (2011) J. Cheminf. 3, 33
        .. [2] Open Babel, version X.X.X http://openbabel.org, (installed
               Month Year)

        If using the 'MMFF94' force field please also cite the following:
        .. [3] T.A. Halgren, "Merck molecular force field. I. Basis, form,
               scope, parameterization, and performance of MMFF94." (1996)
               J. Comput. Chem. 17, 490-519
        .. [4] T.A. Halgren, "Merck molecular force field. II. MMFF94 van der
               Waals and electrostatic parameters for intermolecular
               interactions." (1996) J. Comput. Chem. 17, 520-552
        .. [5] T.A. Halgren, "Merck molecular force field. III. Molecular
               geometries and vibrational frequencies for MMFF94." (1996)
               J. Comput. Chem. 17, 553-586
        .. [6] T.A. Halgren and R.B. Nachbar, "Merck molecular force field.
               IV. Conformational energies and geometries for MMFF94." (1996)
               J. Comput. Chem. 17, 587-615
        .. [7] T.A. Halgren, "Merck molecular force field. V. Extension of
               MMFF94 using experimental data, additional computational data,
               and empirical rules." (1996) J. Comput. Chem. 17, 616-641

        If using the 'MMFF94s' force field please cite the above along with:
        .. [8] T.A. Halgren, "MMFF VI. MMFF94s option for energy minimization
               studies." (1999) J. Comput. Chem. 20, 720-729

        If using the 'UFF' force field please cite the following:
        .. [3] Rappe, A.K., Casewit, C.J., Colwell, K.S., Goddard, W.A. III,
               Skiff, W.M. "UFF, a full periodic table force field for
               molecular mechanics and molecular dynamics simulations." (1992)
               J. Am. Chem. Soc. 114, 10024-10039

        If using the 'GAFF' force field please cite the following:
        .. [3] Wang, J., Wolf, R.M., Caldwell, J.W., Kollman, P.A., Case, D.A.
               "Development and testing of a general AMBER force field" (2004)
               J. Comput. Chem. 25, 1157-1174

        If using the 'Ghemical' force field please cite the following:
        .. [3] T. Hassinen and M. Perakyla, "New energy terms for reduced
               protein models implemented in an off-lattice force field" (2001)
               J. Comput. Chem. 22, 1229-1242
        """

        openbabel = self.import_('openbabel')

        for particle in self.particles(0):
            try:
                get_by_symbol(particle.name)
            except KeyError:
                raise Exception("Element name {} not recognized. Cannot "
                                  "perform minimization."
                                  "".format(particle.name))

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("mol2", "pdb")
        mol = openbabel.OBMol()

        obConversion.ReadFile(mol, os.path.join(tmp_dir, "un-minimized.mol2"))

        ff = openbabel.OBForceField.FindForceField(forcefield)
        if ff is None:
            raise Exception("Force field '{}' not supported for energy "
                              "minimization. Valid force fields are 'MMFF94', "
                              "'MMFF94s', 'UFF', 'GAFF', and 'Ghemical'."
                              "".format(forcefield))
        # warn(
        #     "Performing energy minimization using the Open Babel package. Please "
        #     "refer to the documentation to find the appropriate citations for "
        #     "Open Babel and the {} force field".format(forcefield))
        ff.Setup(mol)
        if algorithm == 'steep':
            ff.SteepestDescent(steps)
        elif algorithm == 'md':
            ff.MolecularDynamicsTakeNSteps(steps, 300)
        elif algorithm == 'cg':
            ff.ConjugateGradients(steps)
        else:
            raise Exception("Invalid minimization algorithm. Valid options "
                              "are 'steep', 'cg', and 'md'.")
        ff.UpdateCoordinates(mol)

        obConversion.WriteFile(mol, os.path.join(tmp_dir, 'minimized.pdb'))

    def save(self, filename, show_ports=False, forcefield_name=None,
             forcefield_files=None, forcefield_debug=False, box=None,
             overwrite=False, residues=None, combining_rule='lorentz',
             foyer_kwargs=None, **kwargs):
        """Save the Compound to a file.

        Parameters
        ----------
        filename : str
            Filesystem path in which to save the trajectory. The extension or
            prefix will be parsed and control the format. Supported
            extensions are: 'hoomdxml', 'gsd', 'gro', 'top',
            'lammps', 'lmp', 'mcf'
        show_ports : bool, optional, default=False
            Save ports contained within the compound.
        forcefield_files : str, optional, default=None
            Apply a forcefield to the output file using a forcefield provided
            by the `foyer` package.
        forcefield_name : str, optional, default=None
            Apply a named forcefield to the output file using the `foyer`
            package, e.g. 'oplsaa'. Forcefields listed here:
            https://github.com/mosdef-hub/foyer/tree/master/foyer/forcefields
        forcefield_debug : bool, optional, default=False
            Choose level of verbosity when applying a forcefield through `foyer`.
            Specifically, when missing atom types in the forcefield xml file,
            determine if the warning is condensed or verbose.
        box : mb.Box, optional, default=self.boundingbox (with buffer)
            Box information to be written to the output file. If 'None', a
            bounding box is used with 0.25nm buffers at each face to avoid
            overlapping atoms.
        overwrite : bool, optional, default=False
            Overwrite if the filename already exists
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.
        combining_rule : str, optional, default='lorentz'
            Specify the combining rule for nonbonded interactions. Only relevant
            when the `foyer` package is used to apply a forcefield. Valid
            options are 'lorentz' and 'geometric', specifying Lorentz-Berthelot
            and geometric combining rules respectively.
        foyer_kwargs : dict, optional, default=None
            Keyword arguments to provide to `foyer.Forcefield.apply`.
        **kwargs
            Depending on the file extension these will be passed to either
            `write_gsd`, `write_hoomdxml`, `write_lammpsdata`,
            `write_mcf`, or `parmed.Structure.save`.
            See https://parmed.github.io/ParmEd/html/structobj/parmed.structure.Structure.html#parmed.structure.Structure.save


        Other Parameters
        ----------------
        ref_distance : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting distance values to reduced units.
        ref_energy : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting energy values to reduced units.
        ref_mass : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting mass values to reduced units.
        atom_style: str, default='full'
            Defines the style of atoms to be saved in a LAMMPS data file. The following atom
            styles are currently supported: 'full', 'atomic', 'charge', 'molecular'
            see http://lammps.sandia.gov/doc/atom_style.html for more
            information on atom styles.
        unit_style: str, default='real'
            Defines to unit style to be save in a LAMMPS data file.  Defaults to 'real' units.
            Current styles are supported: 'real', 'lj'
            see https://lammps.sandia.gov/doc/99/units.html for more information
            on unit styles

        Notes
        ------
        When saving the compound as a json, only the following arguments are used:
            - filename
            - show_ports

        See Also
        --------
        formats.gsdwrite.write_gsd : Write to GSD format
        formats.hoomdxml.write_hoomdxml : Write to Hoomd XML format
        formats.xyzwriter.write_xyz : Write to XYZ format
        formats.lammpsdata.write_lammpsdata : Write to LAMMPS data format
        formats.cassandramcf.write_mcf : Write to Cassandra MCF format
        formats.json_formats.compound_to_json : Write to a json file

        """

        extension = os.path.splitext(filename)[-1]

        if extension == '.json':
            compound_to_json(self,
                             file_path=filename,
                             include_ports=show_ports)
            return

        # Savers supported by mbuild2.formats
        savers = {'.xyz': self.write_xyz,
                  '.lammps': self.write_lammpsdata,
                  '.lmp': self.write_lammpsdata}

        try:
            saver = savers[extension]
        except KeyError:
            saver = None

        structure = self.to_parmed(residues=residues,
                                   show_ports=show_ports)
        # Apply a force field with foyer if specified
        if forcefield_name or forcefield_files:
            foyer = import_('foyer')
            ff = foyer.Forcefield(forcefield_files=forcefield_files,
                            name=forcefield_name, debug=forcefield_debug)
            if not foyer_kwargs:
                foyer_kwargs = {}
            structure = ff.apply(structure, **foyer_kwargs)
            structure.combining_rule = combining_rule

        total_charge = sum([atom.charge for atom in structure])
        if round(total_charge, 4) != 0.0:
            warn('System is not charge neutral. Total charge is {}.'
                 ''.format(total_charge))

        if saver:  # mbuild supported saver.
            if extension in ['.gsd', '.hoomdxml']:
                kwargs['rigid_bodies'] = [
                        p.rigid_id for p in self.particles()]
            saver(filename=filename, structure=structure)

        elif extension == '.sdf':
            pybel = import_('pybel')
            new_compound = Compound()
            # Convert pmd.Structure to mb.Compound
            new_compound.from_parmed(structure)
            # Convert mb.Compound to pybel molecule
            pybel_molecule = new_compound.to_pybel()
            # Write out pybel molecule to SDF file
            output_sdf = pybel.Outputfile("sdf", filename,
                    overwrite=overwrite)
            output_sdf.write(pybel_molecule)
            output_sdf.close()

        else:  # ParmEd supported saver.
            structure.save(filename, overwrite=1, **kwargs)

    def translate_to(self, pos):
        """Translate the Compound to a specific position

        Parameters
        ----------
        pos : np.ndarray, shape=3(,), dtype=float

        """
        self.translate(pos - self.center(1))

    def spin(self, theta, around):
        """Rotate Compound in place around an arbitrary vector.

        Parameters
        ----------
        theta : float
            The angle by which to rotate the Compound, in radians.
        around : np.ndarray, shape=(3,), dtype=float
            The axis about which to spin the Compound.

        """
        around = np.asarray(around).reshape(3)
        center_pos = self.center()
        self.translate(-center_pos)
        self.rotate(theta, around)
        self.translate(center_pos)

    def gen_vel(self, temp):
        ''' generate random gaussian velocities for atoms'''
        import numpy.random as rnd
        vels = rnd.normal(size=[self.n_particles(), 3])
        for i, x in enumerate(self.particles_label_sorted()):
            tmp = (u.BOLTZMANN_CONSTANT_kB*temp*u.kelvin/float(x.type['mass'])/u.gram*u.AVOGADRO_CONSTANT_NA._value).\
                      in_units_of(u.angstrom**2/u.femtoseconds**2)
            vels[i] *= np.sqrt(tmp._value)

    def supercell(self,p):
        p = np.array(p)
        if np.array(p).flatten().size == 3:
            p = np.diag(p)
        ncells = np.rint(det(p)).astype(int)

        self.latmat = p @ self.latmat

        corners = np.array([[0, 0, 0], [0, 0, 1],
                            [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1],
                            [1, 1, 0], [1, 1, 1]]) @ p;

        mn = np.amin(corners,0)
        mx = np.amax(corners,0)
        aa, bb, cc = np.meshgrid(*[np.arange(x,y+1) for x,y in zip(mn,mx)])
        fracs = np.vstack([x.flatten(order='F') for x in [aa,bb,cc]]).transpose() @ inv(p)

        eps=1e-8
        tmp1 = deepcopy(self)
        self.remove(self.particles(1))
        flg = 1
        for mv in fracs:
            if np.all((mv < 1 - eps) & (mv > -eps)):
                val = mv @ self.latmat
                tmp = deepcopy(tmp1)
                if flg:
                    first_cell = tmp
                    flg = 0
                tmp.xyz += val
                self.add(tmp)
        nlst1 = tmp1.particles()
        nlst2 = self.particles()
        n1 = tmp1.n_particles()
        n2 = self.n_particles()
        if self.bond_graph:
            for x in self.bond_graph.edges:
                equivalent_parts = [nlst2[i] for i in range(nlst2.index(x[1]) % n1, n2, n1)]
                p = self.closest_img(x[0], equivalent_parts)
                if x[1] is not p:
                    self.bond_graph.remove_edge(*x)
                    self.bond_graph.add_edge(x[0], p)
            self.gen_angs_and_diheds()

        self.wrap_atoms()



    @property
    def index(self):
        return self.root.particles_label_sorted().index(self)

    # Interface to Trajectory for reading/writing .pdb and .mol2 files.
    # -----------------------------------------------------------------
    def from_trajectory(self, traj, frame=-1, infer_hierarchy=True):
        """Extract atoms and bonds from a md.Trajectory.

        Will create sub-compounds for every chain if there is more than one
        and sub-sub-compounds for every residue.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to load.
        frame : int, optional, default=-1 (last)
            The frame to take coordinates from.
        infer_hierarchy : bool, optional, default=True
            If True, infer compound hierarchy from chains and residues
        """

        atom_mapping = dict()
        for chain in traj.topology.chains:
            if traj.topology.n_chains > 1:
                chain_compound = Compound()
                self.add(chain_compound, 'chain[$]')
            else:
                chain_compound = self
            for res in chain.residues:
                if infer_hierarchy:
                    res_compound = Compound(name=res.name)
                    chain_compound.add(res_compound)
                    parent_cmpd = res_compound
                else:
                    parent_cmpd = chain_compound
                for atom in res.atoms:
                    new_atom = Compound(name=str(atom.name),
                                        pos=traj.xyz[frame, atom.index])
                    parent_cmpd.add(new_atom)
                    atom_mapping[atom] = new_atom

        for mdtraj_atom1, mdtraj_atom2 in traj.topology.bonds:
            atom1 = atom_mapping[mdtraj_atom1]
            atom2 = atom_mapping[mdtraj_atom2]
            self.add_bond((atom1, atom2))

    def to_trajectory(self, show_ports=False, chains=None,
                      residues=None, box=None):
        """Convert to an md.Trajectory and flatten the compound.

        Parameters
        ----------
        show_ports : bool, optional, default=False
            Include all port atoms when converting to trajectory.
        chains : mb.Compound or list of mb.Compound
            Chain types to add to the topology
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.
        box : mb.Box, optional, default=self.boundingbox (with buffer)
            Box information to be used when converting to a `Trajectory`.
            If 'None', a bounding box is used with a 0.5nm buffer in each
            dimension. to avoid overlapping atoms, unless `self.periodicity`
            is not None, in which case those values are used for the
            box lengths.

        Returns
        -------
        trajectory : md.Trajectory

        See also
        --------
        _to_topology

        """
        import mdtraj as md
        atom_list = [particle for particle in self.particles_label_sorted(0)]

        top = self._to_topology(atom_list, chains, residues)

        return md.Trajectory(self.xyz, top, \
                             unitcell_lengths=[np.linalg.norm(x) for x in self.latmat],
                             unitcell_angles=[angle_between_vecs(v1,v2) for v1,v2 in combinations(self.latmat,2)][::-1])

    def _to_topology(self, atom_list, chains=None, residues=None):
        """Create a mdtraj.Topology from a Compound.

        Parameters
        ----------
        atom_list : list of mb.Compound
            Atoms to include in the topology
        chains : mb.Compound or list of mb.Compound
            Chain types to add to the topology
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.

        Returns
        -------
        top : mdtraj.Topology

        See Also
        --------
        mdtraj.Topology : Details on the mdtraj Topology object

        """
        from mdtraj.core.topology import Topology
        from mdtraj.core.element import get_by_symbol

        if isinstance(chains, str):
            chains = [chains]
        if isinstance(chains, (list, set)):
            chains = tuple(chains)

        if isinstance(residues, str):
            residues = [residues]
        if isinstance(residues, (list, set)):
            residues = tuple(residues)
        top = Topology()
        atom_mapping = {}

        default_chain = top.add_chain()
        default_residue = top.add_residue('RES', default_chain)

        compound_residue_map = dict()
        atom_residue_map = dict()
        compound_chain_map = dict()
        atom_chain_map = dict()

        for atom in atom_list:
            # Chains
            if chains:
                if atom.name in chains:
                    current_chain = top.add_chain()
                    compound_chain_map[atom] = current_chain
                else:
                    for parent in atom.ancestors():
                        if chains and parent.name in chains:
                            if parent not in compound_chain_map:
                                current_chain = top.add_chain()
                                compound_chain_map[parent] = current_chain
                                current_residue = top.add_residue(
                                    'RES', current_chain)
                            break
                    else:
                        current_chain = default_chain
            else:
                current_chain = default_chain
            atom_chain_map[atom] = current_chain

            # Residues
            if residues:
                if atom.name in residues:
                    current_residue = top.add_residue(atom.name, current_chain)
                    compound_residue_map[atom] = current_residue
                else:
                    for parent in atom.ancestors():
                        if residues and parent.name in residues:
                            if parent not in compound_residue_map:
                                current_residue = top.add_residue(
                                    parent.name, current_chain)
                                compound_residue_map[parent] = current_residue
                            break
                    else:
                        current_residue = default_residue
            else:
                if chains:
                    try:  # Grab the default residue from the custom chain.
                        current_residue = next(current_chain.residues)
                    except StopIteration:  # Add the residue to the current chain
                        current_residue = top.add_residue('RES', current_chain)
                else:  # Grab the default chain's default residue
                    current_residue = default_residue
            atom_residue_map[atom] = current_residue

            # Add the actual atoms
            try:
                elem = get_by_symbol(atom.name)
            except KeyError:
                elem = get_by_symbol("VS")
            at = top.add_atom(atom.name, elem, atom_residue_map[atom])
            atom_mapping[atom] = at

        # Remove empty default residues.
        chains_to_remove = [
            chain for chain in top.chains if chain.n_atoms == 0]
        residues_to_remove = [res for res in top.residues if res.n_atoms == 0]
        for chain in chains_to_remove:
            top._chains.remove(chain)
        for res in residues_to_remove:
            for chain in top.chains:
                try:
                    chain._residues.remove(res)
                except ValueError:  # Already gone.
                    pass

        for atom1, atom2 in self.bond_graph.edges:
            # Ensure that both atoms are part of the compound. This becomes an
            # issue if you try to convert a sub-compound to a topology which is
            # bonded to a different subcompound.
            if all(a in atom_mapping.keys() for a in [atom1, atom2]):
                top.add_bond(atom_mapping[atom1], atom_mapping[atom2])
        return top

    def from_parmed(self, structure, coords_only=False,
            infer_hierarchy=True):
        """Extract atoms and bonds from a pmd.Structure.

        Will create sub-compounds for every chain if there is more than one
        and sub-sub-compounds for every residue.

        Parameters
        ----------
        structure : pmd.Structure
            The structure to load.
        coords_only : bool
            Set preexisting atoms in compound to coordinates given by structure.
        infer_hierarchy : bool, optional, default=True
            If true, infer compound hierarchy from chains and residues
        """
        if coords_only:
            if len(structure.atoms) != self.n_particles(0):
                raise ValueError(
                    'Number of atoms in {structure} does not match'
                    ' {self}'.format(
                        **locals()))
            atoms_particles = zip(structure.atoms,
                                  self.particles(include_ports=False))
            if None in self._particles(include_ports=False):
                raise ValueError('Some particles are None')
            for parmed_atom, particle in atoms_particles:
                particle.pos = np.array([parmed_atom.xx,
                                         parmed_atom.xy,
                                         parmed_atom.xz]) / 10
            return

        atom_mapping = dict()
        chain_id = None
        chains = defaultdict(list)
        for residue in structure.residues:
            chains[residue.chain].append(residue)

        for chain, residues in chains.items():
            if len(chains) > 1:
                chain_compound = Compound()
                self.add(chain_compound, chain_id)
            else:
                chain_compound = self
            for residue in residues:
                if infer_hierarchy:
                    residue_compound = Compound(name=residue.name)
                    chain_compound.add(residue_compound)
                    parent_cmpd = residue_compound
                else:
                    parent_cmpd = chain_compound
                for atom in residue.atoms:
                    pos = np.array([atom.xx, atom.xy, atom.xz])
                    new_atom = Particle(name=str(atom.name), pos=pos)
                    parent_cmpd.add(
                        new_atom, name=atom.name)
                    atom_mapping[atom] = new_atom

        for bond in structure.bonds:
            atom1 = atom_mapping[bond.atom1]
            atom2 = atom_mapping[bond.atom2]
            self.add_bond((atom1, atom2))

        self.latmat=np.array(structure.box_vectors._value)

    # def from_ase(self, atoms_obj):
        


    @property
    def frac_coords(self):
        return self.xyz @ np.linalg.inv(self.latmat)

    def to_pymatgen(self, show_ports=False):
        """create a pymatgen structure from a compound

        Parameters
        ----------
        returns: pymatgen.Structure"""
        trj = self.to_trajectory()
        return Structure(Lattice.from_parameters(*trj.unitcell_lengths,*trj.unitcell_angles),\
                         [x.name for x in self.particles()],self.xyz)


    def to_parmed(self, box=None, title='', residues=None, show_ports=False,
            infer_residues=False):
        """Create a ParmEd Structure from a Compound.

        Parameters
        ----------
        box : mb.Box, optional, default=self.boundingbox (with buffer)
            Box information to be used when converting to a `Structure`.
            If 'None', a bounding box is used with 0.25nm buffers at
            each face to avoid overlapping atoms, unless `self.periodicity`
            is not None, in which case those values are used for the
            box lengths.
        title : str, optional, default=self.name
            Title/name of the ParmEd Structure
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.
        show_ports : boolean, optional, default=False
            Include all port atoms when converting to a `Structure`.
        infer_residues : bool, optional, default=False
            Attempt to assign residues based on names of children.

        Returns
        -------
        parmed.structure.Structure
            ParmEd Structure object converted from self

        See Also
        --------
        parmed.structure.Structure : Details on the ParmEd Structure object

        """
        structure = pmd.Structure()
        structure.title = title if title else self.name
        atom_mapping = {}  # For creating bonds below
        guessed_elements = set()

        # Attempt to grab residue names based on names of children
        if not residues and infer_residues:
            residues = list(set([child.name for child in self.children]))

        if isinstance(residues, str):
            residues = [residues]
        if isinstance(residues, (list, set)):
            residues = tuple(residues)

        default_residue = pmd.Residue('RES')
        port_residue = pmd.Residue('PRT')
        compound_residue_map = dict()
        atom_residue_map = dict()

        # Loop through particles and add initialize ParmEd atoms
        for atom in self.particles(0):
            if '!' in atom.name:
                current_residue = port_residue
                atom_residue_map[atom] = current_residue

                if current_residue not in structure.residues:
                    structure.residues.append(current_residue)

                pmd_atom = pmd.Atom(atomic_number=0, name='VS',
                                    mass=0, charge=0)
                pmd_atom.xx, pmd_atom.xy, pmd_atom.xz = atom.pos  # Angstroms

            else:
                if residues and atom.name in residues:
                    current_residue = pmd.Residue(atom.name)
                    atom_residue_map[atom] = current_residue
                    compound_residue_map[atom] = current_residue
                elif residues:
                    for parent in atom.ancestors():
                        if residues and parent.name in residues:
                            if parent not in compound_residue_map:
                                current_residue = pmd.Residue(parent.name)
                                compound_residue_map[parent] = current_residue
                            atom_residue_map[atom] = current_residue
                            break
                    else:  # Did not find specified residues in ancestors.
                        current_residue = default_residue
                        atom_residue_map[atom] = current_residue
                else:
                    current_residue = default_residue
                    atom_residue_map[atom] = current_residue

                if current_residue not in structure.residues:
                    structure.residues.append(current_residue)

                atomic_number = None
                name = ''.join(char for char in atom.name if not char.isdigit())
                try:
                    atomic_number = AtomicNum[atom.name.capitalize()]
                except KeyError:
                    element = element_by_name(atom.name.capitalize())
                    if name not in guessed_elements:
                        # warn(
                        #     'Guessing that "{}" is element: "{}"'.format(
                        #         atom, element))
                        guessed_elements.add(name)
                else:
                    element = atom.name.capitalize()

                atomic_number = atomic_number or AtomicNum[element]
                mass = Mass[element]
                pmd_atom = pmd.Atom(atomic_number=atomic_number, name=atom.name,
                                    mass=mass)
                pmd_atom.xx, pmd_atom.xy, pmd_atom.xz = atom.pos  # Angstroms

            residue = atom_residue_map[atom]
            structure.add_atom(pmd_atom, resname=residue.name,
                               resnum=residue.idx)

            atom_mapping[atom] = pmd_atom

        # "Claim" all of the items it contains and subsequently index all of its items
        structure.residues.claim()

        # Create and add bonds to ParmEd Structure
        for atom1, atom2 in self.bond_graph.edges:
            bond = pmd.Bond(atom_mapping[atom1], atom_mapping[atom2])
            structure.bonds.append(bond)

        bb = self.to_trajectory()
        structure.box = bb.unitcell_lengths[0].tolist()+bb.unitcell_angles[0].tolist()
        return structure

    def to_networkx(self, names_only=False):
        """Create a NetworkX graph representing the hierarchy of a Compound.

        Parameters
        ----------
        names_only : bool, optional, default=False
        Store only the names of the
            compounds in the graph, appended with their IDs, for distinction even
            if they have the same name. When set to False, the default behavior,
            the nodes are the compounds themselves.

        Returns
        -------
        G : networkx.DiGraph

        Notes
        -----
        This digraph is not the bondgraph of the compound.

        See Also
        --------
        mbuild2.bond_graph
        """
        nx = import_('networkx')

        nodes = list()
        edges = list()
        if names_only:
            nodes.append(self.name + '_' + str(id(self)))
        else:
            nodes.append(self)
        nodes, edges = self._iterate_children(nodes, edges, names_only=names_only)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _iterate_children(self, nodes, edges, names_only=False):
        """ Create nodes and edges that connect parents and their corresponding children"""
        if not self.children:
            return nodes, edges
        for child in self.children:
            if names_only:
                unique_name = child.name + '_' + str(id(child))
                unique_name_parent = child.parent.name + '_' + str((id(child.parent)))
                nodes.append(unique_name)
                edges.append([unique_name_parent, unique_name])
            else:
                nodes.append(child)
                edges.append([child.parent, child])
            nodes, edges = child._iterate_children(nodes, edges, names_only=names_only)
        return nodes, edges

    def to_pybel(self, show_ports=False):
        """ Create a pybel.Molecule from a Compound

        Parameters
        ---------
        box : mb.Box, def None
        title : str, optional, default=self.name
            Title/name of the ParmEd Structure
        residues : str or list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.
        show_ports : boolean, optional, default=False
            Include all port atoms when converting to a `Structure`.
        infer_residues : bool, optional, default=False
            Attempt to assign residues based on names of children

        Returns
        ------
        pybel.Molecule

        Notes
        -----
        Most of the mb.Compound is first converted to openbabel.OBMol
        And then pybel creates a pybel.Molecule from the OBMol
        Bond orders are assumed to be 1
        OBMol atom indexing starts at 1, with spatial dimension Angstrom
        """

        mol = openbabel.OBMol()
        particle_to_atom_index = dict()
        for i, part in enumerate(self.particles(show_ports)):

            temp = mol.NewAtom()

            try:
                temp.SetAtomicNum(AtomicNum[part.name.capitalize()])
            except KeyError:
                warn("Could not infer atomic number from "
                        "{}, setting to 0".format(part.name))
                temp.SetAtomicNum(0)


            temp.SetVector(*(part.pos))
            particle_to_atom_index[part] = i

        ucell = openbabel.OBUnitCell()
        ucell.SetData(*map(lambda x:openbabel.vector3(*x),np.array(self.latmat, dtype=float)))

        if self.bond_graph:
            for bond in self.bond_graph.edges:
                bond_order = 1
                mol.AddBond(particle_to_atom_index[bond[0]]+1,
                        particle_to_atom_index[bond[1]]+1,
                        bond_order)

        return pybel.Molecule(mol)

    def from_pybel(self, pybel_mol, use_element=True):
        """Create a Compound from a Pybel.Molecule

        Parameters
        ---------
        pybel_mol: pybel.Molecule
        use_element : bool, default True
            If True, construct mb Particles based on the pybel Atom's element.
            If False, construcs mb Particles based on the pybel Atom's type
        """
        self.name = pybel_mol.title.split('.')[0]
        resindex_to_cmpd = {}

        # Iterating through pybel_mol for atom/residue information
        # This could just as easily be implemented by
        # an OBMolAtomIter from the openbabel library,
        # but this seemed more convenient at time of writing
        # pybel atoms are 1-indexed, coordinates in Angstrom
        for atom in pybel_mol.atoms:
            xyz = np.array(atom.coords)/10
            if use_element:
                try:
                    temp_name = Element[atom.atomicnum]
                except KeyError:
                    warn("No element detected for atom at index "
                            "{} with number {}, type {}".format(
                                atom.idx, atom.atomicnum, atom.type))
                    temp_name = atom.type
            else:
                temp_name = atom.type
            temp = Particle(name=temp_name, pos=xyz)
            if infer_hierarchy and hasattr(atom, 'residue'):
                # Is there a safer way to check for res?
                if atom.residue.idx not in resindex_to_cmpd:
                    res_cmpd = Compound(name=atom.residue.name)
                    resindex_to_cmpd[atom.residue.idx] = res_cmpd
                    self.add(res_cmpd)
                resindex_to_cmpd[atom.residue.idx].add(temp)
            else:
                self.add(temp)

        # Iterating through pybel_mol.OBMol for bond information
        # Bonds are 0-indexed, but the atoms are 1-indexed
        # Bond information doesn't appear stored in pybel_mol,
        # so we need to look into the OBMol object,
        # using an iterator from the openbabel library
        for bond in openbabel.OBMolBondIter(pybel_mol.OBMol):
            self.add_bond([self[bond.GetBeginAtomIdx()-1],
                            self[bond.GetEndAtomIdx()-1]])

        if hasattr(pybel_mol, 'unitcell'):
            box = Box(lengths=[pybel_mol.unitcell.GetA()/10,
                                pybel_mol.unitcell.GetB()/10,
                                pybel_mol.unitcell.GetC()/10],
                        angles=[pybel_mol.unitcell.GetAlpha(),
                                pybel_mol.unitcell.GetBeta(),
                                pybel_mol.unitcell.GetGamma()])
            self.periodicity = box.lengths
        else:
            warn("No unitcell detected for pybel.Molecule {}".format(pybel_mol))
#       TODO: Decide how to gather PBC information from openbabel. Options may
#             include storing it in .periodicity or writing a separate function
#             that returns the box.

    def to_intermol(self, molecule_types=None): # pragma: no cover
        """Create an InterMol system from a Compound.

        Parameters
        ----------
        molecule_types : list or tuple of subclasses of Compound

        Returns
        -------
        intermol_system : intermol.system.System
        """
        from intermol.atom import Atom as InterMolAtom
        from intermol.molecule import Molecule
        from intermol.system import System
        import simtk.unit as u

        if isinstance(molecule_types, list):
            molecule_types = tuple(molecule_types)
        elif molecule_types is None:
            molecule_types = (type(self),)
        intermol_system = System()

        last_molecule_compound = None
        for atom_index, atom in enumerate(self.particles()):
            for parent in atom.ancestors():
                # Don't want inheritance via isinstance().
                if type(parent) in molecule_types:
                    # Check if we have encountered this molecule type before.
                    if parent.name not in intermol_system.molecule_types:
                        self._add_intermol_molecule_type(
                            intermol_system, parent)
                    if parent != last_molecule_compound:
                        last_molecule_compound = parent
                        last_molecule = Molecule(name=parent.name)
                        intermol_system.add_molecule(last_molecule)
                    break
            else:
                # Should never happen if molecule_types only contains
                # type(self)
                raise ValueError(
                    'Found an atom {} that is not part of any of '
                    'the specified molecule types {}'.format(
                        atom, molecule_types))

            # Add the actual intermol atoms.
            intermol_atom = InterMolAtom(atom_index + 1, name=atom.name,
                                         residue_index=1, residue_name='RES')
            intermol_atom.position = atom.pos * u.nanometers
            last_molecule.add_atom(intermol_atom)
        return intermol_system

    def get_smiles(self):
        """Get SMILES string for compound

        Bond order is guessed with pybel and may lead to incorrect SMILES
        strings.

        Returns
        -------
        smiles_string: str
        """

        pybel_cmp = self.to_pybel()
        pybel_cmp.OBMol.PerceiveBondOrders()
        # we only need the smiles string
        smiles = pybel_cmp.write().split()[0]
        return smiles


    @staticmethod
    def _add_intermol_molecule_type(intermol_system, parent):  # pragma: no cover
        """Create a molecule type for the parent and add bonds.

        This method takes an intermol system and adds a
        parent compound, including its particles and bonds, to it.
        """
        from intermol.moleculetype import MoleculeType
        from intermol.forces.bond import Bond as InterMolBond

        molecule_type = MoleculeType(name=parent.name)
        intermol_system.add_molecule_type(molecule_type)

        for index, parent_atom in enumerate(parent.particles()):
            parent_atom.index = index + 1

        for atom1, atom2 in parent.bonds():
            intermol_bond = InterMolBond(atom1.index, atom2.index)
            molecule_type.bonds.add(intermol_bond)

    def __getitem__(self, selection):

        if isinstance(selection, int):
            return self.children[selection]
        if isinstance(selection, str):
            if selection not in [x.name for x in self.children]:
                raise Exception('{}[\'{}\'] does not exist.'.format(self.name,selection))
            out = [x for x in self.children if x.name == selection]
            if out.__len__()==1:
                return out[0]
            else:
                return out
        lst = []
        if isinstance(selection, Iterable):
            for x in selection:
                lst.append(self.__getitem__(x))
            return lst

    def __repr__(self):
        descr = list('<')
        descr.append(self.name + ' ')

        if self.children:
            descr.append('{:d} particles, '.format(self.n_particles(1)))
        else:
            descr.append('pos=({: .4f},{: .4f},{: .4f}), '.format(*self.pos))

        # descr.append('{:d} bonds, '.format(self.n_bonds))

        descr.append('id: {}>'.format(str(id(self))[-4:-1]))
        return ''.join(descr)

    def _clone(self, clone_of=None, root_container=None):
        """A faster alternative to deepcopying.

        Does not resolve circular dependencies. This should be safe provided
        you never try to add the top of a Compound hierarchy to a
        sub-Compound. Clones compound hierarchy only, not the bonds.
        """
        if root_container is None:
            root_container = self
        if clone_of is None:
            clone_of = dict()

        # If this compound has already been cloned, return that.
        if self in clone_of:
            return clone_of[self]

        # Otherwise we make a new clone.
        cls = self.__class__
        newone = cls.__new__(cls)

        # Remember that we're cloning the new one of self.
        clone_of[self] = newone

        newone.name = deepcopy(self.name)
        # newone.periodicity = deepcopy(self.periodicity)
        newone._pos = deepcopy(self._pos)
        newone.port_particle = deepcopy(self.port_particle)
        if hasattr(self, 'index'):
            newone.index = deepcopy(self.index)

        if self.children is None:
            newone.children = None
        else:
            newone.children = OrderedSet()
        # Parent should be None initially.
        newone.parent = None
        newone.labels = OrderedDict()
        newone.referrers = set()
        newone.bond_graph = None

        # Add children to clone.
        if self.children:
            for child in self.children:
                newchild = child._clone(clone_of, root_container)
                newone.children.add(newchild)
                newchild.parent = newone

        return newone

    def _clone_bonds(self, clone_of=None):
        """While cloning, clone the bond of the source compound to clone compound"""
        newone = clone_of[self]
        for c1, c2 in self.bonds():
            try:
                newone.add_bond((clone_of[c1], clone_of[c2]))
            except KeyError:
                raise Exception(
                    "Cloning failed. Compound contains bonds to "
                    "Particles outside of its containment hierarchy.")

    def wrap_atoms(self):
        """wraps atoms to equivalent positions inside the unit cell"""
        self.xyz=(self.frac_coords-np.floor(self.frac_coords)) @ self.latmat


    def read_xyz(self,filename):
        """Read an XYZ file. The expected format is as follows:
        The first line contains the number of atoms in the file The second line
        contains a comment, which is not read.  Remaining lines, one for each
        atom in the file, include an elemental symbol followed by X, Y, and Z
        coordinates in Angstroms. Columns are expected tbe separated by
        whitespace. See https://openbabel.org/wiki/XYZ_(format).

        Parameters
        ----------
        filename : str
            Path of the input file

        Returns
        -------
        compound : mb.Compound

        Notes
        -----
        The XYZ file format neglects many important details, including bonds,
        residues, and box information.

        There are some other flavors of the XYZ file format and not all are
        guaranteed to be compatible with this reader. For example, the TINKER
        XYZ format is not expected to be properly read.
        """

        if self is None:
            compound = Compound()
        else:
            compound = self

        with open(filename, 'r') as xyz_file:
            n_atoms = int(xyz_file.readline())
            xyz_file.readline()
            coords = np.zeros(shape=(n_atoms, 3), dtype=np.float64)
            for row, _ in enumerate(coords):
                line = xyz_file.readline().split()
                if not line:
                    msg = ('Incorrect number of lines in input file. Based on the '
                           'number in the first line of the file, {} rows of atoms '
                           'were expected, but at least one fewer was found.')
                    raise Exception(msg.format(n_atoms))
                coords[row] = line[1:4]
                particle = Compound(pos=coords[row], name=line[0])
                compound.add(particle)

            # Verify we have read the last line by ensuring the next line in blank
            line = xyz_file.readline().split()
            if line:
                msg = ('Incorrect number of lines in input file. Based on the '
                       'number in the first line of the file, {} rows of atoms '
                       'were expected, but at least one more was found.')
                raise Exception(msg.format(n_atoms))

        return compound


    def write_xyz(self, filename, label_sorted=0):
        '''filename can be an already opened file'''
        types = [atom.name for atom in self.particles_label_sorted(0)]

        fle = open(filename, 'w') if isinstance(filename, str) else filename

        with fle as xyz_file:
            xyz_file.write(str(self.n_particles(0)))
            xyz_file.write('\ncreated by mBuild\n')
            for typ, coords in zip(types, self.xyz_label_sorted if label_sorted else self.xyz):
                xyz_file.write('{:s} {:11.6f} {:11.6f} {:11.6f}\n'.format(typ, *coords))

    def x_axis_transform(compound, new_origin=None,
                         point_on_x_axis=None,
                         point_on_xy_plane=None):
        """Move a compound such that the x-axis lies on specified points.

        Parameters
        ----------
        compound : mb.Compound
            The compound to move.
        new_origin : mb.Compound or list-like of size 3, optional, default=[0.0, 0.0, 0.0]
            Where to place the new origin of the coordinate system.
        point_on_x_axis : mb.Compound or list-like of size 3, optional, default=[1.0, 0.0, 0.0]
            A point on the new x-axis.
        point_on_xy_plane : mb.Compound, or list-like of size 3, optional, default=[1.0, 0.0, 0.0]
            A point on the new xy-plane.

        """

        if new_origin is None:
            new_origin = np.array([0, 0, 0])
        elif isinstance(new_origin, Compound):
            new_origin = new_origin.pos
        elif isinstance(new_origin, (tuple, list, np.ndarray)):
            new_origin = np.asarray(new_origin)
        else:
            raise TypeError('x_axis_transform, y_axis_transform, and z_axis_transform only accept'
                            ' mb.Compounds, list-like of length 3 or None for the new_origin'
                            ' parameter. User passed type: {}.'.format(type(new_origin)))
        if point_on_x_axis is None:
            point_on_x_axis = np.array([1.0, 0.0, 0.0])
        elif isinstance(point_on_x_axis, Compound):
            point_on_x_axis = point_on_x_axis.pos
        elif isinstance(point_on_x_axis, (list, tuple, np.ndarray)):
            point_on_x_axis = np.asarray(point_on_x_axis)
        else:
            raise TypeError('x_axis_transform, y_axis_transform, and z_axis_transform only accept'
                            ' mb.Compounds, list-like of size 3, or None for the point_on_x_axis'
                            ' parameter. User passed type: {}.'.format(type(point_on_x_axis)))
        if point_on_xy_plane is None:
            point_on_xy_plane = np.array([1.0, 1.0, 0.0])
        elif isinstance(point_on_xy_plane, Compound):
            point_on_xy_plane = point_on_xy_plane.pos
        elif isinstance(point_on_xy_plane, (list, tuple, np.ndarray)):
            point_on_xy_plane = np.asarray(point_on_xy_plane)
        else:
            raise TypeError('x_axis_transform, y_axis_transform, and z_axis_transform only accept'
                            ' mb.Compounds, list-like of size 3, or None for the point_on_xy_plane'
                            ' parameter. User passed type: {}.'.format(type(point_on_xy_plane)))

        atom_positions = compound.xyz_with_ports
        transform = AxisTransform(new_origin=new_origin,
                                  point_on_x_axis=point_on_x_axis,
                                  point_on_xy_plane=point_on_xy_plane)
        atom_positions = transform.apply_to(atom_positions)
        compound.xyz_with_ports = atom_positions


    def force_overlap(self,comp_to_mv,port_of_comp,
                      port_of_target,add_bond=1,flip=0,rotate_ang=0):
        """
        rotate_ang: in degrees
        """
        port_of_comp.orientation /= norm(port_of_comp.orientation)
        port_of_target.orientation /= norm(port_of_target.orientation)

        comp_to_mv.xyz_with_ports -= port_of_comp.pos
        comp_to_mv.rotate_vecs(port_of_comp.orientation, port_of_target.orientation, pnt=port_of_comp.pos)
        if flip:
            comp_to_mv.reflect(port_of_comp.pos, port_of_comp.orientation)

        comp_to_mv.rotate_around(port_of_comp.orientation, rotate_ang)

        comp_to_mv.xyz_with_ports += port_of_target.pos

        if port_of_comp.anchor and port_of_target.anchor:
            self.add_bond([port_of_comp.anchor, port_of_target.anchor])
        port_of_comp.parent.remove(port_of_comp)
        port_of_target.parent.remove(port_of_target)


    def _create_equivalence_transform(self,equiv):
        """Compute an equivalence transformation that transforms this compound
        to another compound's coordinate system.

        Parameters
        ----------
        equiv : np.ndarray, shape=(n, 3), dtype=float
            Array of equivalent points.

        Returns
        -------
        T : CoordinateTransform
            Transform that maps this point cloud to the other point cloud's
            coordinates system.

        """
        from compound import Compound
        self_points = np.array([])
        self_points.shape = (0, 3)
        other_points = np.array([])
        other_points.shape = (0, 3)

        for pair in equiv:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('Equivalence pair not a 2-tuple')
            if not (isinstance(pair[0], Compound) and isinstance(pair[1], Compound)):
                raise ValueError('Equivalence pair type mismatch: pair[0] is a {0} '
                                 'and pair[1] is a {1}'.format(type(pair[0]),
                                                               type(pair[1])))

            # TODO: vstack is slow, replace with list concatenation
            if not pair[0].children:
                self_points = np.vstack([self_points, pair[0].pos])
                other_points = np.vstack([other_points, pair[1].pos])
            else:
                for atom0 in pair[0]._particles(include_ports=True):
                    self_points = np.vstack([self_points, atom0.pos])
                for atom1 in pair[1]._particles(include_ports=True):
                    other_points = np.vstack([other_points, atom1.pos])
        T = RigidTransform(self_points, other_points)
        return T


    def bond_partners(self):

        out = set()
        for b in self.root.bond_graph.edges:
            if b[0] is self:
                out.add(b[1])
            elif b[1] is self:
                out.add(b[0])
        return out


    def _node_match(self, host, pattern):
        """ Determine if two graph nodes are equal """
        atom_expr = pattern['atom'].children[0]
        atom = host['atom']
        return self._atom_expr_matches(atom_expr, atom)

    def _atom_expr_matches(self, atom_expr, atom):
        """ Helper function for evaluating SMARTS string expressions """
        if atom_expr.data == 'not_expression':
            return not self._atom_expr_matches(atom_expr.children[0], atom)
        elif atom_expr.data in ('and_expression', 'weak_and_expression'):
            return (self._atom_expr_matches(atom_expr.children[0], atom) and
                    self._atom_expr_matches(atom_expr.children[1], atom))
        elif atom_expr.data == 'or_expression':
            return (self._atom_expr_matches(atom_expr.children[0], atom) or
                    self._atom_expr_matches(atom_expr.children[1], atom))
        elif atom_expr.data == 'atom_id':
            return self._atom_id_matches(atom_expr.children[0], atom, self.typemap)
        elif atom_expr.data == 'atom_symbol':
            return self._atom_id_matches(atom_expr, atom, self.typemap)
        else:
            raise TypeError('Expected atom_id, atom_symbol, and_expression, '
                            'or_expression, or not_expression. '
                            'Got {}'.format(atom_expr.data))

    @staticmethod
    def _atom_id_matches(atom_id, atom, typemap):
        """ Helper func for comparing atomic indices, symbols, neighbors, rings """
        # atomic_num = atom.element.atomic_number
        if atom_id.data == 'atomic_num':
            return atomic_num == int(atom_id.children[0])
        elif atom_id.data == 'atom_symbol':
            if str(atom_id.children[0]) == '*':
                return True
            elif str(atom_id.children[0]).startswith('_'):
                return atom.element.name == str(atom_id.children[0])
            else:
                return str(atom_id.children[0]) == atom.name
                # return atomic_num == pt.AtomicNum[str(atom_id.children[0])]
        elif atom_id.data == 'has_label':
            label = atom_id.children[0][1:]  # Strip the % sign from the beginning.
            return label == atom.type[0]['name']
        elif atom_id.data == 'neighbor_count':
            return len(atom.bond_partners()) == int(atom_id.children[0])
        elif atom_id.data == 'ring_size':
            cycle_len = int(atom_id.children[0])
            for cycle in typemap[atom.index]['cycles']:
                if len(cycle) == cycle_len:
                    return True
            return False
        elif atom_id.data == 'ring_count':
            n_cycles = len(typemap[atom.index]['cycles'])
            if n_cycles == int(atom_id.children[0]):
                return True
            return False
        elif atom_id.data == 'matches_string':
            raise NotImplementedError('matches_string is not yet implemented')

    def network_b_a_d(self):
         # create angles and dihedrals
        nlst = list(self.particles(0))
        angles = OrderedSet()
        propers = OrderedSet()
        for i, part1 in enumerate(nlst):
            for part2 in nlst[i+1:]:
                tmp = nx.all_simple_paths(self.bond_graph, part1, part2, 4) # type: list
                for ipath in list(tmp):
                    if len(ipath) == 3:
                        angles.add(tuple([x for x in ipath]))
                    elif len(ipath) == 4:
                        propers.add(tuple([x for x in ipath]))
        return self.bond_graph.edges, angles, propers
    
    def create_bonding_all(self, kb=300, ka=500, kd=500, acfpath=None):
        import pymatgen.core.periodic_table as pt
        # ff = ElementTree(element=Element('ForceField'))
        types = equivalence_classes(self.particles_label_sorted(), lambda x,y: x.name==y.name)
        # atom_type = SubElement(ff.getroot(), 'AtomTypes')
        self.ff = FF()
        # aa = [[ 89.  , 227.03], [ 47.  , 107.87], [ 13.  ,  26.98], [ 95.  , 243.  ], [ 18.  ,  39.95], [ 33.  ,  74.92], [ 85.  , 210.  ], [ 79.  , 196.97], [  5.  ,  10.81], [ 56.  , 137.33], [  4.  ,   9.01], [ 83.  , 208.98], [ 97.  , 247.  ], [ 35.  ,  79.9 ], [  6.  ,  12.01], [ 20.  ,  40.08], [ 48.  , 112.41], [ 58.  , 140.12], [ 98.  , 251.  ], [ 17.  ,  35.45], [ 96.  , 247.  ], [ 27.  ,  58.93], [ 24.  ,  52.  ], [ 55.  , 132.91], [ 29.  ,  63.55], [ 66.  , 162.5 ], [ 68.  , 167.26], [ 99.  , 252.  ], [ 63.  , 151.97], [  9.  ,  19.  ], [ 26.  ,  55.85], [100.  , 257.  ], [ 87.  , 223.  ], [ 31.  ,  69.72], [ 64.  , 157.25], [ 32.  ,  72.61], [  1.  ,   1.01], [105.  , 260.  ], [  2.  ,   4.  ], [ 72.  , 178.49], [ 80.  , 200.59], [ 67.  , 164.93], [ 53.  , 126.91], [ 49.  , 114.82], [ 77.  , 192.22], [ 19.  ,  39.1 ], [ 36.  ,  83.8 ], [ 57.  , 138.91], [  3.  ,   6.94], [103.  , 260.  ], [ 71.  , 174.97], [101.  , 258.  ], [ 12.  ,  24.31], [ 25.  ,  54.94], [ 42.  ,  95.94], [  7.  ,  14.01], [ 11.  ,  22.99], [ 41.  ,  92.91], [ 60.  , 144.24], [ 10.  ,  20.18], [ 28.  ,  58.69], [102.  , 259.  ], [ 93.  , 237.05], [  8.  ,  16.  ], [ 76.  , 190.2 ], [ 15.  ,  30.97], [ 91.  , 231.04], [ 82.  , 207.2 ], [ 46.  , 106.42], [ 61.  , 145.  ], [ 84.  , 209.  ], [ 59.  , 140.91], [ 78.  , 195.08], [ 94.  , 244.  ], [ 88.  , 226.03], [ 37.  ,  85.47], [ 75.  , 186.21], [104.  , 261.  ], [ 45.  , 102.91], [ 86.  , 222.  ], [ 44.  , 101.07], [ 16.  ,  32.07], [ 51.  , 121.75], [ 21.  ,  44.96], [ 34.  ,  78.96], [ 14.  ,  28.09], [ 62.  , 150.36], [ 50.  , 118.71], [ 38.  ,  87.62], [ 73.  , 180.95], [ 65.  , 158.93], [ 43.  ,  98.  ], [ 52.  , 127.6 ], [ 90.  , 232.04], [ 22.  ,  47.88], [ 81.  , 204.38], [ 69.  , 168.93], [ 92.  , 238.03], [ 23.  ,  50.94], [ 74.  , 183.85], [ 54.  , 131.29], [ 39.  ,  88.91], [ 70.  , 173.04], [ 30.  ,  65.39], [ 40.  ,  91.22], [106.  ,   2.01]]
        # aa = {x[0]:x[1] for x in aa}

        chg = check_output(''' awk 'BEGIN{flg=0};/--/{getline;flg=!flg}flg{print $5}' '''+os.path.join(acfpath,'ACF.dat'), shell=1)
        chg = np.genfromtxt(io.BytesIO(chg))
        elemchg = {'Ca': 8, 'H': 1, 'O': 6, 'Si': 4, 'C': 4}  # Ca, H, O, Si, C

        for x in types:
            mass = pt.Element(x[0].name).atomic_mass
            for i, atom in enumerate(x, 1):
                nme = atom.name+str(i)
                atom.type = {'name':nme, 'mass':str(mass.real), 'element':atom.name, 'charge':str(elemchg[atom.name] - chg[i])}
                self.ff.atom_types.append(atom.type)

        self.bonds_typed = OrderedDict()
        self.angles_typed = OrderedDict()
        self.propers_typed = OrderedDict()

        bonds, angles, propers = self.network_b_a_d()
        for x in bonds:
            # if cnt==1:
            mag = bond_length(self.closest_img_bond(*x), 0)
            self.ff.bond_types.append({'k':str(kb), 'length':str(mag[0]), 'type1':x[0].type['name'], 'type2':x[1].type['name']})
            self.bonds_typed[x] = self.ff.bond_types[-1]
            # cnt += 1

        for x in angles:
            mag = bend_angle(self.closest_img_angle(*x), 0)
            if np.isclose(mag, np.pi, atol=.001, rtol=0) or np.isclose(mag, 0, atol=.001):
                continue
            self.ff.angle_types.append({'k':str(ka), 'angle':str(math.degrees(mag[0])), 'type1':x[0].type['name'], 'type2':x[1].type['name'], 'type3':x[2].type['name']})
            self.angles_typed[x] = self.ff.angle_types[-1]

        for x in propers:
            img = self.closest_img_dihed
            coords = img(*x)
            v1 = coords[0] - coords[1]
            v2 = coords[2] - coords[1]
            v3 = coords[3] - coords[2]
            cv1 = cross(v1, v2)
            cv2 = cross(v2, v3)
            close = lambda x, y: np.allclose(x, y, atol=.001, rtol=0)
            if close(cv1, 0) or close(cv2, 0):
                continue
            mag = dihed_angle(coords, 0)
            if close(mag, 0) or close(np.abs(mag), np.pi):
                continue
                
            self.ff.proper_types.append({'k':str(kd), 'phi':str(math.degrees(abs(mag[0]))), 'type1':x[0].type['name'],
                                        'type2':x[1].type['name'], 'type3':x[2].type['name'], 'type4':x[3].type['name']})
            self.propers_typed[x] = self.ff.proper_types[-1]


    def rm_redun_cons(self):
        '''remove redundant connections. Note that we assume here that #of cons=# of con types for b/a/d'''
        nlst = self.particles_label_sorted()
        n = self.n_particles()
        qlist = [*self.bonds_typed, *self.angles_typed, *self.propers_typed]
        B=[]
        removed = []
        angles, diheds = [], []
        for cnt, x in enumerate(qlist):
            idx = []
            for v in x:
                tmp = nlst.index(v)
                idx.extend([3 * tmp, 3 * tmp + 1, 3 * tmp + 2])
            jac = np.zeros(3*n)
            if len(x) == 2:
                img = self.closest_img_bond
                fun = bond_length
            elif len(x) == 3:
                img = self.closest_img_angle
                fun = bend_angle
            else:
                img = self.closest_img_dihed
                fun = dihed_angle
            _, tmp = fun(img(*x), 1)
            jac[idx] = tmp.flatten()

            if matrix_rank([*B, jac]) == len(B) + 1:
                B.append(jac)
            else:
                removed.append(x)
                if len(x) == 2:
                    raise Exception('bonds are not independent')
                elif len(x) == 3:
                    self.ff.angle_types.remove(self.angles_typed[x])
                    self.angles_typed.pop(x)
                else:
                    self.ff.proper_types.remove(self.propers_typed[x])
                    self.propers_typed.pop(x)
        return removed

    def getB(self):
        '''get Wilson B matrix'''
        nlst = self.particles_label_sorted()
        n = self.n_particles()
        qlist = [*self.bonds_typed, *self.angles_typed, *self.propers_typed]
        B=np.zeros([3*n-6, 3*n])
        grad = np.zeros(3*n-6)
        hessian = np.zeros([3*n, 3*n])
        K = deepcopy(hessian)
        for cnt, x in enumerate(qlist):
            idx = []
            for v in x:
                tmp = nlst.index(v)
                idx.extend([3 * tmp, 3 * tmp + 1, 3 * tmp + 2])
            if len(x) == 2:
                img = self.closest_img_bond
                fun = bond_length
                type = self.bonds_typed[x]
                mag0 = float(type['length'])
            elif len(x) == 3:
                img = self.closest_img_angle
                fun = bend_angle
                type = self.angles_typed[x]
                mag0 = math.radians(float(type['angle']))
            else:
                img = self.closest_img_dihed
                fun = dihed_angle
                type = self.propers_typed[x]
                mag0 = math.radians(float(type['phi']))
            mag, fderiv, sderiv = fun(img(*x), 2)
            B[cnt, idx] = fderiv.flatten()
            k = float(type['k'])
            if mag < 0:
                mag *= -1
                sderiv *= -1

            sderiv = sderiv.reshape([-1, 3 * len(x)])
            if len(x) == 4 and (np.isclose(mag, 0, atol=1e-15, rtol=0) or
                                np.isclose(mag, np.pi, atol=1e-15, rtol=0)): #or np.allclose(cv1, 0, atol=.0001) or np.allclose(cv2, 0, atol=.0001)):
                fderiv = np.zeros(fderiv.shape)
                sderiv = np.zeros(sderiv.shape)
            hessian[np.ix_(idx, idx)] = sderiv * 2 * k * (mag - mag0)
            K += hessian
        return B, K


    def hessx_to_internal(self, Hx):
        B, K = self.getB()

        Binv = np.linalg.pinv(B)

        return Binv.T @ (Hx - K) @ Binv   # eq 10 in "the efficient optimiztion of molecular geometries using redundant . . . "



    def gen_angs_and_diheds(self):
        angles = OrderedSet()
        propers = OrderedSet()
        nlst = list(self.particles(0))
        for i, part1 in enumerate(nlst):
            for part2 in nlst[i+1:]:
                try:
                    tmp = nx.all_simple_paths(self.bond_graph, part1, part2, 4) # type: list
                except nx.exception.NodeNotFound:
                    continue
                for ipath in list(tmp):
                    if len(ipath) == 3:
                        angles.add(tuple([x for x in ipath]))
                    elif len(ipath) == 4:
                        propers.add(tuple([x for x in ipath]))
        self.angles_typed = OrderedDict()
        self.propers_typed = OrderedDict()
        for a in angles:
            for atype in self.ff.angle_types:
                vals = [atype[f'type{i}'] for i in range(1,4)]
                if [x.type['name'] for x in a] in (vals,vals[::-1]):
                    self.angles_typed[a] = atype

        for p in propers:
            for ptype in self.ff.proper_types:
                vals = [ptype[f'type{i}'] for i in range(1, 5)]
                if [x.type['name'] for x in p] in (vals, vals[::-1]):
                    self.propers_typed[p] = ptype

    def applyff(self,file):
        from foyer.forcefield import Forcefield
        from foyer.smarts_graph import SMARTSGraph
        from networkx.algorithms.isomorphism import GraphMatcher
        
        self.ff = Forcefield(file)

        self.typemap = {x: {'whitelist': set(), 'blacklist': set(),
                       'atomtype': None} for x in self.particles(0)}

        for x in self.ff.atom_types:
            x['graph'] = SMARTSGraph(smarts_string=x['def'],
                                     parser=self.ff.parser,
                                     name=x['name'],
                                     overrides=x['overrides'],
                                     typemap=self.typemap)

        nlst = list(self.particles(0))
        # initialize atom types
        for x in nlst:
            x.type=[]

        for n in nlst:
            self.bond_graph.add_node(n,atom=n)

        for type in self.ff.atom_types:
            gm = GraphMatcher(self.bond_graph,type['graph'],node_match=self._node_match)
            matches = {y[0] for x in gm.subgraph_isomorphisms_iter() for y in x.items() if y[1] == 0}
            for atom in matches:
                atom.type.append(type)

        for x in nlst:
            for y in [z['overrides'] for z in x.type if z['overrides']]:
                [x.type.remove(z) for z in x.type if z['name'] == y]
            x.type = x.type[0]

        # create angles and dihedrals
        self.gen_angs_and_diheds()

        self.nonbond_typed = OrderedDict()
        self.bonds_typed = OrderedDict()

        for p in self.particles():
            for nb_type in self.ff.nonbond_types:
                if p.type['name'] == nb_type['type']:
                    self.nonbond_typed[p] = nb_type
                    break

        for b in self.bond_graph.edges:
            for btype in self.ff.bond_types:
                if {x.type['name'] for x in b} == {btype[x] for x in ['type1','type2']}:
                    self.bonds_typed[b] = btype

    def write_lammpsdata(self, filename='data.dat', lj=1, elec=1, bond=1, angle=1, prop=1, atom_style='full',
                        unit_style='real', buff=0):
        """Output a LAMMPS data file.

        Parameters
        ----------
        filename : str
            Path of the output file
        atom_style: str
            Defines the style of atoms to be saved in a LAMMPS data file. The following atom
            styles are currently supported: 'full', 'atomic', 'charge', 'molecular'
            information on atom styles.
        unit_style: str
            Defines to unit style to be save in a LAMMPS data file.  Defaults to 'real' units.
            Current styles are supported: 'real', 'lj'
        """

        # Convert coordinates to LJ units
        if unit_style == 'lj':
            # Get sigma, mass, and epsilon conversions by finding maximum of each
            sigma_conversion_factor = np.max([atom.sigma for atom in structure.atoms])
            epsilon_conversion_factor = np.max([atom.epsilon for atom in structure.atoms])
            mass_conversion_factor = np.max([atom.mass for atom in structure.atoms])

            xyz = xyz / sigma_conversion_factor
            charges = (charges*1.6021e-19) / np.sqrt(4*np.pi*(sigma_conversion_factor*1e-10)*
              (epsilon_conversion_factor*4184)*epsilon_0)
            charges[np.isinf(charges)] = 0
            # TODO: FIX CHARGE UNIT CONVERSION
        else:
            sigma_conversion_factor = 1
            epsilon_conversion_factor = 1
            mass_conversion_factor = 1

        with open(filename, 'w') as data:
            data.write(f'''{filename} - created by mBuild; units = {unit_style}
{self.n_particles(0)} atoms
{len(self.bonds_typed)} bonds
{len(self.angles_typed)} angles
{len(self.propers_typed)} dihedrals

{len(self.ff.atom_types)} atom types
{len(self.ff.bond_types)} bond types
{len(self.ff.angle_types)} angle types
{len(self.ff.proper_types)} dihedral types\n\n''')

            lo = np.min(self.xyz,axis=0)
            hi = np.diag(self.latmat)
            for i, dim in enumerate(['x', 'y', 'z']):
                data.write(f'{lo[i]-buff:.6f} {lo[i]+hi[i]+buff:.6f} {dim}lo {dim}hi\n')
            # Box data
            if not np.allclose(self.latmat-np.diag(np.diagonal(self.latmat)),np.zeros([3,3])):
                data.write('{0:.6f} {1:.6f} {2:6f} xy xz yz\n'.format(
                    *latmat[np.tril_indices(3,k=-1)].tolist()))

            data.write('\nMasses\n\n')
            for i,type in enumerate(self.ff.atom_types,1):
                data.write(f'{i}\t{type["mass"]}\t# {type["name"]}\n')


            data.write('\nPair Coeffs # real \n')
            data.write('#\tepsilon (kcal/mol)\t\tsigma (Angstrom)\n')

            if self.ff.nonbond_types:
                for i,y in enumerate(self.ff.atom_types,1):
                    nb = [x for x in self.ff.nonbond_types if x['type'] == y['name']][0]
                    data.write(f"{i}\t{nb['epsilon'] if lj else 0}\t\t{nb['sigma']}\t\t# {nb['type']}\n")

            # Bond coefficients
            data.write('\nBond Coeffs # harmonic\n')
            data.write('#\tk(kcal/mol/angstrom^2)\t\treq(angstrom)\n')
            for i,y in enumerate(self.ff.bond_types,1):
                data.write(f"{i}\t{y['k'] if bond else 0}\t{y['length']}\t# {y['type1']}\t{y['type2']}\n")

            # Angle coefficients
            if self.ff.angle_types:
                data.write('\nAngle Coeffs # harmonic\n')
                data.write('#\tk(kcal/mol/rad^2)\t\ttheteq(deg)\tk(kcal/mol/angstrom^2)\treq(angstrom)\n')
                for i,y in enumerate(self.ff.angle_types,1):
                    data.write(f"{i}\t{y['k'] if angle else 0}\t\t{y['angle']}\t# {y['type1']}\t{y['type2']}\t{y['type3']}\n")

            # Dihedral coefficients
            if self.ff.proper_types:
                data.write('\nDihedral Coeffs # harmonic\n')
                data.write('#k, d, n\n')
                for i,y in enumerate(self.ff.proper_types,1):
                    keys = [x for x in y.keys() if 'type' not in x]
                    data.write(f"{i}\t "+(len(keys)*'{} ').format(*[y[z] for z in keys])
                               +f"\t# {y['type1']}\t{y['type2']}\t{y['type3']}\t{y['type4']}\n")

            # Atom data
            data.write('\nAtoms\n\n')
            if atom_style == 'atomic':
                atom_line = '{index:d}\t{type_index:d}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n'
            elif atom_style == 'charge':
                if unit_style == 'real':
                    atom_line = '{index:d}\t{type_index:d}\t{charge:.6f}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n'
                elif unit_style == 'lj':
                    atom_line = '{index:d}\t{type_index:d}\t{charge:.4ef}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n'
            elif atom_style == 'molecular':
                atom_line = '{index:d}\t{zero:d}\t{type_index:d}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n'
            elif atom_style == 'full':
                if unit_style == 'real':
                    atom_line ='{index}\t{zero}\t{type_index}\t{charge}\t{x:.7f}\t{y:.7f}\t{z:.7f}\t#{t}\n'
                elif unit_style == 'lj':
                    atom_line ='{index:d}\t{zero:d}\t{type_index:d}\t{charge:.4e}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n'

            nlst = self.particles_label_sorted()
            comps = list(nx.connected_components(self.bond_graph))

            for i,p in enumerate(nlst,1):
                data.write(atom_line.format(
                    index=i,type_index=self.ff.atom_types.index(p.type)+1,
                    zero=[i for i in range(len(comps)) if p in comps[i]][0],
                    charge=[x['charge'] for x in self.ff.nonbond_types if p.type['name'] == x['type']][0] if elec else 0,
                    x=p.pos[0],y=p.pos[1],z=p.pos[2],t=p.type['name']))

            if atom_style in ['full', 'molecular']:
                # Bond data
                # if bond:
                data.write('\nBonds\n\n')
                for i,bond in enumerate(self.bonds_typed.items(),1):
                    bidx=[nlst.index(j)+1 for j in bond[0]]
                    data.write(f"{i}\t{self.ff.bond_types.index(bond[1])+1}\t"
                               f"{bidx[0]}\t{bidx[1]}\n")

                # Angle data
                if self.ff.angle_types and self.angles_typed:
                    data.write('\nAngles\n\n')
                    for i, angle in enumerate(self.angles_typed.items(), 1):
                        aidx=[nlst.index(j)+1 for j in angle[0]]
                        data.write(f"{i}\t{self.ff.angle_types.index(angle[1])+1}\t"
                                   f"{aidx[0]}\t{aidx[1]}\t{aidx[2]}\n")

                # Dihedral data
                if self.ff.proper_types and self.propers_typed:
                    data.write('\nDihedrals\n\n')
                    for i,prop in enumerate(self.propers_typed.items(),1):
                        pidx=[nlst.index(j)+1 for j in prop[0]]
                        data.write(f"{i}\t{self.ff.proper_types.index(prop[1])+1}\t"
                                   f"{pidx[0]}\t{pidx[1]}\t{pidx[2]}\t{pidx[3]}\n")

    def get_bond_lengths(self, bonds):
        out = []
        for p in bonds:
            v1 = p[0]._pos - p[1]._pos
            out.append(np.linalg.norm(v1))
        return out

    def get_angles(self, angles, units='degrees'):
        '''angles:list of particles'''
        import math
        out = []
        for p in angles:
            v1 = p[0].pos - p[1].pos
            v2 = p[2].pos - p[1].pos
            d = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
            d = min(d, 1)
            d = max(d, -1)
            angle = math.acos(d)
            if units == "degrees":
                out.append(math.degrees(angle))
            elif units == "radians":
                out.append(angle)
            else:
                raise ValueError("Invalid units {}".format(units))
        return out

    @staticmethod
    def import_(module):
        """Import a module, and issue a nice message to stderr if the module isn't installed.
        
        Parameters
        ----------
        module : str
            The module you'd like to import, as a string
        
        Returns
        -------
        module : {module, object}
            The module object
        
        Examples
        --------
        >>> # the following two lines are equivalent. the difference is that the
        >>> # second will check for an ImportError and print you a very nice
        >>> # user-facing message about what's wrong (where you can install the
        >>> # module from, etc) if the import fails
        >>> import tables
        >>> tables = import_('tables')
        
        Notes
        -----
        The pybel/openbabel block is meant to resolve compatibility between
        openbabel 2.x and 3.0.  There may be other breaking changes but the change
        in importing them is the major one we are aware of. For details, see
        https://open-babel.readthedocs.io/en/latest/UseTheLibrary/migration.html#python-module
        """
        if module == 'pybel':
            try:
                return importlib.import_module('openbabel.pybel')
            except ModuleNotFoundError:
                pass
            try:
                pybel = importlib.import_module('pybel')
                msg = ('openbabel 2.0 detected and will be dropped in a future '
                       'release. Consider upgrading to 3.x.')
                warnings.warn(msg, DeprecationWarning)
                return pybel
            except ModuleNotFoundError:
                pass
        try:
            return importlib.import_module(module)
        except ImportError as e:
            try:
                message = MESSAGES[module]
            except KeyError:
                message = 'The code at {filename}:{line_number} requires the ' + module + ' package'
                e = ImportError('No module named %s' % module)
        
            frame, filename, line_number, function_name, lines, index = \
                inspect.getouterframes(inspect.currentframe())[1]
        
            m = message.format(filename=os.path.basename(filename), line_number=line_number)
            m = textwrap.dedent(m)
        
            bar = '\033[91m' + '#' * max(len(line) for line in m.split(os.linesep)) + '\033[0m'
        
            print('', file=sys.stderr)
            print(bar, file=sys.stderr)
            print(m, file=sys.stderr)
            print(bar, file=sys.stderr)
            raise DelayImportError(m)

    def solvate(self, solvent, n_solvent, overlap=2,
                seed=12345, edge=0.2, fix_orientation=False, temp_file=None,
                update_port_locations=False):
        """Solvate a compound in a box of solvent using packmol.

        Parameters
        ----------
        solute : mb.Compound
            Compound to be placed in a box and solvated.
        solvent : mb.Compound
            Compound to solvate the box.
        n_solvent : int
            Number of solvents to be put in box.
        box : mb.Box
            Box to be filled by compounds.
        overlap : float, units nm, default=0.2
            Minimum separation between atoms of different molecules.
        seed : int, default=12345
            Random seed to be passed to PACKMOL.
        edge : float, units nm, default=0.2
            Buffer at the edge of the box to not place molecules. This is necessary
            in some systems because PACKMOL does not account for periodic boundary
            conditions in its optimization.
        fix_orientation : bool
            Specify if solvent should not be rotated when filling box,
            default=False.
        temp_file : str, default=None
            File name to write PACKMOL's raw output to.
        update_port_locations : bool, default=False
            After packing, port locations can be updated, but since compounds
            can be rotated, port orientation may be incorrect.

        Returns
        -------
        solvated : mb.Compound

        """
        if not isinstance(solvent, (list, set)):
            solvent = [solvent]
        if not isinstance(n_solvent, (list, set)):
            n_solvent = [n_solvent]
        if not isinstance(fix_orientation, (list, set)):
            fix_orientation = [fix_orientation] * len(solvent)

        if len(solvent) != len(n_solvent):
            msg = ("`n_solvent` and `n_solvent` must be of equal length.")
            raise ValueError(msg)
        # In angstroms for packmol.
        box_mins = np.array([self.box.xlo_bound, self.box.ylo_bound, self.box.zlo_bound])
        box_maxs = np.array([self.box.xhi_bound, self.box.yhi_bound, self.box.zhi_bound])
        center_solute = (box_maxs + box_mins) / 2

        # Apply edge buffer
        box_maxs -= edge

        # Build the input file for each compound and call packmol.
        solute_xyz, solvated_xyz = [open(os.path.abspath(x), 'w') for x in ('tmp_solute.xyz', 'tmp_solvated.xyz')]#tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)

        # generate list of temp files for the solvents
        solvent_xyz_list = list()
        self.write_xyz(solute_xyz)
        input_text = (f'tolerance {overlap:.16f} \n'
                      f'filetype xyz \n'
                      f'output {solvated_xyz.name} \n'
                      f'seed {seed}\n') #+
# """
# structure {0}
#     number 1
#     center
#     fixed {1:.3f} {2:.3f} {3:.3f} 0. 0. 0.
# end structure
# """.format(solute_xyz.name, *center_solute))

        for i,(solv, m_solvent, rotate) in enumerate(zip(solvent, n_solvent, fix_orientation)):
             m_solvent = int(m_solvent)

             solvent_xyz = open(os.path.abspath('tmp_solvent_'+str(i)+'.xyz'), 'w')#tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)

             solv.write_xyz(solvent_xyz.name)
             input_text += """
structure {0}
    number {1:d}
    inside box {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:.3f} {7:.3f}
end structure
""".format(solvent_xyz.name, m_solvent, *box_mins, *box_maxs)
        self._run_packmol(input_text, solvated_xyz)

        tmp = compload(solvated_xyz.name)
        tmp.latmat = self.latmat
        initself = deepcopy(self)
        added = [] #future: remove water molecules that are close due to periodic boundary conditions.
        for comp, m_compound in zip(solvent, n_solvent):

            for i in range(m_compound):
                coords = tmp.xyz[i*comp.n_particles():(i+1)*comp.n_particles()]
                _,out = self.neighbs(coords,initself.particles(), rcut=1.5)

                if not list(chain.from_iterable(out)):
                    cop = deepcopy(comp)
                    cop.xyz = coords
                    self.add(cop)
                    added.extend(cop)

        for f in [solvent_xyz, solvated_xyz, solute_xyz]:
            f.close()

        from glob import glob
        [os.remove(x) for x in glob('tmp_*')]

    def _run_packmol(self, input_text, filled_xyz):
        """Call PACKMOL to pack system based on the input text.
        
        Parameters
        ----------
        input_text : str, required
            String formatted in the input file syntax for PACKMOL.
        filled_xyz : `tempfile` object, required
            Tempfile that will store the results of PACKMOL's packing.
        """
        # Create input file
        f = open(os.path.abspath('tmp_input.inp'), 'w')
        f.write(input_text)
        f.close()

        from subprocess import PIPE, Popen

        proc = Popen('{} < {}'.format('~/software/packmol/packmol', f.name),
                     stdin=PIPE, stdout=PIPE, stderr=PIPE,
                     universal_newlines=True, shell=True)
        out, err = proc.communicate()
        
        # if 'WITHOUT PERFECT PACKING' in out:
        #     msg = ("Packmol finished with imperfect packing. Using "
        #            "the .xyz_FORCED file instead. This may not be a "
        #            "sufficient packing result.")
        #     warnings.warn(msg)
        #     os.system('cp {0}_forced {0}'.format(filled_xyz.name))
        
        if 'ERROR' in out or proc.returncode != 0:
            _packmol_error(out, err)

    def total_charge(self, group=None):
        '''only makes sense after applyff
        group: list of particles'''
        if not isinstance(group, Iterable) and group:
            group = [group]

        particles = group if group else self.particles()

        return np.sum(np.array([self.nonbond_typed[x]['charge'] for x in particles], dtype=float))
    
    
    def total_dipole(self, dir, group=None):
        '''only makes sense after applyff'''
        
        particles = group if group else self.particles()
        
        total_dipole = 0.0
        for i, p in enumerate(particles):
            total_dipole += eval([x['charge'] for x in self.ff.nonbond_types if p.type['name'] == x['type']][0]) * p.pos[dir] 
        return total_dipole        
    

    def neutralize(self, types=None):
        '''only makes sense after applyff'''
        if not types:
            types = self.ff.atom_types
        cnt = 0
        for x in self.particles():
            if x.type in types:
                cnt += 1

        adjust = -self.total_charge()/cnt
        for x in types:
            for y in self.ff.nonbond_types:
                if x['name'] == y['type']:
                   y['charge'] = str(float(y['charge']) + adjust)

    def write_gulp(self, direc='.', ordered=1,
                   keys='''opti conp molmec fix noautobond nosym norepulsive_cutoff prop thermal  & \nfreq  eigen lower optlower kcal conj\n\n'''):

        types = equivalence_classes(self.ff.atom_types, lambda x,y: x['element']==y['element'])
        
        type_map = dict()
        for x in types:
            for i,y in enumerate(x, 1):
                type_map[y['name']] = y['element']+str(i)
        with open(os.path.join(direc, 'gulp.in'), 'w') as f:
            f.write(keys)
            f.write('\n\nelement\n')
            for x in types:
                f.write(f"mass {x[0]['element']} {x[0]['mass']}\n")
            f.write("end \n\nvector\n")
            f.write((3*(3*'{} '+'\n')).format(*self.latmat.flatten()))
            
            f.write('\ncart\n')
            for x in self.particles_label_sorted() if ordered else self.particles():
                f.write(f'{type_map[x.type["name"]]:4}' + (3*'{:7f}  '+'\n').format(*x.pos))

            f.write('\n')
            bonds,_ = self.bonds_angles_index()
            for x in bonds + 1:
                f.write('connect {} {}\n'.format(*x))

            if self.ff.nonbond_types:
                f.write("\nepsilon/sigma kcal\n")
                for x in self.ff.nonbond_types:
                    f.write(f"{type_map[x['type']]}  {x['epsilon']}  {x['sigma']} \n")

                f.write("\nlenn epsilon zero product 12  6 x13 kcal all\n")
                f.write('12\n')
                
            f.write("\nharmonic intra bond kcal\n" if self.ff.bond_types else '')
            for x in self.ff.bond_types:
                f.write(f"{type_map[x['type1']]} {type_map[x['type2']]}  {2*float(x['k'])}  {x['length']}  0   \t# {x['type1']}\t{x['type2']}\n")

            f.write("\nthree bond intra regular kcal\n" if self.ff.angle_types else '')
            for x in self.ff.angle_types:
                f.write(f"{type_map[x['type2']]} {type_map[x['type1']]}  {type_map[x['type3']]}  {2*float(x['k'])}  {x['angle']}  \t# {x['type1']}\t{x['type2']}\t{x['type3']}\n")

            f.write('\n torharm intra bond kcal\n' if self.ff.proper_types else '')
            for x in self.ff.proper_types:
                keys = [y for y in x.keys() if 'type' not in y]
                f.write(f"{type_map[x['type1']]} {type_map[x['type2']]}  {type_map[x['type3']]} {type_map[x['type4']]} "+ (len(keys)*'{} ').format(*[x[z] for z in keys]) +f"\t# {x['type1']}\t{x['type2']}\t{x['type3']}\t{x['type4']}\n")
            f.write('\n slower .05')
            f.write('\noutput movie xyz')


    def rotate_vecs(self, v1, v2, pnt=None):
        ''' rotates the particles according to the angle between v1 and v2'''
        v1, v2 = map(np.array, [v1, v2])
        v1, v2 = map(lambda x:x/norm(x), [v1, v2])
        normal = np.cross(v1, v2)
        if np.allclose(normal, 0):
            if not np.array_equal(pnt, None):
                self.reflect(pnt, v1)
            return
        rot = R.from_rotvec(normal*angle_between_vecs(v1, v2, degrees=0)/norm(normal))
        self.xyz_with_ports = rot.apply(self.xyz_with_ports)
        for x in [x for x in self.particles(1) if '!' in x.name]:
            x.orientation = rot.apply(x.orientation)

    def rotate_around(self, vec, ang, degrees=1):
        ''' rotate compound about an axis'''
        vec = np.array(vec)
        vec = vec/norm(vec)
        if degrees:
            ang = math.radians(ang)
        rot = R.from_rotvec(vec*ang)
        self.xyz_with_ports = rot.apply(self.xyz_with_ports)
        for x in [x for x in self.particles(1) if '!' in x.name]:
            x.orientation = rot.apply(x.orientation)



class Box:
    def __init__(self, comp):
        self.comp = comp

    @property
    def lx(self):
        return self.comp.latmat[0, 0]

    @property
    def ly(self):
        return self.comp.latmat[1, 1]

    @property
    def lz(self):
        return self.comp.latmat[2, 2]

    @property
    def xy(self):
        return self.comp.latmat[1, 0]

    @property
    def xz(self):
        return self.comp.latmat[2, 0]

    @property
    def yz(self):
        return self.comp.latmat[2, 1]

    @property
    def xlo(self):
        return np.min(self.comp.xyz[:, 0]) - .01

    @property
    def ylo(self):
        return np.min(self.comp.xyz[:, 1]) - .01

    @property
    def zlo(self):
        return np.min(self.comp.xyz[:, 2]) - .01

    @property
    def xhi(self):
        return self.xlo + self.lx

    @property
    def yhi(self):
        return self.ylo + self.ly

    @property
    def zhi(self):
        return self.zlo + self.lz

    # self.lx,self.ly,self.lz = np.diag(comp.latmat)
    # self.xy,self.xz,self.yz = comp.latmat[np.tril_indices(3, k=-1)]

    @property
    def xlo_bound(self):
        return self.xlo + np.min([0, self.xy, self.xz, self.xy + self.xz])

    @property
    def ylo_bound(self):
        return self.ylo + np.min([0, self.yz])


    @property
    def xhi_bound(self):
        return self.xhi + np.max([0.0, self.xy, self.xz, self.xy + self.xz])

    @property
    def yhi_bound(self):
        return self.yhi + np.max([0.0, self.yz])

    zlo_bound, zhi_bound = zlo, zhi

    a = lx

    @property
    def b(self):
        return norm(self.comp.latmat[1])

    @property
    def c(self):
        return norm(self.comp.latmat[2])

    @property
    def alph(self):
        return angle_between_vecs(self.comp.latmat[1], self.comp.latmat[2])

    @property
    def bet(self):
        return angle_between_vecs(self.comp.latmat[0], self.comp.latmat[2])

    @property
    def gam(self):
        return angle_between_vecs(self.comp.latmat[0], self.comp.latmat[1])


class Port(Compound):
    """A set of four ghost Particles used to connect parts.

    Parameters
    ----------
    anchor : mb.Particle, optional, default=None
        A Particle associated with the port. Used to form bonds.
    orientation : array-like, shape=(3,)
        Vector along which to orient the port
    anchor : mb.Particle, optional, default=None
        A Particle associated with the port. Used to form bonds.

    """
    def __init__(self, anchor=None, orientation=[1, 0, 0], pos=[0, 0, 0], name='p!'):

        orientation, loc_vec = map(np.asarray, [orientation, pos])

        super(Port, self).__init__(name=name, pos=pos)
        self.anchor, self.orientation = anchor, orientation / norm(orientation)
        
class FF():
    """
    """
    def __init__(self):
        
        self.atom_types,self.nonbond_types,self.bond_types,self.angle_types,self.proper_types = [[], [], [], [], []]
        
    def read_xml(self, file):
        # start of my additions:
        ff = et.parse(file) # type: et.ElementTree
        self.atom_types = [defaultdict(lambda: None,x.attrib) for x in ff.getroot().find('AtomTypes').getchildren()]
        try:
            self.nonbond_types = [defaultdict(lambda: None,x.attrib) for x in ff.getroot().find('NonbondedForce').getchildren()]
        except:
            self.nonbond_types = []
        try:
            self.bond_types = [defaultdict(lambda: None,x.attrib) for x in ff.getroot().find('HarmonicBondForce').getchildren()]
        except:
            self.bond_types = []
        try:
            self.angle_types = [defaultdict(lambda: None,x.attrib) for x in ff.getroot().find('HarmonicAngleForce').getchildren()]
        except AttributeError:
            self.angle_types = []
        tmp = ff.getroot().find('PeriodicTorsionForce')
        try:
            self.proper_types = [defaultdict(lambda: None,x.attrib) for x in tmp.getchildren()] if len(tmp) else []
        except:
            self.proper_types = []

        # self.parser = smarts.SMARTS(self.non_element_types)
