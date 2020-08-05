import numpy as np
from collections import OrderedDict, defaultdict, Iterable
from copy import deepcopy
import itertools
import os
import sys
import tempfile
from warnings import warn

import mdtraj as md
from mdtraj.core.element import get_by_symbol
from oset import oset as OrderedSet
import parmed as pmd
from parmed.periodic_table import AtomicNum, element_by_name, Mass, Element

from os import system as syst

# from openbabel import openbabel as ob
# from openbabel import pybel as pb
from itertools import compress as cmp
from pymatgen.util.coord import pbc_shortest_vectors, get_angle
from pymatgen.core import Structure
from itertools import combinations

import mdtraj.geometry as geom
from pymatgen import Lattice, Structure
# from openbabel import pybel

from numpy.linalg import inv, norm, det


