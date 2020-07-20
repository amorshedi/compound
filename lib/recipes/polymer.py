import itertools as it

from compound import Compound,clone
from utils.validation import assert_port_exists
from copy import deepcopy

def Polymer(monomers, n):

    comp = Compound()

    first_part = 1
    for i in range(n):
        this_part = deepcopy(monomers)
        comp.add(this_part,expand=0)
        if first_part:
            first_part = 0
        else:
            comp.force_overlap(comp[-1], comp[-1]['p1'], comp[-2]['p2'])
    return comp