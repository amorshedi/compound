from compound import Compound,compload
from port import Port

def CH2():

    comp = Compound(name='ch2')
    compload('ch2.pdb', compound=comp,infer_hierarchy=False,relative_to_module=__file__)
    comp.translate(-comp[0].pos)  # Move carbon to origin.

    comp.add(Port(anchor=comp[0],name='p1'),expand=0)
    comp['p1'].translate([0, 0.7, 0])

    comp.add(Port(anchor=comp[0],name='p2'),expand=0)
    comp['p2'].translate([0, -0.7, 0])
    return comp
