from compound import Compound,compload
from port import Port

def CH3(name='ch3',ptype='down'):

    comp = Compound(name=name)
    compload('ch3.pdb',compound=comp,relative_to_module=__file__, infer_hierarchy=False)
    comp.translate(-comp[0].pos)  # Move carbon to origin.

    comp.add(Port(anchor=comp[0],name='pch3',type=ptype),expand=0)
    comp['pch3'].translate([0, -0.7, 0])
    return comp

if __name__ == '__main__':
    m = CH3()
    m.save('ch3.mol2', overwrite=True)
