from compound import Compound
from lib.moieties import CH2
from lib.moieties import CH3
from lib.recipes.polymer import Polymer

class Alkane(Compound):
    """An alkane which may optionally end with a hydrogen or a Port."""
    def __init__(self, n=3, cap_front=True, cap_end=True):
        """Initialize an Alkane Compound.

        Args:
            n: Number of carbon atoms.
            cap_front: Add methyl group to beginning of chain ('down' port).
            cap_end: Add methyl group to end of chain ('up' port).
        """
        if n < 2:
            raise ValueError('n must be 1 or more')
        super(Alkane, self).__init__()

        # Adjust length of Polmyer for absence of methyl terminations.
        if not cap_front:
            n += 1
        if not cap_end:
            n += 1
        chain = Polymer(CH2(), n=n-2)
        self.add(chain)

        if cap_front:
            self.add(CH3(),expand=0)
            self.force_overlap(move_this=self['ch3'],
                             from_positions=self['ch3']['pch3'],
                             to_positions=self[-2]['p2'])

        if cap_end:
            self.add(CH3(name='ch3_2',ptype='up'),expand=0)
            self.force_overlap(move_this=self['ch3_2'],
                             from_positions=self['ch3_2']['pch3'],
                             to_positions=self[0]['p1'])