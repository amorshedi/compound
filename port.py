
from compound import Compound, Particle,clone
from coordinate_transform import unit_vector, angle
from myimports import *



class Port(Compound):
    """A set of four ghost Particles used to connect parts.

    Parameters
    ----------
    anchor : mb.Particle, optional, default=None
        A Particle associated with the port. Used to form bonds.
    orientation : array-like, shape=(3,), optional, default=[0, 1, 0]
        Vector along which to orient the port
    separation : float, optional, default=0
        Distance to shift port along the orientation vector from the anchor
        particle position. If no anchor is provided, the port will be shifted
        from the origin.

    Attributes
    ----------
    anchor : mb.Particle, optional, default=None
        A Particle associated with the port. Used to form bonds.
    up : mb.Compound
        Collection of 4 ghost particles used to perform equivalence transforms.
        Faces the opposite direction as self['down'].
    down : mb.Compound
        Collection of 4 ghost particles used to perform equivalence transforms.
        Faces the opposite direction as self['up'].
    used : bool
        Status of whether a port has been occupied following an equivalence
        transform.

    """
    def __init__(self, anchor=None, orientation=None, separation=0,type='up',name='port'):
        super(Port, self).__init__(name=name)
        self.port_particle = 1

        if not anchor:
            self.anchor = [0,0,0]
        else:
            self.anchor = anchor

        default_direction = np.array([0, 1, 0])
        if orientation is None:
            orientation = default_direction
        else:
            orientation = np.asarray(orientation)

        coords = [[0.05, 0.025, -0.025],
                  [0.05, 0.225, -0.025],
                  [-0.15, -0.075, -0.025],
                  [0.05, -0.175, 0.075]]

        for x in coords:
            self.add(Compound(name='_p',pos=x))

        if orientation is not default_direction:
            normal = np.cross(default_direction, orientation)
            self.rotate(angle(default_direction, orientation), normal)

        if type is not 'up':
            self.xyz_with_ports = self.reflect(self.xyz_with_ports,np.zeros(3),orientation)
        try:
            self.translate_to(self.anchor.pos+separation*unit_vector(orientation))
        except:
            self.translate_to(self.anchor+separation*unit_vector(orientation))
        self.orientation = orientation

    def _clone(self, clone_of=None, root_container=None):
        newone = super(Port, self)._clone(clone_of, root_container)
        newone.anchor = clone(self.anchor, clone_of, root_container)
        newone.used = self.used
        return newone

    @property
    def direction(self):
        """The unit vector pointing in the 'direction' of the Port
        """
        return unit_vector(self.xyz_with_ports[1]-self.xyz_with_ports[0])


    def __repr__(self):
        descr = list('<')
        descr.append(self.name + ', ')

        # if self.anchor:
        #     descr.append("anchor: '{}', ".format(self.anchor.name))
        # else:
        #     descr.append('anchor: None, ')

        # descr.append('labels: {}, '.format(', '.join(self.access_labels)))

        descr.append('id: {}>'.format(id(self)))
        return ''.join(descr)
