class Beam:
    position = None
    normal = None
    wavenumber = 0
    twist = 0
    beam_z = 0
    clipping = False

    def __init__(self, position, normal, wavenumber):
        self.position = position
        self.normal = normal
        self.wavenumber = wavenumber

    def copy_with(self, position, normal):
        return Beam(position, normal, self.wavenumber)

    def width(self):
        return 0

    def max_width(self):
        return 3*self.width()

    def e_field(self, x, y):
        raise NotImplementedError()

    def propagate_z(self, z):
        self.beam_z += z
        self.position += z * self.normal

    def clip(self, clip_information):
        pass


class BeamMissed(Beam):
    def __init__(self):
        pass
