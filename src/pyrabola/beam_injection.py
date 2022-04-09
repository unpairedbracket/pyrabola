import numpy as np


class BeamInjection:
    position = None
    beam_normal = None
    wavenumber = None
    beam_parameters = {}
    beam_type = None

    def __init__(self, position, direction, wavelength,
                 beam_type, **beam_parameters):
        self.position = position
        self.beam_normal = direction
        self.wavenumber = 2*np.pi / wavelength
        self.beam_type = beam_type
        self.beam_parameters = beam_parameters

    def get_beam(self):
        return self.beam_type(
            position=self.position,
            normal=self.beam_normal,
            wavenumber=self.wavenumber,
            **self.beam_parameters
        )
