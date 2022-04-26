import numpy as np

from .beams import BeamGaussianGeometric
from . import parser

class BeamInjection:
    @staticmethod
    def from_dict(config_dict):
        position = parser.position(config_dict['position'])
        direction = parser.normal(config_dict['direction'])
        wavelength = config_dict['wavelength']
        beam_type = BeamGaussianGeometric
        beam_waist = config_dict['beam_waist']

        return BeamInjection(
            position,
            direction,
            wavelength,
            beam_type,
            beam_waist=beam_waist,
            distance_to_focus=0
        )

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
