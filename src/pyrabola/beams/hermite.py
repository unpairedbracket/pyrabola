import numpy as np

from . import Beam
from ..propagators.hermite import Hermite
from .. import util

from scipy.interpolate import interp2d, RegularGridInterpolator

import matplotlib.pyplot as plt

class BeamHermite(Beam):
    '''
    A beam class suitable for both collimated and focusing propagation.
    Models the beam profile by decomposition into Hermite modes.
    '''
    hermite = None
    coeffs = None


    def __init__(self, position, normal, wavenumber, beam_width, u, v, focal_length, e_field, initialise=True):
        '''
        For this beam type, `position` should be the position of the focusing optic
        The focal point is located at `position + focal_length * normal`
        beam_width is defined at the focusing optic, not at the focal point. For a
        Gaussian beam profile it should be equal to the 1/e field radius of the beam.
        u, v are also defined at the focusing optic.
        e_field is taken to be equal to the E field at infinity up to a factor w0/w(z),
        when expressed as a function of (u, v)/w(z), which is true for focal_length >> z_R
        Therefore you should make sure that focal_length >> z_R if you want this to be accurate
        '''

        super().__init__(position, normal, wavenumber)
        if initialise:
            self.e_field_current = e_field
            F0 = focal_length / (2 * beam_width)
            # F_actual = F0 * ( (1/2 (k0 W / 4F0)^2) - sqrt( (1/2 (k0 W / 4F0)^2)^2 - (k0 W / 4F0)^2 ) )
            self.beam_waist = 4 * F0 / self.wavenumber
            self.rayleigh = 8 * F0**2 / self.wavenumber
            self.beam_z = -focal_length
            self.u_norm = u / beam_width
            self.v_norm = v / beam_width
            self.psi_0 = np.arctan(self.beam_z / self.rayleigh)
            self.hermite = Hermite(self.u_norm, self.v_norm, 900)
            field_factor = np.sqrt(1 + (self.beam_z / self.rayleigh)**2)
            self.coeffs = self.hermite.decompose(e_field * field_factor)

    def setup(self, waist, rayleigh, beam_z, u_norm, v_norm, psi_0, hermite, coeffs, e_field):
            self.beam_waist = waist
            self.rayleigh = rayleigh
            self.beam_z = beam_z
            self.u_norm = u_norm.copy()
            self.v_norm = v_norm.copy()
            self.psi_0 = psi_0
            self.hermite = hermite.copy()
            self.coeffs = coeffs.copy()
            self.e_field_current = e_field.copy()


    def copy_with(self, position, normal):
        B = BeamHermite(position, normal, self.wavenumber,
            None, None, None, None, None, False)
        B.setup(
            self.beam_waist, self.rayleigh, self.beam_z, self.u_norm,
            self.v_norm, self.psi_0, self.hermite, self.coeffs, self.e_field_current
        )
        return B

    def width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def max_width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def e_field(self, u, v):
        beam_width = self.width()
        u_norm = u / beam_width
        v_norm = v / beam_width

        u_full = np.linspace(u_norm.min(), u_norm.max(), u.shape[0]*4)
        v_full = np.linspace(v_norm.min(), v_norm.max(), v.shape[1]*4)

        H_reconstruct = Hermite(u_full, v_full, self.hermite.bandlimit)

        psi_new = np.arctan(self.beam_z / self.rayleigh)
        self.coeffs[:] = 0
        self.coeffs[-1,-1] = 1
        e_full = H_reconstruct.recompose(self.coeffs * self.hermite.get_phases(psi_new - self.psi_0))
        actual_terp = RegularGridInterpolator((u_full, v_full), e_full, bounds_error=False, fill_value=0)
        e_norm = actual_terp((u_norm, v_norm))
        
        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(H_reconstruct.U)
        #axs[1].imshow(H_reconstruct.V)
        #plt.show()

        return e_norm / np.sqrt(1 + (self.beam_z / self.rayleigh)**2)

    def propagate_z(self, z):
        super().propagate_z(z)

        psi_new = np.arctan(self.beam_z / self.rayleigh)

        self.e_field_current = self.hermite.recompose(self.coeffs * self.hermite.get_phases(psi_new - self.psi_0))

        self.e_r_interpolant = interp2d(self.u_norm, self.v_norm, self.e_field_current.T.real, bounds_error=False, fill_value=0)
        self.e_i_interpolant = interp2d(self.u_norm, self.v_norm, self.e_field_current.T.imag, bounds_error=False, fill_value=0)
