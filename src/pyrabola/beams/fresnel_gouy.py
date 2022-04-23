import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from scipy.interpolate import RectBivariateSpline

from . import Beam

class BeamGouyFresnel(Beam):
    def __init__(self, position, normal, wavenumber, beam_width, u, v, focal_length, e_field):
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
        self.apodise = None
        self.e_field_arr = e_field
        F0 = focal_length / (2 * beam_width)
        self.beam_waist = 4 * F0 / self.wavenumber
        self.rayleigh = 8 * F0**2 / self.wavenumber
        self.beam_z = -focal_length
        self.psi = np.arctan(self.beam_z / self.rayleigh)
        self.u_norm = np.sqrt(2) * u / beam_width
        self.v_norm = np.sqrt(2) * v / beam_width
        self.max_step = np.pi / 10

        dx = np.diff(self.u_norm, axis=0).mean()
        dy = np.diff(self.v_norm, axis=1).mean()

        self.r2 = self.u_norm**2 + self.v_norm**2

        kx = 2*np.pi * self.u_norm / (self.u_norm.shape[0]*dx**2)
        ky = 2*np.pi * self.v_norm / (self.v_norm.shape[1]*dy**2)
        self.k2 = kx**2 + ky**2

    def copy_with(self, position, normal):
        beam_width = self.width()
        return BeamGouyFresnel(position, normal, self.wavenumber,
                beam_width, self.u_norm * beam_width / np.sqrt(2), self.v_norm * beam_width / np.sqrt(2), -self.beam_z, self.e_field_arr.copy())

    def width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def max_width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)


    def e_field(self, u, v):
        beam_width = self.width()
        u_norm = np.sqrt(2) * u / beam_width
        v_norm = np.sqrt(2) * v / beam_width

        e_norm = np.zeros((u_norm.shape[0], v_norm.shape[1]), 'complex')
        valid = (self.u_norm.min() < u_norm) & (u_norm < self.u_norm.max())
        valid &= (self.v_norm.min() < v_norm) & (v_norm < self.v_norm.max())

        e_norm[valid] = self.e_r_terp.ev(u_norm[valid], v_norm[valid]) + 1j * self.e_i_terp.ev(u_norm[valid], v_norm[valid])
        # TODO Add other phase factors here, e.g. kz & the parabolic phase
        # Maybe also guoy, but I think that's handled in the FrFT already.
        return e_norm / np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def propagate_z(self, dz):
        super().propagate_z(dz)

        new_psi = np.arctan(self.beam_z / self.rayleigh)
        Dpsi = new_psi - self.psi
        N_steps = np.ceil(Dpsi / self.max_step).astype('int')

        dpsi = Dpsi / N_steps

        # Free space propagation factor
        spectral_phase_halfstep = ifftshift(np.exp(-1j * self.k2/4 * dpsi/2))
        spectral_phase = ifftshift(np.exp(-1j * self.k2/4 * dpsi))

        # pseudo-ior function
        spatial_phase = np.exp(-1j * self.r2 * dpsi)

        Eff = fft2(self.e_field_arr)
        E = ifft2(Eff * spectral_phase_halfstep)

        for _ in range(N_steps-1):
            E *= spatial_phase
            if self.apodise is not None:
                E *= self.apodise

            Eff = fft2(E)
            E = ifft2(Eff * spectral_phase)

        E *= spatial_phase
        Eff = fft2(E)
        self.e_field_arr = ifft2(Eff * spectral_phase_halfstep)
        if self.apodise is not None:
            self.e_field_arr *= self.apodise
        
        self.e_r_terp = RectBivariateSpline(self.u_norm[:,0], self.v_norm[0,:], self.e_field_arr.real)
        self.e_i_terp = RectBivariateSpline(self.u_norm[:,0], self.v_norm[0,:], self.e_field_arr.imag)

