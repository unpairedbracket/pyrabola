import numpy as np

from . import Beam

class BeamGaussianGeometric(Beam):
    beam_waist = 0
    rayleigh = 0

    def __init__(self, position, normal, wavenumber, beam_waist, distance_to_focus):
        super().__init__(position, normal, wavenumber)
        self.beam_waist = beam_waist
        self.rayleigh = wavenumber * beam_waist**2 / 2
        self.beam_z = -distance_to_focus

    def copy_with(self, position, normal):
        return BeamGaussianGeometric(position, normal, self.wavenumber, self.beam_waist, -self.beam_z)

    def width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def e_field(self, x, y):
        beam_width = self.width()
        return np.exp(-(x**2+y**2)/beam_width**2) * np.sqrt(np.pi)/beam_width

class BeamFresnel(Beam):
    e_field_complex = None

    def __init__(self, position, normal, wavenumber, u, v, e_field):
        super().__init__(position, normal, wavenumber)
        self.u = u
        self.v = v
        self.e_field_complex = e_field

    def copy_with(self, position, normal):
        return BeamFresnel(position, normal, self.wavenumber,
                self.u, self.v, self.e_field_complex)

    def width(self):
        # approximation of beam width
        intensity = abs(self.e_field_complex)**2
        u_pos = (self.u[:,None] * intensity).mean() / intensity.mean()
        v_pos = (self.v[None,:] * intensity).mean() / intensity.mean()
        u_var = (((self.u[:,None] - u_pos)**2 * intensity).mean() / intensity.mean())
        v_var = (((self.v[None,:] - v_pos)**2 * intensity).mean() / intensity.mean())
        avg_var = (u_var + v_var)
        return np.sqrt(avg_var)

    def max_width(self):
        return np.sqrt(np.range(self.u)**2 + np.range(self.v)**2) / 2

    def e_field(self, x, y):
        return self.e_field_complex

    def propagate_z(self, z):
        super().propagate_z(z)
        # Use Angular Spectrum Method
        kx, ky, fourier = util.centred_fft2(self.u, self.v, self.e_field_complex)
        kz = np.sqrt(self.wavenumber**2 - (kx**2 + ky**2) + 0j)
        fourier *= np.exp(1j * kz * z)
        self.e_field_complex = util.centred_ifft2(self.u, self.v, fourier)

