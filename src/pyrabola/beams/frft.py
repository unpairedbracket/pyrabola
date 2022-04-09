from . import Beam
class BeamFocusing(Beam):
    e_field_at_infinity = None
    e_r_interpolant = None
    e_i_interpolant = None
    beam_waist = 0.
    rayleigh = 0.
    u_norm = None
    v_norm = None

    def __init__(self, position, normal, wavenumber, beam_width, u, v, focal_length, e_field, do_prop=True):
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
        self.e_field_at_infinity = e_field
        F0 = focal_length / (2 * beam_width)
        # F_actual = F0 * ( (1/2 (k0 W / 4F0)^2) - sqrt( (1/2 (k0 W / 4F0)^2)^2 - (k0 W / 4F0)^2 ) )
        self.beam_waist = 4 * F0 / self.wavenumber
        self.rayleigh = 8 * F0**2 / self.wavenumber
        self.beam_z = -focal_length
        self.u_norm = np.sqrt(2) * u / beam_width
        self.v_norm = np.sqrt(2) * v / beam_width
        if do_prop:
            self.propagate_z(0)

    def copy_with(self, position, normal):
        B = BeamFocusing(position, normal, self.wavenumber,
                1, 0, 0, -self.beam_z, self.e_field_at_infinity.copy(), False)
        B.beam_waist = self.beam_waist
        B.rayleigh = self.rayleigh
        B.u_norm = self.u_norm.copy()
        B.v_norm = self.v_norm.copy()
        B.e_r_interpolant = self.e_r_interpolant
        B.e_i_interpolant = self.e_i_interpolant
        return B

    def width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def max_width(self):
        return self.beam_waist * np.sqrt(1 + (self.beam_z/self.rayleigh)**2)


    def e_field(self, u, v):
        beam_width = self.width()
        u_norm = np.sqrt(2) * u / beam_width
        v_norm = np.sqrt(2) * v / beam_width

        if u.ndim==1:
            e_norm = (
                self.e_r_interpolant(u_norm, v_norm)
                + 1j * self.e_i_interpolant(u_norm, v_norm)
            ).T
        else:
            u_full = np.linspace(u_norm.min(), u_norm.max(), u.shape[0]*2)
            v_full = np.linspace(v_norm.min(), v_norm.max(), v.shape[1]*2)
            e_full = (
                self.e_r_interpolant(u_full, v_full)
                + 1j * self.e_i_interpolant(u_full, v_full)
            ).T
            actual_terp = RegularGridInterpolator((u_full, v_full), e_full, bounds_error=False, fill_value=0)
            e_norm = actual_terp((u_norm, v_norm))

        # TODO Add other phase factors here, e.g. kz & the parabolic phase
        # Maybe also guoy, but I think that's handled in the FrFT already.
        return e_norm / np.sqrt(1 + (self.beam_z/self.rayleigh)**2)

    def propagate_z(self, z):
        super().propagate_z(z)

        tan_psi = self.beam_z / self.rayleigh

        ku = util.k_from_x(self.u_norm)
        kv = util.k_from_x(self.v_norm)
        u_prime = ku / np.sqrt(1 + tan_psi**2)
        v_prime = kv / np.sqrt(1 + tan_psi**2)

        e_field = FrFT_tangent(
            self.v_norm, v_prime, FrFT_tangent(
                self.u_norm, u_prime, self.e_field_at_infinity.T, tan_psi
            ).T, tan_psi
        )
        self.e_r_interpolant = interp2d(u_prime, v_prime, e_field.real.T, bounds_error=False, fill_value=0)
        self.e_i_interpolant = interp2d(u_prime, v_prime, e_field.imag.T, bounds_error=False, fill_value=0)

