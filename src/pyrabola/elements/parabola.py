import numpy as np
from numpy import sqrt, sin, cos, tan, arctan2
from numpy import array as arr

from ..beams import Beam
from ..beams.geometric import BeamFresnel
from ..beams.hermite import BeamHermite
from .mirror import Mirror
from .. import util

from matplotlib import pyplot as plt

class Parabola(Mirror):
    f0 = None
    # Position of centre of OAP in parabola-centred coordinates
    # The Mirror.position property of parabolas refers to
    # the position of this point on their surface and
    # rotation angles are defined about this point
    X0 = np.array([0, 0])

    def __init__(self, position, correct_angles, radius,
                 parent_focal_length, parabola_angle=None,
                 parabola_roll=0, X0=None):
        super().__init__(position, correct_angles, radius)
        self.f0 = parent_focal_length
        if parabola_angle is not None and X0 is not None:
            print('Parabola angle and X0 both specified. Using X0.')
        if X0 is not None:
            x0, y0 = X0[0]/(2*self.f0), X0[1]/(2*self.f0)
            self.X0 = np.array([x0, y0, (x0**2 + y0**2 - 1)/2])
        elif parabola_angle is not None:
            r0 = np.tan(parabola_angle/2)
            x0, y0 = r0 * np.cos(parabola_roll), r0 * np.sin(parabola_roll)
            self.X0 = np.array([x0, y0, (r0**2 - 1) / 2])

    def normal(self, hit):
        return np.array([-hit[0], -hit[1], 1]) / sqrt(1 + hit[0]**2 + hit[1]**2)

    def propagate(self, beam):
        transformed_beam = self.transform_beam(beam)
        hit_normal = self.normal(transformed_beam.position)
        reflect = util.reflection_matrix(hit_normal)
        new_beam_normal = reflect @ transformed_beam.normal

        x = np.linspace(-2, 2, 4001) * beam.width()
        u = x / (2*self.f0)

        z_residual, focal_factor, clip_R = self.optical_path_difference(
            u, u, transformed_beam, False
        )

        nb_dot_nh = np.dot(hit_normal, -transformed_beam.normal)

        f = self.f0 / (focal_factor * nb_dot_nh)

        phase = beam.wavenumber * z_residual * nb_dot_nh
        mirror_e_field = beam.e_field(x, x) * np.exp(1j * phase)
        mirror_e_field[np.abs(clip_R.imag) > 0] = 0
        if beam.clipping:
            mirror_e_field *= (clip_R <= self.radius)

        #plt.imshow(abs(np.fft.fftshift(np.fft.fft2(mirror_e_field)))**2)
        #plt.show()

        return self.parabola_to_world(BeamHermite(
            position=transformed_beam.position,
            normal=new_beam_normal, wavenumber=beam.wavenumber,
            beam_width=beam.width(), u=x, v=x, focal_length=f,
            e_field=mirror_e_field
        ))

    def world_to_parabola(self, beam):
        R_parabola = util.rot_matrix_angles(self.pitch, self.yaw).T
        parabola_position = self.X0 + R_parabola @ (
                beam.position - self.position
            ) / (2*self.f0)
        parabola_normal = R_parabola @ beam.normal
        return beam.copy_with(position=parabola_position, normal=parabola_normal)

    def parabola_to_world(self, beam):
        R_parabola = util.rot_matrix_angles(self.pitch, self.yaw)
        world_position = self.position + (
            (2*self.f0) * R_parabola @ (
                    beam.position - self.X0
            ))
        world_normal = R_parabola @ beam.normal
        return beam.copy_with(position=world_position, normal=world_normal)

    def transform_beam(self, beam):
        R_parabola = util.rot_matrix_angles(self.pitch, self.yaw).T
        transformed_position = self.X0 + R_parabola @ (
                beam.position - self.position
            ) / (2*self.f0)
        transformed_normal = R_parabola @ beam.normal
        x0, y0, z0 = transformed_position
        nx, ny, nz = transformed_normal

        if nz > 0:
            print('beam travelling in wrong direction')

        A = nx**2 + ny**2
        Bon2 = (nx * x0 + ny * y0) - nz
        C = x0**2 + y0**2 - 1 - 2*z0  # = 2 ( Z(x0, y0) - z0 )

        #A x^2 + 2 B x + C = 0
        #x = - C/(2 B)
        #- C^2/8B^3 = dx/dA

        if C > 0:
            print('beam behind parabola')

        param = A*C/Bon2**2
        delta = -C/(Bon2) / (sqrt(1 - param) + 1)

        hit_position = transformed_position + delta * transformed_normal

        dx, dy, dz = hit_position - self.X0
        if dx**2 + dy**2 > self.radius**2:
            print('missed mirror')

        return Beam(hit_position, transformed_normal, beam.wavenumber)

    def optical_path_difference(self, u_beam, v_beam, beam_ray, full_evaluation=True):
        '''
        Calculate optical path difference for specified beam with transverse coordinates u,v
        beam_ray: ray in parabola coordinates. position should already be in parabola coordinates,
            in contact with the parabola and in 'parabola reduced' coordinates, i.e. divided by 2f0
        u_beam, v_beam: coordinates to evaluate phases at in beam transverse directions.
        '''

        xh, yh, zh = beam_ray.position

        U, V, N = util.rot_matrix_normal(beam_ray.normal).T
        nx, ny, nz = N

        u, v = np.meshgrid(u_beam, v_beam, indexing='ij')

        x0 = xh + U[0] * u + V[0] * v
        y0 = yh + U[1] * u + V[1] * v
        z0 = zh + U[2] * u + V[2] * v

        cosPhi = (1 + xh**2 + yh**2)**(-1/2)

        if nx**2 + ny**2 < 1e-20:
            hit_x = x0
            hit_y = y0

            OPD_aberration = 0
            focal_factor = cosPhi

        else:
            # delta (2nd order) = 1/2 ( cu2 u^2 + 2 cuv u v + cv2 v^2 )
            # = (cu2+cv2)/2 * (u^2+v^2)/2 + (cu2-cv2)/2 * (u^2-v^2)/2 + cuv u v
            # = { ----focusing part---- } + { --------astigmatism part------- }

            M = np.array([[1, 0, xh], [0, 1, yh], [xh, yh, xh**2 + yh**2]])
            A = np.array([-xh, -yh, 1])

            cu2 = (V @ M @ V) / (N @ A)**2
            cuv = (U @ M @ V) / (N @ A)**2
            cv2 = (U @ M @ U) / (N @ A)**2

            focal_factor = (cu2 + cv2)/2 * cosPhi
            focus = focal_factor * (u**2+v**2)/2

            if full_evaluation:
                A = nx**2 + ny**2
                Bon2 = (nx * x0 + ny * y0) - nz
                C = x0**2 + y0**2 - 1 - 2*z0

                param = A*C/Bon2**2
                delta = -C/(Bon2) / (sqrt(1 - param) + 1)

                hit_x = x0 + delta * nx
                hit_y = y0 + delta * ny

                Delta = ((hit_x - xh)**2 + (hit_y - yh)**2) / 2 * cosPhi

                OPD_aberration = Delta - focus
            else:
                dc = (cu2 - cv2)/2
                astigma_factor = sqrt(dc**2 + cuv**2)
                astigma_tilt = arctan2(-cuv, dc) / 2

                u_as = u * cos(astigma_tilt) + v * sin(astigma_tilt)
                v_as = v * cos(astigma_tilt) - u * sin(astigma_tilt)
                astigma = astigma_factor * ( u_as**2 - v_as**2 ) / 2

                delta0 = u * xh - v * yh - (u**2 + v**2) / 2
                
                hit_x = x0 + delta0 * nx
                hit_y = y0 + delta0 * ny
                
                coma = (u * nx - v * ny) * (u**2 + v**2) / 2
                OPD_aberration = ( astigma + coma ) * cosPhi

        clip_R = sqrt((hit_x - self.X0[0])**2 + (hit_y-self.X0[1])**2)

        return OPD_aberration, focal_factor, clip_R

