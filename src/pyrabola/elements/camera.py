import numpy as np

from .. import parser
from .. import util


class Camera():
    @staticmethod
    def from_dict(config_dict):
        position = parser.position(config_dict['position'])
        angles = parser.euler_angles(config_dict['pointing'])
        px = np.array(config_dict['pixels'])
        size = np.array(config_dict['sensor_size'])
        magnification = config_dict['magnification']
        return Camera(position, angles, px, size, magnification)

    def __init__(self, position, angles, px, size, magnification):
        self.position = np.array(position, 'float')
        self.pitch, self.yaw = np.array(angles, 'float')
        self.sensor = np.zeros(px)
        self.size = size
        self.magnification = magnification

    def propagate(self, beam):
        cam_normal = util.normal(self.pitch, self.yaw)
        Uc, Vc, Nc = util.rot_matrix_normal(cam_normal).T
        Ub, Vb, Nb = util.rot_matrix_normal(beam.normal).T

        if Nc @ Nb > 0:
            print('Camera looking in same direction as beam is looking')
            self.sensor[:] = 0
            return beam

        # Place the origin at the camera centre for now
        dpos = beam.position - self.position
        proj = util.projection_matrix(Nb, Nc)
        sensor_intersect = proj @ dpos

        dz = (sensor_intersect - dpos) @ Nb
        if dz < 0:
            print('Beam moving away from camera')
            self.sensor[:] = 0
            return beam

        with util.timer('    Propagating beam to camera plane'):
            beam.propagate_z(dz)

        max_dist = np.sqrt((self.size**2).sum()) + 2*beam.max_width()

        if sensor_intersect @ sensor_intersect > max_dist**2:
            print('Beam missed camera')
            self.sensor[:] = 0
            return beam

        cam_u = np.linspace(-1, 1, self.sensor.shape[0]) * self.size[0]/2 / self.magnification
        cam_v = np.linspace(-1, 1, self.sensor.shape[1]) * self.size[1]/2 / self.magnification
        cam_u_vals, cam_v_vals = np.meshgrid(cam_u, cam_v, indexing='ij')
        cam_x = cam_u_vals * Uc[0] + cam_v_vals * Vc[0] - sensor_intersect[0]
        cam_y = cam_u_vals * Uc[1] + cam_v_vals * Vc[1] - sensor_intersect[1]
        cam_z = cam_u_vals * Uc[2] + cam_v_vals * Vc[2] - sensor_intersect[2]

        proj_to_beam = util.projection_matrix(Nb, Nb)

        beam_x = (proj_to_beam[0, 0] * cam_x +
                  proj_to_beam[0, 1] * cam_y +
                  proj_to_beam[0, 2] * cam_z)

        beam_y = (proj_to_beam[1, 0] * cam_x +
                  proj_to_beam[1, 1] * cam_y +
                  proj_to_beam[1, 2] * cam_z)

        beam_z = (proj_to_beam[2, 0] * cam_x +
                  proj_to_beam[2, 1] * cam_y +
                  proj_to_beam[2, 2] * cam_z)

        beam_u = Ub[0] * beam_x + Ub[1] * beam_y + Ub[2] * beam_z
        beam_v = Vb[0] * beam_x + Vb[1] * beam_y + Vb[2] * beam_z

        self.sensor = abs(beam.e_field(beam_u, beam_v))**2

    def read(self):
        return self.sensor

    def plot(self):
        pass
