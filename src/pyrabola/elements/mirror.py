import numpy as np
from numpy import array as arr
from numpy import sqrt, sin, cos, dot, outer

from ..beams import BeamMissed
from .. import parser
from .. import util


class Mirror:
    @staticmethod
    def from_dict(config_dict):
        position = parser.position(config_dict['position'])
        correct_angles = parser.euler_angles(config_dict['pointing'])
        try:
            radius = config_dict['radius']
        except KeyError:
            radius = float('inf')

        return Mirror(position, correct_angles, radius)

    def __init__(self, position, correct_angles, radius):
        self.position = position
        self.pitch, self.yaw = correct_angles
        self.radius = radius

    def adjust_yaw(self, d_yaw):
        self.yaw += d_yaw

    def adjust_pitch(self, d_pitch):
        self.pitch += d_pitch

    @property
    def normal(self):
        return util.normal(self.pitch, self.yaw)

    def rotation_matrix(self):
        return util.rot_matrix_angles(self.pitch, self.yaw)

    def plot(self):
        '''
        Not yet implemented
        '''
        pass

    def propagate(self, beam):
        if beam.normal.dot(beam.position - self.position) > 0:
            print('travelling away from mirror')
            return BeamMissed()

        mirror_normal = self.normal
        if dot(mirror_normal, beam.normal) > 0:
            print('mirror facing away from beam')
            return BeamMissed()

        proj_beam = util.projection_matrix(beam.normal, mirror_normal)
        reflect_matrix = util.reflection_matrix(mirror_normal)

        hit_location = proj_beam @ (beam.position - self.position)
        new_location = hit_location + self.position
        new_normal = reflect_matrix @ beam.normal

        proj_new = np.eye(3) - outer(new_normal, mirror_normal) / dot(new_normal, mirror_normal)

        if dot(hit_location, hit_location) > self.radius**2:
            print('missed mirror')
            return BeamMissed()

        if beam.clipping:
            # This is as-yet unimplemented
            raise NotImplementedError()
            rot_mirror = self.rotation_matrix()
            rot_beam = util.rot_matrix_normal(beam.normal)
            rot_new = util.rot_matrix_normal(new_normal)
            truncated_identity = np.eye(3)[:, :2]
            m_beam = (rot_mirror @ truncated_identity).T @ proj_beam @ (rot_beam @ truncated_identity)
            m_new = (rot_mirror @ truncated_identity).T @ proj_new @ (rot_new @ truncated_identity)
            twist = np.linalg.inv(m_new) @ m_beam
            twist_angle = arctan2(twist[0,1], twist[0,0])
            beam.twist += twist_angle
            beam.clip()
        ''' Unconverted plotting code
        figure(1)
        plot3([beam_location(3), new_location(3)],...
              [beam_location(1), new_location(1)],...
              [beam_location(2), new_location(2)], 'r')
        axis equal
        xlim([0,1]+[-1,1]*0.1)
        ylim([0,1]+[-1,1]*0.1)
        hit_uvw = rot_mirror' * hit_location
        figure(2)
        subplot( 1, N_mirrors, i)
        cla
        hold on
        scatter( hit_uvw(1), hit_uvw(2) )
        plotMirror3d([0,0,0], mirror_radius, pi/2, 0)
        view(2)
        axis equal
        xlim([-1,1]*mirror_radius)
        ylim([-1,1]*mirror_radius)
        '''
        beam.position = new_location
        beam.normal = new_normal
        return beam

