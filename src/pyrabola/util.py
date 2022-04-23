from contextlib import contextmanager

import numpy as np
from numpy import array as arr
from numpy import sqrt, sin, cos, arctan2, dot, outer

import time

@contextmanager
def timer(message):
    print(message)
    spaces = len(message) - len(message.lstrip(' '))
    t0 = time.monotonic_ns()
    try:
        yield
    finally:
        t1 = time.monotonic_ns()
        print(spaces * ' ' + f'Took {(t1-t0)*1e-6:0.3f}ms')


def normal(theta, phi):
    '''
    Returns a normal vector for a given pitch theta and yaw phi
    '''
    return arr([
        cos(theta) * sin(phi),
        sin(theta),
        cos(theta) * cos(phi)
    ])


def angles(n):
    '''
    Returns the pitch and yaw angle of a given normal vector
    '''
    x, y, z = n
    phi_beam = arctan2(x, z)
    theta_beam = arctan2(y, sqrt(x**2 + z**2))
    return theta_beam, phi_beam


def rot_matrix_angles(theta, phi):
    '''
    This matrix rotates from object coordinates to world coordinates
    Its columns correspond respectively to
        * The object u vector (in-plane, horizontal)
        * The object v vector (in-plane, with a vertical component)
        * The object w (normal) vector
    Multiply an object uvw vector by this matrix
    to transform to world space
    Multiply a world-space xyz vector by its transpose
    to transform to object coordinates.
    '''
    return arr(
        [
            [cos(phi), -sin(theta)*sin(phi), cos(theta)*sin(phi)],
            [0, cos(theta), sin(theta)],
            [-sin(phi), -sin(theta)*cos(phi), cos(theta)*cos(phi)]
        ]
    )


def rot_matrix_normal(object_normal):
    '''
    This matrix rotates from object coordinates to world coordinates
    Its columns correspond respectively to
        * The object u vector (in-plane, horizontal)
        * The object v vector (in-plane, with a vertical component)
        * The object w (normal) vector
    Multiply an object uvw vector by this matrix
    to transform to world space.
    Multiply a world-space xyz vector by its transpose
    to transform to object coordinates.
    '''
    x, y, z = object_normal
    scale = sqrt(1 - y**2)
    u_vec = arr([z, 0, -x]) / scale
    v_vec = arr([-x * y, x**2 + z**2, -y * z]) / scale
    return arr([u_vec, v_vec, object_normal]).T


def projection_matrix(n_1, n_2):
    '''
    This matrix projects a vector along n1 such that
    the result's inner product with n2 is zero.
    Usually n1 will be a beam normal vector
    and n2 the normal of an optical element
    '''
    return np.eye(3) - outer(n_1, n_2) / dot(n_1, n_2)


def reflection_matrix(n_r):
    '''
    This matrix reflects a vector in the plane whose
    normal is n_r
    '''
    return np.eye(3) - 2 * outer(n_r, n_r) / dot(n_r, n_r)


def centred_fft(x, f, **kwargs):
    F = np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(f), **kwargs
        )
    )
    k_x = k_from_x(x)
    return k_x, F


def centred_ifft(x, f, **kwargs):
    F = np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(f), **kwargs
        )
    )
    k_x = k_from_x(x)
    return k_x, F


def centred_fft2(x, y, f, **kwargs):
    F = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(f), **kwargs
        )
    )
    k_x = k_from_x(x)
    k_y = k_from_x(y)
    return k_x, k_y, F


def centred_ifft2(x, y, f, **kwargs):
    F = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(f), **kwargs
        )
    )
    k_x = k_from_x(x)
    k_y = k_from_x(y)
    return k_x, k_y, F


def k_from_x(x):
    return np.fft.fftshift(np.fft.fftfreq( x.size, np.diff(x).mean())) * 2*np.pi
