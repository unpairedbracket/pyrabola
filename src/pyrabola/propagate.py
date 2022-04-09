from numpy import array as arr
from numpy import pi, abs, cos, sin, flip, linspace

import matplotlib.pyplot as plt

from .beams.geometric import BeamGaussianGeometric
from .beam_injection import BeamInjection
from .elements.mirror import Mirror
from .elements.parabola import Parabola
from .elements.camera import Camera

#plt.ion()

def propagate(beam_injector, optical_system):
    ''' Unconverted plot code
    figure(1);
    cla;
    hold on;
    scatter3(mirror_locations(:,3), mirror_locations(:,1), mirror_locations(:,2), 50, 'ob');
    plot3(mirror_locations(:,3), mirror_locations(:,1), mirror_locations(:,2),'--', 'Color', [0.8,0.8,0.8]);
    '''

    for element in optical_system:
        element.plot()

    beam = beam_injector.get_beam()

    for element in optical_system:
        beam = element.propagate(beam)

def main():
    beam_injector = BeamInjection(
        position=arr([0,0,-1]),
        direction=arr([0,0,1]),
        wavelength=1e-6,
        beam_type=BeamGaussianGeometric,
        beam_waist=30e-3,
        distance_to_focus=0
    )

    f = 0.03
    Phi = pi/180 * 15
    f0 = f * cos(Phi)**2
    elements = [
        Mirror(arr([0, 0, 0]), [0, pi-pi/4], 5*25.4e-3),
        Mirror(arr([1, 0, 0]), [0,   -pi/4], 5*25.4e-3),
        Mirror(arr([1, 0, 1]), [0, pi+pi/4], 5*25.4e-3),
        Mirror(arr([0, 0, 1]), [0,   +pi/4], 5*25.4e-3),
        Parabola(arr([0, 0, 2]), [0.0, pi+0.0], radius=0.2, parent_focal_length=f0, parabola_angle=2*Phi),
        Camera(arr([f*sin(2*Phi), 0, 2-f*cos(2*Phi)]), [0, -2*Phi], (640, 480), arr([64, 48])*1e-6)
    ]
    parabola = elements[-2]
    camera = elements[-1]
    fig, axs = plt.subplots(1, 1)
    axs = arr([[axs]])
    for i, dtheta in enumerate([0]):
        parabola.pitch = 5e-6 * dtheta
        for j, dphi in enumerate([0]):
            parabola.yaw = pi + 5e-6 * dphi
            propagate(beam_injector, elements)
            axs[i,j].imshow(flip(camera.read().T, axis=0), vmin=0, extent=(-32,32,-24,24))
            axs[i,j].axis('off')
            print(i,j)
            #plt.imsave(f'images/phi_{dphi}_theta_{dtheta}.png', flip(camera.read().T, axis=0), vmin=0, vmax=200)
    plt.show()


if __name__ == '__main__':
    main()
