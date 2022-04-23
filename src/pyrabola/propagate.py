
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

    return beam

