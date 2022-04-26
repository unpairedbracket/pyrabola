from numpy import pi, sin, cos, arctan2, sqrt, array

def position(entry):
    match entry:
        case [x,y,z]:
            return array([x,y,z])

def angle(entry):
    match entry:
        case {'degrees': value}:
            return pi/180 * value
        case {'radians': value}:
            return value
        case {'cycles': value}:
            return 2*pi * value
        case value:
            return value

def normal(entry):
    named = {'x':(1,0,0), 'y':(0,1,0), 'z':(0,0,1)}
    match entry:
        case {'auto': True}:
            return 'auto'
        case {'axis': name} if name in named:
            return named[name]
        case {'normal': [nx,ny,nz]}:
            total = (nx**2 + ny**2 + nz**2)**0.5
            return array([nx/total, ny/total, nz/total])
        case {'angles': [theta, phi]}:
            return (cos(theta)*sin(phi), sin(theta), cos(theta)*cos(phi))

def euler_angles(entry):
    named = {'x':(0,pi/2), 'y':(pi/2,0), 'z':(0,0)}
    match entry:
        case {'auto': True}:
            return 'auto'
        case {'axis': name} if name in named:
            return named[name]
        case {'normal': [nx,ny,nz]}:
            return (arctan2(ny, sqrt(nx**2 + nz**2)), arctan2(nx, nz))
        case {'angles': [theta, phi]}:
            return (theta, phi)
