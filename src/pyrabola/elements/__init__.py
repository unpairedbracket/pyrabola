from .mirror import Mirror
from .parabola import Parabola
from .camera import Camera


def initialise_element(el):
    match el['type']:
        case 'mirror':
            return Mirror.from_dict(el)
        case 'parabola':
            return Parabola.from_dict(el)
        case 'camera':
            return Camera.from_dict(el)
        case _ as type:
            print(f'Unknown element type {type}')
            sys.exit(1)
