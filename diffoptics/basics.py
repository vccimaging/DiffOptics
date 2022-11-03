import math
from enum import Enum
import torch
import numpy as np


class PrettyPrinter():
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    lines += '{}[{}]: {}'.format(key, i, v).split('\n')

            elif val.__class__.__name__ in 'dict':
                pass
            elif key == key.upper() and len(key) > 5:
                pass
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

    def to(self, device=torch.device('cpu')):
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif issubclass(type(val), PrettyPrinter):
                exec(f'self.{key}.to(device)')
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec('self.{x}[{i}] = self.{x}[{i}].to(device)'.format(x=key, i=i))
                    elif issubclass(type(v), PrettyPrinter):
                        exec('self.{}[{}].to(device)'.format(key, i))


class Ray(PrettyPrinter):
    """
    Definition of a geometric ray.

    - o is the ray origin
    - d is the ray direction (normalized)
    """
    def __init__(self, o, d, wavelength, device=torch.device('cpu')):
        self.o = o
        self.d = d

        # scalar-version
        self.wavelength = wavelength # [nm]
        self.mint = 1e-5 # [mm]
        self.maxt = 1e5  # [mm]
        self.to(device)

    def __call__(self, t):
        return self.o + t[..., None] * self.d


class Transformation(PrettyPrinter):
    """
    Rigid Transformation.
    
    - R is the rotation matrix.
    - t is the translational vector.
    """
    def __init__(self, R, t):
        if torch.is_tensor(R):
            self.R = R
        else:
            self.R = torch.Tensor(R)
        if torch.is_tensor(t):
            self.t = t
        else:
            self.t = torch.Tensor(t)

    def transform_point(self, o):
        return torch.squeeze(self.R @ o[..., None]) + self.t

    def transform_vector(self, d):
        return torch.squeeze(self.R @ d[..., None])
    
    def transform_ray(self, ray):
        o = self.transform_point(ray.o)
        d = self.transform_vector(ray.d)
        if o.is_cuda:
            return Ray(o, d, ray.wavelength, device=torch.device('cuda'))
        else:
            return Ray(o, d, ray.wavelength)

    def inverse(self):
        RT = self.R.T
        t = self.t
        return Transformation(RT, -RT @ t)


class Spectrum(PrettyPrinter):
    """
    Spectrum distribution of the rays.
    """
    def __init__(self):
        self.WAVELENGTH_MIN = 400 # [nm]
        self.WAVELENGTH_MAX = 760 # [nm]
        self.to()

    def sample_wavelength(self, sample):
        return self.WAVELENGTH_MIN + (self.WAVELENGTH_MAX - self.WAVELENGTH_MIN) * sample


class Sampler(PrettyPrinter):
    """
    Sampler for generating random sample points.
    """
    def __init__(self):
        self.to()
        self.pi_over_2 = np.pi / 2
        self.pi_over_4 = np.pi / 4
    
    def concentric_sample_disk(self, x, y):
        # https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
        
        # map uniform random numbers to [-1,1]^2
        x = 2 * x - 1
        y = 2 * y - 1
        
        # handle degeneracy at the origin when xy == [0,0]

        # apply concentric mapping to point
        eps = np.finfo(float).eps
        
        if type(x) is torch.Tensor and type(y) is torch.Tensor:
            cond = torch.abs(x) > torch.abs(y)
            r = torch.where(cond, x, y)
            theta = torch.where(cond,
                self.pi_over_4 * (y / (x + eps)),
                self.pi_over_2 - self.pi_over_4 * (x / (y + eps))
            )
            return r * torch.cos(theta), r * torch.sin(theta)
        
        if type(x) is np.ndarray and type(y) is np.ndarray:
            cond = np.abs(x) > np.abs(y)
            r = np.where(cond, x, y)
            theta = np.where(cond,
                self.pi_over_4 * (y / (x + eps)),
                self.pi_over_2 - self.pi_over_4 * (x / (y + eps))
            )
            return r * np.cos(theta), r * np.sin(theta)


class Filter(PrettyPrinter):
    def __init__(self, radius):
        self.radius = radius
    def eval(self, p):
        raise NotImplementedError()
    
class Box(Filter):
    def __init__(self, radius=None):
        if radius is None:
            radius = [0.5, 0.5]
        Filter.__init__(self, radius)
    def eval(self, x):
        return torch.ones_like(x)

class Triangle(Filter):
    def __init__(self, radius):
        if radius is None:
            radius = [2.0, 2.0]
        Filter.__init__(self, radius)
    def eval(self, p):
        x, y = p[...,0], p[...,1]
        return (torch.maximum(torch.zeros_like(x), self.radius[0] - x) *
                torch.maximum(torch.zeros_like(y), self.radius[1] - y))

# ----------------------------------------------------------------------------------------

class Material(PrettyPrinter):
    """
    Optical materials for computing the refractive indices.
    
    The following follows the simple formula that

    n(\lambda) = A + B / \lambda^2

    where the two constants A and B can be computed from nD (index at 589.3 nm) and V (abbe number). 
    """
    def __init__(self, name=None):
        self.name = 'vacuum' if name is None else name.lower()

        # This table is hard-coded. TODO: Import glass libraries from Zemax.
        self.MATERIAL_TABLE = { # [nD, Abbe number]
            "vacuum":     [1.,       math.inf],
            "air":        [1.000293, math.inf],
            "occluder":   [1.,       math.inf],
            "f2":         [1.620,    36.37],
            "f15":        [1.60570,  37.831],
            "uvfs":       [1.458,    67.82],
            
            # https://shop.schott.com/advanced_optics/
            "bk10":       [1.49780,  66.954],
            "n-baf10":    [1.67003,  47.11],
            "n-bk7":      [1.51680,  64.17],
            "n-sf1":      [1.71736,  29.62],
            "n-sf2":      [1.64769,  33.82],
            "n-sf4":      [1.75513,  27.38],
            "n-sf5":      [1.67271,  32.25],
            "n-sf6":      [1.80518,  25.36],
            "n-sf6ht":    [1.80518,  25.36],
            "n-sf8":      [1.68894,  31.31],
            "n-sf10":     [1.72828,  28.53],
            "n-sf11":     [1.78472,  25.68],
            "sf1":        [1.71736,  29.51],
            "sf2":        [1.64769,  33.85],
            "sf4":        [1.75520,  27.58],
            "sf5":        [1.67270,  32.21],
            "sf6":        [1.80518,  25.43],
            "sf18":       [1.72150,  29.245],

            # HIKARI.AGF
            "baf10":      [1.67,     47.05],

            # SUMITA.AGF
            "sk1":        [1.61030,  56.712],
            "sk16":       [1.62040,  60.306],
            "ssk4":       [1.61770,  55.116],

            # https://www.pgo-online.com/intl/B270.html
            "b270":       [1.52290,  58.50],
            
            # https://refractiveindex.info, nD at 589.3 nm
            "s-nph1":     [1.8078,   22.76], 
            "d-k59":      [1.5175,   63.50],
            
            "flint":      [1.6200,   36.37],
            "pmma":       [1.491756, 58.00],
            "polycarb":   [1.585470, 30.00]
        }
        self.A, self.B = self._lookup_material()

    def ior(self, wavelength):
        """Computes index of refraction of a given wavelength (in [nm])"""
        return self.A + self.B / wavelength**2

    @staticmethod
    def nV_to_AB(n, V):
        def ivs(a): return 1./a**2
        lambdas = [656.3, 589.3, 486.1]
        B = 0.0 if V == 0 else (n - 1) / V / ( ivs(lambdas[2]) - ivs(lambdas[0]) )
        A = n - B * ivs(lambdas[1])
        return A, B
    
    def _lookup_material(self):
        out = self.MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            n, V = out
        elif out is None:
            # try parsing input as a n/V pair
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])
        return self.nV_to_AB(n, V)
    
    def to_string(self):
        return f'{self.A} + {self.B}/lambda^2'


class InterpolationMode(Enum):
    nearest = 1
    linear  = 2

class BoundaryMode(Enum):
    zero      = 1
    replicate = 2
    symmetric = 3
    periodic  = 4

class SimulationMode(Enum):
    render = 1
    trace  = 2


"""
Utility functions.
"""

def init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DiffMetrology is using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device

def length2(d):
    return torch.sum(d**2, axis=-1)

def length(d):
    return torch.sqrt(length2(d))

def normalize(d):
    return d / length(d)[..., None]

def set_zeros(x, valid=None):
    if valid == None:
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)
    else:
        mask = valid[...,None] if len(x.shape) > len(valid.shape) else valid
        return torch.where(~mask, torch.zeros_like(x), x)

def rodrigues_rotation_matrix(k, theta): # theta: [rad]
    """
    This function implements the Rodrigues rotation matrix.
    """
    # cross-product matrix
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.Tensor([
        [  0, -kz,  ky],
        [ kz,   0, -kx],
        [-ky,  kx,   0]
    ]).to(k.device)
    if not torch.is_tensor(theta):
        theta = torch.Tensor(np.asarray(theta)).to(k.device)
    return torch.eye(3, device=k.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

def set_axes_equal(ax, scale=np.ones(3)):
    """
    Make axes of 3D plot have equal scale (or scaled by `scale`).
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    tmp = np.abs(limits[:,1]-limits[:,0])
    ax.set_box_aspect(scale * tmp/np.min(tmp))


"""
Test functions
"""

def generate_test_rays():
    filmsize = np.array([4, 2])

    o = np.array([3,4,-200])
    o = np.tile(o[None, None, ...], [*filmsize, 1])
    o = torch.Tensor(o)
    
    dx = 0.1 * torch.rand(*filmsize)
    dy = 0.1 * torch.rand(*filmsize)
    d = normalize(torch.stack((dx, dy, torch.ones_like(dx)), axis=-1))

    wavelength = 500 # [nm]
    return Ray(o, d, wavelength)

def generate_test_transformation():
    k = np.random.rand(3)
    k = k / np.sqrt(np.sum(k**2))
    k = torch.Tensor(k)
    theta = 1 # [rad]
    R = rodrigues_rotation_matrix(k, theta)
    t = np.random.rand(3)
    return Transformation(R, t)

def generate_test_material():
    return Material('N-BK7')


if __name__ == "__main__":
    init()

    rays = generate_test_rays()
    to_world = generate_test_transformation()
    print(to_world)
    
    rays_new = to_world.transform_ray(rays)
    o_old = rays.o[2,1,...].numpy()
    o_new = rays_new.o[2,1,...].numpy()
    assert (to_world.transform_point(o_old) - o_new).abs().sum() < 1e-15

    material = generate_test_material()
    print(material)
