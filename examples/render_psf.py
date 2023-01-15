import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

# initialize a lens
device = torch.device('cuda')
lens = do.Lensgroup(device=device)

# load optics
lens.load_file(Path('./lenses/DoubleGauss/US02532751-1.txt'))

# sensor area
pixel_size = 6.45e-3 # [mm]
film_size = torch.tensor([1200, 1600], device=device)
R_square = film_size * pixel_size

# generate array of rays
wavelength = 532.8 # [nm]
R = 5.0 # [mm]
# lens.plot_setup2D()

def render_psf(I, p):
    # compute shifts and do linear interpolation
    uv = (p + R_square/2) / pixel_size
    index_l = torch.vstack((
        torch.clamp(torch.floor(uv[:,0]).long(), min=0, max=film_size[0]),
        torch.clamp(torch.floor(uv[:,1]).long(), min=0, max=film_size[1]))
    ).T
    index_r = torch.vstack((
        torch.clamp(index_l[:,0] + 1, min=0, max=film_size[0]),
        torch.clamp(index_l[:,1] + 1, min=0, max=film_size[1]))
    ).T
    w_r = torch.clamp(uv - index_l, min=0, max=1)
    w_l = 1.0 - w_r
    del uv

    # compute image
    I = torch.index_put(I, (index_l[...,0],index_l[...,1]), w_l[...,0]*w_l[...,1], accumulate=True)
    I = torch.index_put(I, (index_r[...,0],index_l[...,1]), w_r[...,0]*w_l[...,1], accumulate=True)
    I = torch.index_put(I, (index_l[...,0],index_r[...,1]), w_l[...,0]*w_r[...,1], accumulate=True)
    I = torch.index_put(I, (index_r[...,0],index_r[...,1]), w_r[...,0]*w_r[...,1], accumulate=True)
    return I

def generate_surface_samples(M):
    Dx = np.random.rand(M,M)
    Dy = np.random.rand(M,M)
    [px, py] = do.Sampler().concentric_sample_disk(Dx, Dy)
    return np.stack((px.flatten(), py.flatten()), axis=1)

def sample_ray(o_obj, M):
    p_aperture_2d = R * generate_surface_samples(M)
    N = p_aperture_2d.shape[0]
    p_aperture = np.hstack((p_aperture_2d, np.zeros((N,1)))).reshape((N,3))
    o = np.ones(N)[:, None] * o_obj[None, :]
    
    o = o.astype(np.float32)
    p_aperture = p_aperture.astype(np.float32)
    
    d = do.normalize(torch.from_numpy(p_aperture - o))
    
    o = torch.from_numpy(o).to(lens.device)
    d = d.to(lens.device)
    
    return do.Ray(o, d, wavelength, device=lens.device)

def render(o_obj, M, rep_count):
    I = torch.zeros(*film_size, device=device)
    for i in range(rep_count):
        rays = sample_ray(o_obj, M)
        ps = lens.trace_to_sensor(rays, ignore_invalid=True)
        I = render_psf(I, ps[..., :2])
    return I / rep_count

# PSF rendering parameters
x_max_halfangle = 10 # [deg]
y_max_halfangle = 7.5 # [deg]
Nx = 2 * 8 + 1
Ny = 2 * 6 + 1

# sampling parameters
M = 1001
rep_count = 1

def render_at_depth(z):
    x_halfmax = np.abs(z) * np.tan(np.deg2rad(x_max_halfangle))
    y_halfmax = np.abs(z) * np.tan(np.deg2rad(y_max_halfangle))

    I_psf_all = torch.zeros(*film_size, device=device)
    for x in tqdm(np.linspace(-x_halfmax, x_halfmax, Nx)):
        for y in np.linspace(-y_halfmax, y_halfmax, Ny):
            o_obj = np.array([y, x, z])
            I_psf = render(o_obj, M, rep_count)
            I_psf_all = I_psf_all + I_psf
    return I_psf_all

# render PSF at different depths
zs = [-1e4, -7e3, -5e3, -3e3, -2e3, -1.5e3, -1e3]
savedir = Path('./rendered_psfs')
savedir.mkdir(exist_ok=True, parents=True)
I_psfs = []
for z in zs:
    I_psf = render_at_depth(z)
    I_psf = I_psf.cpu().numpy()
    plt.imsave(str(savedir / 'I_psf_z={}.png'.format(z)), np.uint8(255 * I_psf / I_psf.max()), cmap='hot')
    I_psfs.append(I_psf)
