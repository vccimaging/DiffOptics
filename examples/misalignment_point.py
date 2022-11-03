import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.image import imread, imsave
import cv2

import sys
sys.path.append("../")
import diffoptics as do

"""
Experimental parameters:

light source to sensor: about 675 [mm]
sensor: GS3-U3-50S5M, pixel size 3.45 [um], resolution 2448 × 2048
"""

# initialize a lens
device = do.init()
# device = torch.device('cpu')
lens = do.Lensgroup(device=device)

# ==== Load lens file
lens.load_file(Path('./lenses/Thorlabs/LA1131.txt'))
lens.d_sensor = torch.Tensor([56.0]).to(device) # [mm] sensor distance
lens.plot_setup2D(with_sensor=True)
R = lens.surfaces[0].r

# sensor information
downsample_N = 4
pixel_size = 3.45e-3 * downsample_N # [mm]
N_total = int(2048 / downsample_N)
R_sensor = N_total * pixel_size / 2 # [mm]

# set scene geometry
wavelength = torch.Tensor([622.5]).to(device) # [nm]

# point light source position
light_o = torch.Tensor([0.0, 0.0, -650]).to(device)
lens.light_o = light_o # hook-up

R_in = 1.42*R # must be >= sqrt(2)
M = 1024
def sample_ray(M, light_o):
    o_x, o_y = torch.meshgrid(
        torch.linspace(-R_in, R_in, M, device=device),
        torch.linspace(-R_in, R_in, M, device=device)
    )
    valid = (o_x**2 + o_y**2) < (0.95*R)**2

    o = torch.stack((o_x, o_y, -torch.ones_like(o_x)), axis=-1)
    d = torch.stack((o_x, o_y, torch.zeros_like(o_x)), axis=-1) - light_o[None, None, ...]
    d = do.normalize(d)
    
    o = o[valid]
    d = d[valid]
    
    return do.Ray(o, d, wavelength, device=device)

lens.pixel_size = pixel_size
lens.film_size = [N_total,N_total]
def render():
    ray = sample_ray(M, lens.light_o)
    I = lens.render(ray)
    I = N_total**2 * I / I.sum()
    return I


# centroid
X, Y = torch.meshgrid(
    1 + torch.arange(N_total, device=device),
    1 + torch.arange(N_total, device=device)
)
def centroid(I):
    return torch.stack((
        torch.sum(X * I) / torch.sum(I),
        torch.sum(Y * I) / torch.sum(I)
    ))

### Optimization utilities
def loss(I, I_mea):
    data_term = torch.mean((I - I_mea)**2)
    comp_centroid = True
    if comp_centroid:
        c_mea = centroid(I_mea)
        c = centroid(I)
        loss = data_term + 0.0005 * torch.mean((c - c_mea)**2)
    else:
        loss = data_term
    return loss


# read image
img = imread('./data/20210304/ref2.tif') # for now we use grayscale
img = img.astype(float)
I_mea = cv2.resize(img, dsize=(N_total, N_total), interpolation=cv2.INTER_AREA)
I_mea = np.maximum(0.0, I_mea - np.median(I_mea))
I_mea = N_total**2 * I_mea / I_mea.sum()
I_mea = torch.Tensor(I_mea).to(device)

# AUTO DIFF
diff_variables = ['d_sensor', 'theta_x', 'theta_y', 'light_o']
out = do.LM(lens, diff_variables, 1e-3, option='diag') \
        .optimize(render, lambda y: I_mea - y, maxit=30, record=True)


# crop images
def crop(I):
    c = 200
    return I[c:I.shape[0]-c, c:I.shape[1]-c]

opath = Path('misalignment_point')
opath.mkdir(parents=True, exist_ok=True)
def save(I_mea, Is):
    import imageio
    images = []
    for I in Is:
        images.append(crop(I))
    imageio.mimsave(str(opath / 'movie.mp4'), images)

    # show results
    plt.figure()
    plt.imshow(crop(Is[0]), cmap='gray')
    plt.title('Simulation (initial)')
    
    plt.figure()
    plt.imshow(crop(Is[-1]), cmap='gray')
    plt.title('Simulation (optimized)')
    
    I_mea = I_mea.cpu().detach().numpy()

    plt.figure()
    plt.imshow(crop(I_mea), cmap='gray')
    plt.title('Measurement')
    plt.show()

    plt.imsave(str(opath / 'I0.jpg'), crop(Is[0]), vmin=0, vmax=np.maximum(I_mea.max(), Is[-1].max()), cmap='gray')
    plt.imsave(str(opath / 'I.jpg'), crop(Is[-1]), vmin=0, vmax=np.maximum(I_mea.max(), Is[-1].max()), cmap='gray')
    plt.imsave(str(opath / 'I_mea.jpg'), crop(I_mea), vmin=0, vmax=np.maximum(I_mea.max(), Is[-1].max()), cmap='gray')


save(I_mea, out['Is'])

fig = plt.figure()
plt.plot(out['ls'], 'k-o')
plt.xlabel('iteration')
plt.ylabel('loss')
fig.savefig(str(opath / "ls.pdf"), bbox_inches='tight')
