import numpy as np
import cv2
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

# set sensor pixel size and film size
pixel_size = 6.45e-3 # [mm]
film_size = [768, 1024]

# set a rendering image sensor, and call prepare_mts to prepare the lensgroup for rendering
lens.prepare_mts(pixel_size, film_size)
# lens.plot_setup2D()

# create a dummy screen
z0 = 10e3 # [mm]
pixelsize = 1.1 # [mm]
texture = cv2.cvtColor(cv2.imread('./images/squirrel.jpg'), cv2.COLOR_BGR2RGB)
texture = np.flip(texture.astype(np.float32), axis=(0,1)).copy()
texturesize = np.array(texture.shape[0:2])
screen = do.Screen(
    do.Transformation(np.eye(3), np.array([0, 0, z0])),
    texturesize * pixelsize, texture, device=device
)
texture_torch = torch.Tensor(texture).to(device=device)

# helper function
def render_single(wavelength, screen):
    valid, ray_new = lens.sample_ray_sensor(wavelength)
    uv, valid_screen = screen.intersect(ray_new)[1:]
    mask = valid & valid_screen
    I = screen.shading(uv, mask)
    return I, mask

# sample wavelengths in [nm]
wavelengths = [656.2725, 587.5618, 486.1327]

# render
ray_counts_per_pixel = 100
Is = []
for wavelength_id, wavelength in enumerate(wavelengths):
    screen.texture = texture_torch[..., wavelength_id]

    # multi-pass rendering by sampling the aperture
    I = 0
    M = 0
    for i in tqdm(range(ray_counts_per_pixel)):
        I_current, mask = render_single(wavelength, screen)
        I = I + I_current
        M = M + mask
    I = I / (M + 1e-10)
    
    # reshape data to a 2D image
    I = I.reshape(*np.flip(np.asarray(film_size))).permute(1,0)
    Is.append(I.cpu())

# show image
plt.imshow(torch.stack(Is, axis=-1).numpy().astype(np.uint8), 'gray')
plt.show()
