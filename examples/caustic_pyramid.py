import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from skimage.transform import resize

import sys
sys.path.append("../")
import diffoptics as do

# initialize a lens
device = do.init()
# device = torch.device('cpu')
lens = do.Lensgroup(device=device)

# construct freeform optics
R = 25.4
ns = [256, 256]
surfaces = [
    do.Aspheric(R, 0.0, c=0., is_square=True, device=device),
    do.Mesh(R, 1.0, ns, is_square=True, device=device)
]
materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]
lens.load(surfaces, materials)

# set scene geometry
D = torch.Tensor([50.0]).to(device) # [mm]
wavelength = torch.Tensor([532.8]).to(device) # [nm]

# example image
filename = 'einstein'
img_org = imread('./images/' + filename + '.jpg') # assume image is grayscale
if img_org.mean() > 1.0:
    img_org = img_org / 255.0

# downsample the image
NN = 2
img_org = img_org[::NN,::NN]
N_max = 128
img_org = img_org[:N_max,:N_max]

# mark differentiable variables
lens.surfaces[1].c.requires_grad = True

# create save dir
savepath = './einstein_pyramid/'
if not os.path.exists(savepath):
    os.mkdir(savepath)

def caustic(N, pyramid_i, lr=1e-3, maxit=100):
    img = resize(img_org, (N, N))
    E = np.sum(img) # total energy
    print(f'image size = {img.shape}')

    N_pad = 0
    N_total = N + 2*N_pad
    img = np.pad(img, (N_pad,N_pad), 'constant', constant_values=np.inf)
    img[np.isinf(img)] = 0.0 # revert img back for visualization
    I_ref = torch.Tensor(img).to(device) # [mask]

    # max square length
    R_square = R * N_total/N

    # set image plane pixel grid
    R_image = R_square
    pixel_size = 2*R_image / N_total # [mm]

    def sample_ray(M=1, random=False):
        M = int(M*N)
        x, y = torch.meshgrid(
            torch.linspace(-R_square, R_square, M, device=device),
            torch.linspace(-R_square, R_square, M, device=device)
        )
        p = 2*R_square / M
        if random:
            x = x + p * (torch.rand(M,M,device=device)-0.5)
            y = y + p * (torch.rand(M,M,device=device)-0.5)
        o = torch.stack((x,y,torch.zeros_like(x, device=device)), axis=2)
        d = torch.zeros_like(o)
        d[...,2] = torch.ones_like(x)
        return do.Ray(o, d, wavelength, device=device), E

    def render_single(I, ray_init, irr):
        ray, valid = lens.trace(ray_init)[:2]
        J = irr * valid * ray.d[...,2]
        p = ray(D)
        p = p[...,:2]
        del ray, valid
        
        # compute shifts and do linear interpolation
        uv = (p + R_square) / pixel_size
        index_l = torch.clamp(torch.floor(uv).long(), min=0, max=N_total-1)
        index_r = torch.clamp(index_l + 1, min=0, max=N_total-1)
        w_r = torch.clamp(uv - index_l, min=0, max=1)
        w_l = 1.0 - w_r
        del uv

        # compute image
        I = torch.index_put(I, (index_l[...,0],index_l[...,1]), w_l[...,0]*w_l[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_l[...,1]), w_r[...,0]*w_l[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_l[...,0],index_r[...,1]), w_l[...,0]*w_r[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_r[...,1]), w_r[...,0]*w_r[...,1]*J, accumulate=True)
        return I

    def render(spp=1):
        I = torch.zeros((N_total,N_total), device=device)
        ray_init, irr = sample_ray(M=24, random=True) # Reduce M if your GPU memory is low
        I = render_single(I, ray_init, irr)
        return I / spp

    # optimize
    ls = []

    save_path = savepath + "/{}".format("pyramid_" + str(pyramid_i))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('optimizing ...')
    optimizer = torch.optim.Adam([lens.surfaces[1].c], lr=lr, betas=(0.99,0.99), amsgrad=True)

    for it in range(maxit+1):
        I = render(spp=8)
        I = I / I.sum() * I_ref.sum()
        L = torch.mean((I - I_ref)**2)
        optimizer.zero_grad()
        L.backward(retain_graph=True)

        # record
        ls.append(L.cpu().detach().numpy())
        if it % 10 == 0:
            print('iter = {}: loss = {:.4e}, grad_bar = {:.4e}'.format(
                it, L.item(), torch.sum(torch.abs(lens.surfaces[1].c.grad))
            ))
            I_current = I.cpu().detach().numpy()
            imsave("{}/{:04d}.png".format(save_path, it), I_current, vmin=0.0, vmax=1.0, cmap='gray')

        # descent
        optimizer.step()

    if pyramid_i == 0: # last one, render final image
        lens.surfaces[1].c.requires_grad = False
        del L
        I_final = 0
        spp = 100
        for i in range(spp):
            if i % 10 == 0:
                print("=== rendering spp = {}".format(i))
            I_final += render().cpu().detach().numpy()
        return I_final / spp, I_ref, ls
    else:
        return I.cpu().detach().numpy(), None, ls

pyramid_levels = 2
for i in range(pyramid_levels, -1, -1):
    N = int(N_max/(2**i))
    print("=== N = {}".format(N))
    I_final, I_ref, ls = caustic(N, i, lr=1e-3, maxit=int(1000/4**i))
    if i == 0:
        I_ref = I_ref.cpu().numpy()
        I_final = I_final / I_final.sum() * I_ref.sum()

imsave(savepath + "/I_target.png", I_ref, vmin=0.0, vmax=1.0, cmap='gray')
imsave(savepath + "/I_final.png", I_final, vmin=0.0, vmax=1.0, cmap='gray')

# final results
plt.imshow(I_final, cmap='gray')
plt.title('Final caustic image')
plt.show()

fig, ax = plt.subplots()
ax.plot(ls, 'k-o', linewidth=2)
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
fig.savefig("ls.pdf", bbox_inches='tight')
plt.title('Loss')

S = lens.surfaces[1].mesh().cpu().detach().numpy()
S = S - S.min()
imsave(savepath + "/phase.png", S, vmin=0, vmax=S.max(), cmap='coolwarm')
imsave(savepath + "/phase_mod.png", np.mod(S*1e3,100), cmap='coolwarm')
print(S.max())

plt.figure()
plt.imshow(S, cmap='jet')
plt.colorbar()
plt.title('Optimized phase plate height [mm]')
plt.show()

