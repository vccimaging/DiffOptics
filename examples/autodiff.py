import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import diffoptics as do

# initialize a lens
device = torch.device('cpu')
lens = do.Lensgroup(device=device)

save_dir = './autodiff_demo/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

R = 12.7
surfaces = [
    do.Aspheric(R, 0.0, c=0.05, device=device),
    do.Aspheric(R, 6.5, c=0., device=device)
]
materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]
lens.load(surfaces, materials)
lens.d_sensor = 25.0
lens.r_last = 12.7

# generate array of rays
wavelength = torch.Tensor([532.8]).to(device) # [nm]
R = 10.0 # [mm]
def render():
    ray_init = lens.sample_ray(wavelength, M=9, R=R, sampling='grid')
    ps = lens.trace_to_sensor(ray_init)
    return ps[...,:2]

def trace_all():
    ray_init = lens.sample_ray_2D(R, wavelength, M=11)
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[...,:2], oss

def compute_Jacobian(ps):
    Js = []
    for i in range(1):
        J = torch.zeros(torch.numel(ps))
        for j in range(torch.numel(ps)):
            mask = torch.zeros(torch.numel(ps))
            mask[j] = 1
            ps.backward(mask.reshape(ps.shape), retain_graph=True)
            J[j] = lens.surfaces[i].c.grad.item()
            lens.surfaces[i].c.grad.data.zero_()
        J = J.reshape(ps.shape)

    # get data to numpy
    Js.append(J.cpu().detach().numpy())
    return Js


N = 20
cs = np.linspace(0.045, 0.063, N)
Iss = []
Jss = []
for index, c in enumerate(cs):
    index_string = str(index).zfill(3)
    # load optics
    lens.surfaces[0].c = torch.Tensor(np.array(c))
    lens.surfaces[0].c.requires_grad = True
    
    # show trace figure
    ps, oss = trace_all()
    ax, fig = lens.plot_raytraces(oss, color='b-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(save_dir + "layout_trace_" + index_string + ".png", bbox_inches='tight')

    # show spot diagram
    RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))
    ps = render()
    rms_org = RMS(ps)
    print(f'RMS: {rms_org}')
    lens.spot_diagram(ps, xlims=[-4, 4], ylims=[-4, 4], savepath=save_dir + "spotdiagram_" + index_string + ".png", show=False)

    # compute Jacobian
    Js = compute_Jacobian(ps)[0]
    print(Js.max())
    print(Js.min())
    ps_ = ps.cpu().detach().numpy()
    fig = plt.figure()
    x, y = ps_[:,0], ps_[:,1]
    plt.plot(x, y, 'b.', zorder=0)
    plt.quiver(x, y, Js[:,0], Js[:,1], color='b', zorder=1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    fig.savefig(save_dir + "flow_" + index_string + ".png", bbox_inches='tight')

    # compute images
    ray = lens.sample_ray(wavelength.item(), view=0.0, M=2049, sampling='grid')
    lens.film_size = [512, 512]
    lens.pixel_size = 50.0e-3/2
    I = lens.render(ray)
    I = I.cpu().detach().numpy()
    lm = do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')
    JI = lm.jacobian(lambda: lens.render(ray)).squeeze()
    J = JI.abs().cpu().detach().numpy()

    Iss.append(I)
    Jss.append(J)
    plt.close()

Iss = np.array(Iss)
Jss = np.array(Jss)
for i in range(N):
    plt.imsave(save_dir + "I_" + str(i).zfill(3) + ".png", Iss[i], cmap='gray')
    plt.imsave(save_dir + "J_" + str(i).zfill(3) + ".png", Jss[i], cmap='gray')

names = [
    'spotdiagram',
    'layout_trace',
    'I',
    'J',
    'flow'
]
