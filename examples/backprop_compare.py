import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import diffoptics as do


device = torch.device('cuda')

# initialize a lens
def init():
    lens = do.Lensgroup(device=device)

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
    lens.film_size = [256, 256]
    lens.pixel_size = 100.0e-3/2
    lens.surfaces[1].ai = torch.zeros(2, device=device)
    return lens

def baseline(network_func, rays):
    lens = init()
    lens.surfaces[0].c.requires_grad = True
    lens.surfaces[1].ai.requires_grad = True
    
    I = 0.0
    for ray in rays:
        I = I + lens.render(ray)
    
    L = network_func(I)
    L.backward()
    print("Baseline:")
    print("primal: {}".format(L))
    print("derivatives: {}".format([lens.surfaces[0].c.grad, lens.surfaces[1].ai.grad]))
    return float(torch.cuda.memory_allocated() / (1024 * 1024))

def ours_new(network_func, rays):
    lens = init()
    adj = do.Adjoint(
        lens, ['surfaces[0].c', 'surfaces[1].ai'],
        network_func, lens.render, rays
    )

    L_item, grads = adj()

    print("Ours:")
    print("primal: {}".format(L_item))
    print("derivatives: {}".format(grads))
    torch.cuda.empty_cache()
    return float(torch.cuda.memory_allocated() / (1024 * 1024))

# Initialize a lens
lens = init()

# generate array of rays
wavelength = torch.Tensor([532.8]).to(device) # [nm]

def prepare_rays(view):
    ray = lens.sample_ray(wavelength.item(), view=view, M=2000+1, sampling='grid')
    return ray

# define a network
torch.manual_seed(0)
I_ref = torch.rand(lens.film_size, device=device)
def network_func(I):
    return ((I - I_ref)**2).mean()

# timings
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# compares
views = [1, 3, 5, 7, 9, 11, 13, 15]
max_views = len(views)

num_rayss = np.zeros(max_views)
time = np.zeros((max_views, 2))
memory = np.zeros((max_views, 2))
for i, num_views in enumerate(views):
    print("view = {}".format(num_views))
    views = np.linspace(0, 1, num_views)
    num_rays = num_views * 2001**2 / 1e6
    num_rayss[i] = num_rays

    # prepare rays
    rays = [prepare_rays(view) for view in views]

    # Baseline
    try:
        start.record()
        memory[i,0] = baseline(network_func, rays)
        end.record()
        torch.cuda.synchronize()
        print("Baseline time: {:.3f} s".format(start.elapsed_time(end)*1e-3))
        time[i,0] = start.elapsed_time(end)
    except:
        print('Baseline: Memory insuffient! Stop running for this case!')
        time[i,0] = np.nan
        memory[i,0] = np.nan
    
    # Ours
    start.record()
    memory[i,1] = ours_new(network_func, rays)
    end.record()
    torch.cuda.synchronize()
    print("Ours (adjoint-based) time: {:.3f} s".format(start.elapsed_time(end)*1e-3))
    time[i,1] = start.elapsed_time(end)


# show results
fig = plt.figure()
plt.plot(num_rayss, time, '-o')
plt.title("Time Comparison")
plt.xlabel("Number of rays [Millions]")
plt.ylabel("Computation time [Seconds]")
plt.legend(["Baseline (backpropagation)", "Ours (adjoint-based)"])
plt.show()
