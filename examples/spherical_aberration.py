import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append("../")
import diffoptics as do

# initialization
# device = do.init()
device = torch.device('cpu')

# load target lens
lens = do.Lensgroup(device=device)
lens.load_file(Path('./lenses/Thorlabs/ACL5040U.txt'))
print(lens.surfaces[0])

# generate array of rays
wavelength = torch.Tensor([532.8]).to(device) # [nm]
R = 15.0 # [mm]
def render():
    ray_init = lens.sample_ray(wavelength, M=31, R=R)
    ps = lens.trace_to_sensor(ray_init)
    return ps[...,:2]

def trace_all():
    ray_init = lens.sample_ray_2D(R, wavelength, M=15)
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[...,:2], oss
ps, oss = trace_all()
ax, fig = lens.plot_raytraces(oss)

ax, fig = lens.plot_setup2D_with_trace([0.0], wavelength, M=5, R=R)
ax.axis('off')
ax.set_title("")
fig.savefig("layout_trace_asphere.pdf", bbox_inches='tight')

# show initial RMS
ps_org = render()
L_org = torch.mean(torch.sum(torch.square(ps_org), axis=-1))
print('original loss: {:.3e}'.format(L_org))
lens.spot_diagram(ps_org, xlims=[-50.0e-3, 50.0e-3], ylims=[-50.0e-3, 50.0e-3])

diff_names = [
    'surfaces[0].c',
    'surfaces[0].k',
    'surfaces[0].ai'
]

# optimize
out = do.LM(lens, diff_names, 1e-4, option='diag') \
        .optimize(render, lambda y: 0.0 - y, maxit=300, record=True)

# show loss
plt.figure()
plt.semilogy(out['ls'], '-o')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# show spot diagram
ps = render()
L = torch.mean(torch.sum(torch.square(ps), axis=-1))
print('final loss: {:.3e}'.format(L))
lens.spot_diagram(ps, xlims=[-50.0e-3, 50.0e-3], ylims=[-50.0e-3, 50.0e-3])
print(lens.surfaces[0])
# lens.plot_setup2D()
