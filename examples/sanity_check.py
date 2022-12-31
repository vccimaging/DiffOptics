import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append("../")
import diffoptics as do

# initialize a lens
device = torch.device('cuda')
lens = do.Lensgroup(device=device)

# load optics
lens.load_file(Path('./lenses/DoubleGauss/US02532751-1.txt'))

# sample wavelengths in [nm]
wavelengths = torch.Tensor([656.2725, 587.5618, 486.1327]).to(device)

colors_list = 'bgry'
views = np.linspace(0, 21, 4, endpoint=True)
ax, fig = lens.plot_setup2D_with_trace(views, wavelengths[1], M=4)
ax.axis('off')
ax.set_title('Sanity Check Setup 2D')
fig.savefig('sanity_check_setup.pdf')

# spot diagrams
spot_rmss = []
valid_maps = []
for i, view in enumerate(views):
    ray = lens.sample_ray(wavelengths[1], view=view, M=31, sampling='grid', entrance_pupil=True)
    ps = lens.trace_to_sensor(ray, ignore_invalid=True)
    lim = 20e-3
    lens.spot_diagram(
        ps[...,:2], show=True, xlims=[-lim, lim], ylims=[-lim, lim], color=colors_list[i]+'.',
        savepath='sanity_check_field_view_{}.png'.format(int(view))
    )

    spot_rmss.append(lens.rms(ps))

plt.show()
