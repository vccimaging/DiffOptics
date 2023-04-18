import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append("../")
import diffoptics as do
from utils_end2end import dict_to_tensor, tensor_to_dict, load_deblurganv2, ImageFolder
torch.manual_seed(0)

# Initialize a lens
device = torch.device('cuda')
lens = do.Lensgroup(device=device)

# Load optics
lens.load_file(Path('./lenses/end2end/end2end_edof.txt')) # norminal design
lens.plot_setup2D()
[surface.to(device) for surface in lens.surfaces]

# set sensor pixel size and film size
downsample_factor = 4     # downsampled for run
pixel_size = downsample_factor * 3.45e-3 # [mm]
film_size = [512 // downsample_factor, 512 // downsample_factor]
lens.prepare_mts(pixel_size, film_size)
print('Check your lens:')
print(lens)

# sample wavelengths in [nm]
wavelengths = [656.2725, 587.5618, 486.1327]

def create_screen(texture: torch.Tensor, z: float, pixelsize: float) -> do.Screen:
    texturesize = np.array(texture.shape[0:2])
    screen = do.Screen(
        do.Transformation(np.eye(3), np.array([0, 0, z])),
        texturesize * pixelsize, texture, device=device
    )
    return screen

def render_single(wavelength: float, screen: do.Screen, sample_ray_function, images: list[torch.Tensor]):
    valid, ray_new = sample_ray_function(wavelength)
    uv, valid_screen = screen.intersect(ray_new)[1:]
    mask = valid & valid_screen

    # Render a batch of images
    I_batch = []    
    for image in images:
        screen.update_texture(image[..., wavelengths.index(wavelength)])
        I_batch.append(screen.shading(uv, mask))
    return torch.stack(I_batch, axis=0), mask

def render(screen: do.Screen, images: list[torch.Tensor], ray_counts_per_pixel: int) -> torch.Tensor:
    Is = []
    for wavelength in wavelengths:
        I = 0
        M = 0
        for i in range(ray_counts_per_pixel):
            I_current, mask = render_single(wavelength, screen, lambda x : lens.sample_ray_sensor(x), images)
            I = I + I_current
            M = M + mask
        I = I / (M[None, ...] + 1e-10)
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0,2,1)
        Is.append(I)
    return torch.stack(Is, axis=-1)

focal_length = 102 # [mm]
def render_gt(screen: do.Screen, images: list[torch.Tensor]) -> torch.Tensor:
    Is = []
    for wavelength in wavelengths:
        I, mask = render_single(wavelength, screen, lambda x : lens.sample_ray_sensor_pinhole(x, focal_length), images)
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0,2,1)
        Is.append(I)
    return torch.stack(Is, axis=-1)


## Set differentiable optical parameters
# XY_surface = (
#     a[0] +
#     a[1] * x + a[2] * y +
#     a[3] * x**2 + a[4] * x*y + a[5] * y**2 +
#     a[6] * x**3 + a[7] * x**2*y + a[8] * x*y**2 + a[9] * y**3 
# )
# We optimize for a cubic profile (o.e. 3rd-order coefficients), as in the wavefront coding technology.
diff_parameters = [
    lens.surfaces[0].ai
]
learning_rates = {
    'surfaces[0].ai': 1e-15 * torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).to(device)
}
for diff_para, key in zip(diff_parameters, learning_rates.keys()):
    if len(diff_para) != len(learning_rates[key]):
        raise Exception('Learning rates of {} is not of equal length to the parameters!'.format(key))
    diff_para.requires_grad = True
diff_parameter_labels = learning_rates.keys()


## Create network
net = load_deblurganv2()
net.prepare()

print('Initial:')
current_parameters = [x.detach().cpu().numpy() for x in diff_parameters]
print('Current optical parameters are:')
for x, label in zip(current_parameters, diff_parameter_labels):
    print('-- lens.{}: {}'.format(label, x))

# Training dataset
train_path = './training_dataset/'
train_dataloader = torch.utils.data.DataLoader(ImageFolder(train_path), batch_size=1, shuffle=False)
it = iter(train_dataloader)
image = next(it).squeeze().to(device)

# Training settings
settings = {
    'spp_forward': 100,             # Rays per pixel for forward
    'spp_backward': 20,             # Rays per pixel for a single-pass backward
    'num_passes': 5,                # Number of accumulation passes for the backward
    'image_batch_size': 5,          # Images per batch
    'network_training_iter': 200,   # Training iterations for network update
    'num_of_training': 10,          # Training outer loop iteration
    'savefig': True                 # Save intermediate results
}

if settings['savefig']:
    opath = Path('end2end_output') / str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    opath.mkdir(parents=True, exist_ok=True)

def wrapper_func(screen, images, squeezed_diff_parameters, diff_parameters, diff_parameter_labels):
    unpacked_diff_parameters = tensor_to_dict(squeezed_diff_parameters, diff_parameters)
    for idx, label in enumerate(diff_parameter_labels):
        exec('lens.{} = unpacked_diff_parameters[{}]'.format(label, idx))
    return render(screen, images, settings['spp_forward'])


# Physical parameters for the screen
zs = [8e3, 6e3, 4.5e3] # [mm]
pixelsizes = [0.1 * z/6e3 for z in zs] # [mm]

print('Training starts ...')
for iteration in range(settings['num_of_training']):
    for z_idx, z in enumerate(zs):
        
        # Print current status
        current_parameters = [x.detach().cpu().numpy() for x in diff_parameters]
        print('=========')
        print('Iteration = {}, z = {} [mm]:'.format(iteration, z))
        print('Current optical parameters are:')
        for x, label in zip(current_parameters, diff_parameter_labels):
            print('-- lens.{}: {}'.format(label, x))
        print('=========')
        
        # Put screen at a desired distance (and with a proper pixel size)
        screen = create_screen(image, z, pixelsizes[z_idx])

        # Forward rendering
        tq = tqdm(range(settings['image_batch_size']))
        tq.set_description('(1) Rendering batch images')
        
        # Load image batch (multiple images)
        images = []
        for image_idx in tq:
            try:
                data = next(it) 
            except StopIteration:
                it = iter(train_dataloader)
                data = next(it) 
            image = data.squeeze().to(device)
            images.append(image.clone())

        with torch.no_grad():
            Is = render(screen, images, settings['spp_forward'])
            Is_gt = render_gt(screen, images)
        tq.close()
        
        # Save images for visualization
        Is_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is], axis=1)
        Is_gt_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is_gt], axis=1)
        
        # Reorder tensors to match neural network input format
        Is = 2 * torch.permute(Is, (0, 3, 1, 2)) / 255 - 1
        Is_gt = 2 * torch.permute(Is_gt, (0, 3, 1, 2)) / 255 - 1

        # Train network weights
        Is_output = net.run(
            Is, Is_gt, is_inference=False,
            num_iters=settings['network_training_iter'], desc='(2) Training network weights'
        )
        Is_output_np = np.transpose(255/2 * (Is_output.detach().cpu().numpy() + 1), (0,2,3,1)).astype(np.uint8)
        Is_output_view = np.concatenate([I for I in Is_output_np], axis=1)
        del Is_output_np

        if settings['savefig']:
            fig, axs = plt.subplots(3, 1)
            for idx, I_view, label in zip(
                range(3), [Is_view, Is_gt_view, Is_output_view], ['Input', 'Ground truth', 'Network output']
            ):
                axs[idx].imshow(I_view)
                axs[idx].set_title(label + ' image(s)')
                axs[idx].set_axis_off()
            fig.tight_layout()
            fig.savefig(
                str(opath / 'iter_{}_z={}mm_images.png'.format(iteration, z)),
                dpi=400, bbox_inches='tight', pad_inches=0.1
            )
            fig.clear()
            plt.close(fig)

        # Back-propagate backend loss and obtain adjoint gradients
        Is.requires_grad = True
        Is_output = net.run(Is, Is_gt, is_inference=False, num_iters=1)

        # Get adjoint gradients of the image batch
        Is_grad = Is.grad.permute(0, 2, 3, 1)
        del Is, Is_gt, Is_output
        torch.cuda.empty_cache()

        # Back-propagate optical parameters with adjoint gradients, and accumulate
        tq = tqdm(range(settings['num_passes']))
        tq.set_description('(3) Back-prop optical parameters')
        dthetas = torch.zeros_like(dict_to_tensor(diff_parameters)).detach()
        for inner_iteration in tq:
            dthetas += torch.autograd.functional.vjp(
                lambda x : wrapper_func(screen, images, x, diff_parameters, diff_parameter_labels),
                dict_to_tensor(diff_parameters), Is_grad
            )[1]
        tq.close()

        # Update optical parameters
        with torch.no_grad():
            for label, diff_para, dtheta in zip(
                diff_parameter_labels, diff_parameters, tensor_to_dict(dthetas, diff_parameters)
            ):
                diff_para -= learning_rates[label] * dtheta.squeeze() / settings['num_passes']
                diff_para.grad = None
