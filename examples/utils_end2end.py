import os
import sys
import numpy as np
import torch
import cv2
import pathlib


def dict_to_tensor(xs):
    """
    Concatenates (or, packs) a dictionary of differentiable parameters into a Tensor array. 
    """
    return torch.cat([x.view(-1) for x in xs], 0)

def tensor_to_dict(xs, diff_parameters):
    """
    Unpacks a Tensor array into a dictionary of differentiable parameters.
    """
    ys = []
    idx = 0
    for diff_para in diff_parameters:
        y = xs[idx:idx+diff_para.numel()]
        idx += diff_para.numel()
        ys.append(y)
    return ys

def load_image(image_name):
    img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    img = np.flip(img.astype(np.float32), axis=(0,1)).copy()
    return img


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path):
        super(ImageFolder, self).__init__()
        self.file_names = sorted(os.listdir(root_path))
        self.root_path = pathlib.Path(root_path)

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        return load_image(str(self.root_path / self.file_names[index]))
    

def load_deblurganv2():
    """
    Loads the DeblurGANv2 as the neural network backend.
    """
    neural_network_path = pathlib.Path('./neural_networks/DeblurGANv2')
    sys.path.append(str(neural_network_path))

    import train_end2end

    trainer = train_end2end.load_from_config(str(neural_network_path / 'config/config.yaml'))

    return trainer
