import torch.utils.data as data
import torch
import json
import os
from os.path import join 
from PIL import Image
import glob

def _is_image_file(filename):
    return any(filename.endwith(extension) for extension in [".jpg",".png", ".bmp"])

def _load_image_label(imagepath):
    #image = Image.open(imagepath).convert('RGB')
    image = Image.open(imagepath).convert('L')
    return image

class CustomDataset(data.Dataset):
    def __init__(self, image_dir, input_transform=None, loader=None):
        super(CustomDataset, self).__init__()     
        self.image_filenames = glob.glob('{}/*.png'.format(image_dir))
        self.image_filepath =  self.image_filenames
        self.input_transform = input_transform
        if loader is None:
            self.loader = _load_image_label
        else:
            self.loader = loader

    def __getitem__(self, index):
     #   image = self.loader(self.image_filepath[index])
        image = Image.open(self.image_filepath[index])

        if self.input_transform:
            noise_image, groundtruth = self.input_transform(image)
        
        return noise_image, groundtruth

    def __len__(self):
        return len(self.image_filepath)
