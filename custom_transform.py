import torch
import collections
import random 
import math
from PIL import Image
import numpy as np
import numbers
import torchvision.transforms as transforms
#from torchvision.transforms.functional import resized_crop,\
#resize, hflip, to_tensor, crop
'''
for train
    size: rescale image size 
    mean: sequence of means for R,G,B channels
    std: sequence of standard deviations for R,G,B channels
'''
to_tensor = transforms.ToTensor()
class AddGaussianNoise(object):
    def __init__(self, mean, sigma):
        self.sigma = sigma
        self.mean = mean
    def __call__(self, image):
        ch, row, col = image.size()
        gauss = torch.Tensor(ch, row, col)
        sigma = self.sigma  / 255.0
        gauss.normal_(self.mean, sigma)
        noise_image = gauss + image
        return noise_image
        #return image

class SingleRandomCropTransform(object):
    def __init__(self, size, noise_level, interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.noise_level = noise_level
    def __call__(self, image):
        #transforms.CenterCrop(32)
        crop = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        tensor_image = crop(image);
        #tmp_noise_level = random.randint(1,55)
        noise_image = AddGaussianNoise(0, self.noise_level)(tensor_image)
        return noise_image, tensor_image
'''
for single crop test
'''
class SingleTransform(object):
    def __init__(self, noise_level, interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.noise_level = noise_level

    def __call__(self, image):
        tensor_image = to_tensor(image)
        noise_image = AddGaussianNoise(0, self.noise_level)(tensor_image)
        #noise_image = tensor_image
        return noise_image, tensor_image
