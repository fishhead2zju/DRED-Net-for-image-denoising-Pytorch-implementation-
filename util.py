from math import log10
import numpy as np
from PIL import Image
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt
from skimage.measure import compare_ssim
from skimage.measure.simple_metrics import compare_psnr
from scipy.fftpack import idct
from scipy.fftpack import dct

from ssim_v2 import ssim_v2
T = 170
class gmds_Loss(nn.L1Loss):
    def __init__(self):
        super(gmds_Loss, self).__init__()
    def forward(self, input, target):
        return loss_gmds(target,input)

def loss_gmds(img1_batch,img2_batch):
    LOSS = 0
    #print(img1_batch.size())
    wei = img1_batch.size()[2]
    hei = img1_batch.size()[3]
    for i in range(img1_batch.size()[0]):
        LOSS += cal_gmds(img1_batch[i,0,:,:].view(1,1,wei,hei),img2_batch[i,0,:,:].view(1,1,wei,hei))
    return (LOSS/img1_batch.size()[0])

def cal_gmds(img1,img2):
    #img1 = Variable(img1)
    #img2 = Variable(img2)
    aveKernel = torch.Tensor(1,1,3,3)
    aveKernel = Variable(aveKernel,requires_grad=False).cuda()
    nn.init.constant(aveKernel,1/9)
    #Y1 = signal.convolve2d(img1, aveKernel, mode='same', boundary='fill', fillvalue=0)
    Y1 = F.conv2d(img1, aveKernel, bias=None, stride=1, padding=1)
    #Y2 = signal.convolve2d(img2, aveKernel, mode='same', boundary='fill', fillvalue=0)
    Y2 = F.conv2d(img2, aveKernel, bias=None, stride=1, padding=1)

    dx = torch.Tensor(1,1,3,3)
    nn.init.constant(dx,0)
    dx[0][0][0][0] = 1/3
    dx[0][0][1][0] = 1/3
    dx[0][0][2][0] = 1/3
    dx[0][0][0][2] = -1/3
    dx[0][0][1][2] = -1/3
    dx[0][0][2][2] = -1/3
    dy = torch.Tensor(1,1,3,3)
    nn.init.constant(dy,0)
    dy[0][0][0][0] = 1/3
    dy[0][0][0][1] = 1/3
    dy[0][0][0][2] = 1/3
    dy[0][0][2][0] = -1/3
    dy[0][0][2][1] = -1/3
    dy[0][0][2][2] = -1/3

    dx = Variable(dx,requires_grad=False).cuda()
    dy = Variable(dy,requires_grad=False).cuda()
    #IxY1 = signal.convolve2d(Y1,dx,mode = 'same')
    IxY1 = F.conv2d(Y1, dx, bias=None, stride=1, padding=1)
    #IyY1 = signal.convolve2d(Y1,dy,mode = 'same')
    IyY1 = F.conv2d(Y1, dy, bias=None, stride=1, padding=1)
    gradientMap1 = torch.sqrt(IxY1 * IxY1 + IyY1 * IyY1)

    #IxY2 = signal.convolve2d(Y2,dx,mode = 'same')
    IxY2 = F.conv2d(Y2, dx, bias=None, stride=1, padding=1)
    #IxY2 = signal.convolve2d(Y2,dy,mode = 'same')
    IyY2 = F.conv2d(Y2, dy, bias=None, stride=1, padding=1)
    gradientMap2 = torch.sqrt(IxY2 * IxY2 + IyY2 ** 2)

    quality_map = (2*gradientMap1*gradientMap2 + T)/(gradientMap1**2 + gradientMap2**2+T)
    return torch.std(quality_map)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_psnr(img1, img2):
    #mse = ((img1 - img2) ** 2).mean()
    mse = ((img1.astype(np.float) - img2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr
'''
def cal_psnr(imclean,img):
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:],255)
    return (PSNR/Img.shape[0])
'''
def cal_ssim(img1, img2):
    shape = img1.view()
    weight = shape.shape[2]
    height = shape.shape[3]
    #s3 = compare_ssim(img1.astype(np.float).reshape((weight,height)), img2.astype(np.float).reshape((weight,height)))
    #s2 = ssim_v1(img1.astype(np.float).reshape((weight,height)), img2.astype(np.float).reshape((weight,height)))
    s3 = ssim_v2(img1.astype(np.float).reshape((weight,height)), img2.astype(np.float).reshape((weight,height)))
    return s3

def save_image(ground_truth, noisy_image, clean_image, filepath):
    # assert the pixel value range is 0-255
    _, _, im_h, im_w = noisy_image.shape
    ground_truth = ground_truth.reshape((im_h, im_w))
    noisy_image = noisy_image.reshape((im_h, im_w))
    clean_image = clean_image.reshape((im_h, im_w))
    cat_image = np.column_stack((noisy_image, clean_image))
    cat_image = np.column_stack((ground_truth, cat_image))
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')
def save_image_single(ground_truth, noisy_image, clean_image, filepath):
    
    _, _, im_h, im_w = noisy_image.shape
    ground_truth = ground_truth.reshape((im_h, im_w))
    ground_image = Image.fromarray(ground_truth.astype('uint8')).convert('L')
    ground_image.save("%s_ground.png"%filepath, 'png')
    noisy_image = noisy_image.reshape((im_h, im_w))
    noisy_image = Image.fromarray(noisy_image.astype('uint8')).convert('L')
    noisy_image.save("%s_noise.png"%filepath, 'png')
    clean_image = clean_image.reshape((im_h, im_w))
    clean_image = Image.fromarray(clean_image.astype('uint8')).convert('L')
    clean_image.save("%s_clean.png"%filepath, 'png')

class dctfunction():
    def forward(self,input):
        dct_input = input.detach().numpy()
        result = abs(dct(dct_input))
        return input.new(result)
    def backward(self,grad_output):
        dct_go = grad_output.numpy()
        result = idct(dct_go)
        return grad_output.new(result)

