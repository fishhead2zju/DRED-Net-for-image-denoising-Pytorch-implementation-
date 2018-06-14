import argparse, os
import time
import torch
import random
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util import AverageMeter
from util import cal_psnr
from util import save_image,save_image_single
#from loss import SSIM, PAMSE

from model import Model
#from  model_dncnn import Model 
#from densenet import DenseNet121
#from hazy_model import Model
#from dncnn1 import Model
#from mlp import Model

from custom_transform import SingleRandomCropTransform
from custom_transform import SingleTransform
#from tensorboard_logger import log_value, configure
from dataset import CustomDataset
# Training and Testing settings
parser = argparse.ArgumentParser(description="PyTorch denoise Experiment")
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help="Models are saved here")
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='Test image result dir')
parser.add_argument('--train_dir', dest='train_dir',default='../denoise_1/data/train_tmp/train_patch', help="Train data dir")
#parser.add_argument('--train_dir', dest='train_dir', default='./data/train', help="Train data dir")
parser.add_argument('--test_dir', dest='test_dir', default='../denoise_1/data/test/Set12', help='Test data dir')
parser.add_argument('--patch_size', type=int, default=40, help='Training patch size')
parser.add_argument('--noise_level', type=int, default=25, help='Noise level') 
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


best_accuracy = 0

def main():
    global opt, best_accuracy
    opt = parser.parse_args()
    print(opt)
    
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    if not os.path.exists(opt.train_dir):
        print("{} not exist".format(opt.train_dir))
        return
    if not os.path.exists(opt.test_dir):
        print("{} not exist".format(opt.test_dir))
        return
    #configure(os.path.join(opt.ckpt_dir, 'log'), flush_secs=5)

    #cuda = False
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    #opt.seed = random.randint(1, 10000)
    opt.seed = 0
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    train_transform = SingleRandomCropTransform(opt.patch_size,
                                                opt.noise_level)
    train_set = CustomDataset(opt.train_dir, train_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_transform = SingleTransform(opt.noise_level)
    test_set = CustomDataset(opt.test_dir, test_transform)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size = 1, shuffle=False)
    print("===> Building model")
    model = Model()
    #model = DenseNet121();
    print(model)
    criterion = nn.MSELoss(size_average=False)
    #criterion = PAMSE()
    #criterion = nn.L1Loss(size_average=False)
    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        #model.cuda()
        criterion = criterion.cuda()
    else:
        print("Not Using GPU")

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  
    
    if opt.evaluate:
        if opt.resume:
            validate(test_data_loader, model, criterion, opt.start_epoch)
            return
        else:
            print("!!!!!!!!!!!!!!!!!! please choose a resume model !!!!!!!!!!")
            return

    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    #for param in model.module.part1.parameters():
    #    param.require_grad = False
    #ignored_params1 = list(map(id, model.branch3x3.parameters()))
    #ignored_params2 = list(map(id, model.branch3x1.parameters()))
    #ignored_params  = ignored_params1 + ignored_params2
    #base_params = filter(lambda p: id(p) not in ignored_params,
    #                     model.parameters())
    #optimizer = torch.optim.Adam([
    #                 {'params': base_params},
    #                 {'params': model.branch3x3.parameters(), 'lr':opt.lr, 'weight_decay': 1e-8},
    #                 {'params': model.branch3x1.parameters(), 'lr':opt.lr, 'weight_decay': 1e-8}
    #                     ], lr=opt.lr*0.1, weight_decay = 0)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):        
        train(training_data_loader, optimizer, model, criterion, epoch)
        
        psnr = validate(test_data_loader, model, criterion, epoch)

        is_best = psnr > best_accuracy
        best_accuracy = max(psnr, best_accuracy)
        save_checkpoint({'epoch': epoch,
                         'best_accuracy':best_accuracy,
                         'model': model.state_dict()}, is_best, epoch)
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    #lr = opt.lr * (0.1 ** (epoch // 40))
    lr = opt.lr
    if epoch == 40:
        lr = opt.lr*0.1
    elif epoch == 60:
        lr = opt.lr*0.01
    elif epoch == 80:
        lr = opt.lr*0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("===>Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

def train(training_data_loader, optimizer, model, criterion, epoch):
     
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()

    #adjust_learning_rate(optimizer, epoch-1)

    model.train()    

    end = time.time()
    for i, batch in enumerate(training_data_loader, 1):
        data_time.update(time.time() - end)
        noise_image, groundtruth = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            noise_image = noise_image.cuda()
            groundtruth = groundtruth.cuda()
        clean_image = model(noise_image)
        loss = criterion(clean_image, groundtruth)/(noise_image.size()[0]*2)

        losses.update(loss.data[0], clean_image.size(0))

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        ground_truth = torch.clamp(255 * groundtruth, 0, 255).byte()
        output_clean_image = torch.clamp(255 * clean_image, 0, 255).byte()
        psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_clean_image.data.cpu().numpy())
        avg_psnr.update(psnr, noise_image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(training_data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, psnr=avg_psnr))

    #log_value('train_loss', losses.avg, epoch)
    #log_value('train_avg_psnr', avg_psnr.avg, epoch)

def validate(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()

    model.eval()

    end = time.time()

    for i,(image, target) in enumerate(test_loader):
        with torch.no_grad():
            image_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)
         
        if opt.cuda:
            image_var = image_var.cuda()
            target_var = target_var.cuda()
   
        clean_image = model(image_var)
        loss = criterion(clean_image, target_var)
        #loss = loss / predict_noise.size(0)
        losses.update(loss.data[0], image_var.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ground_truth = torch.clamp(255 * target_var, 0, 255).byte()
        output_image = torch.clamp(255 * clean_image, 0, 255).byte()
        noise_image = torch.clamp(255 * image_var, 0, 255).byte()
        save_image(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
        #save_image_single(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
        psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())

        #output_image = torch.clamp(255 * clean_image_hr, 0, 255).byte()
        #psnr1 = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
        #print("psnr hr %.3f!!!!-----------psnr lr %.3f!!!!!!!!!-"%(psnr,psnr1))
        #save_image(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(),
        #        output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test_lr%d.png'%i))

        avg_psnr.update(psnr, image_var.size(0))

        if i % 1 == 0:
            print('Test Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(test_loader), batch_time=batch_time,
                   loss=losses, psnr=avg_psnr))

    #log_value('test_loss', losses.avg, epoch)
    #log_value('test_avg_psnr', avg_psnr.avg, epoch)

    print("--- Epoch %d  --------- Average PSNR %.2f ---" %(epoch, avg_psnr.avg))

    value_out_path = os.path.join(opt.ckpt_dir, "num.txt")
    F = open(value_out_path,'a')
    F.write("Epoch %d: PSNR %.2f\n"%(epoch,avg_psnr.avg))
    F.close()

    return avg_psnr.avg

def save_checkpoint(state, is_best, epoch):
    model_out_path = os.path.join(opt.ckpt_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(state, model_out_path)
    #print("Checkpoint saved to {}".format(model_out_path))
    if is_best:
        best_model_name = os.path.join(opt.ckpt_dir, "model_best.pth")
        shutil.copyfile(model_out_path, best_model_name)
        print('Best model {} saved to {}'.format(model_out_path, best_model_name))

if __name__ == "__main__":
    main()
