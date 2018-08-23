import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
def BasicConv2d(inch, outch, k_size, pad = 0):
    return nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size = k_size, stride = 1, padding
            = pad ,bias = False)

class init_conv(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,pad = 1):
        super(init_conv,self).__init__()
        #self.conv = BasicConv2d(in_ch,out_ch,k_size,pad)
        self.conv = CDilated(in_ch,out_ch,k_size,d=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self,x):
        conv = self.conv(x)
        out = self.act(self.bn(conv))
        return out
class end_deconv(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,pad = 1):
        super(end_deconv,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k_size,padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
    def forward(self,x):
        deconv = self.deconv(x)
        out = self.act(self.bn(deconv))
        return out
class conv_bn_relu(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,pad = 1):
        super(conv_bn_relu,self).__init__()
        #self.conv = BasicConv2d(in_ch,out_ch,k_size,pad)
        n = int(out_ch/2)
        self.conv_reduce = BasicConv2d(in_ch,n,1)
        self.bn_reduce = nn.BatchNorm2d(n)
        self.act1 = nn.ReLU()
        self.conv_conv = CDilated(n,n,k_size,d=2)
        self.bn_conv = nn.BatchNorm2d(n)
        self.act2 = nn.ReLU()
        self.conv_expend = BasicConv2d(n,out_ch,1)
        self.bn_expend = nn.BatchNorm2d(out_ch)
        self.act3 = nn.ReLU()

    def forward(self,x):
        out1 = self.act1(self.bn_reduce(self.conv_reduce(x)))
        out2 = self.act2(self.bn_conv(self.conv_conv(out1)))
        out3 = self.act3(self.bn_expend(self.conv_expend(out2))+x)
        return out3

class deconv_bn_relu(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,pad = 1):
        super(deconv_bn_relu,self).__init__()
        n = int(out_ch/2)
        self.deconv_reduce = nn.ConvTranspose2d(in_ch, n, 1, bias=False)
        self.deconv_conv = nn.ConvTranspose2d(n, n, k_size,padding=2, bias=False,dilation=2)
        self.deconv_expend = nn.ConvTranspose2d(n, out_ch, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(n)
        self.bn_conv= nn.BatchNorm2d(n)
        self.bn_expend = nn.BatchNorm2d(out_ch)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
    def forward(self,x):
        out1 = self.act1(self.bn_reduce(self.deconv_reduce(x)))
        out2 = self.act2(self.bn_conv(self.deconv_conv(out1)))
        out3 = self.act3(self.bn_expend(self.deconv_expend(out2))+x)
        return out3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encode1 = init_conv(1,64)
        self.encode2 = conv_bn_relu(64,64)
        self.encode3 = conv_bn_relu(64,64)
        self.encode4 = conv_bn_relu(64,64)
        self.encode5 = conv_bn_relu(64,64)
        self.encode6 = conv_bn_relu(64,64)
        self.encode7 = conv_bn_relu(64,64)
        
        self.encode8 = conv_bn_relu(64,64)
        self.encode9 = conv_bn_relu(64,64)
        self.encode10 = conv_bn_relu(64,64)
        self.encode11 = conv_bn_relu(64,64)
        '''
        self.encode12 = conv_bn_relu(64,64)
        self.encode13 = conv_bn_relu(64,64)
        self.encode14 = conv_bn_relu(64,64)
        self.encode15 = conv_bn_relu(64,64)
        self.encode16 = conv_bn_relu(64,64)
        self.encode17 = conv_bn_relu(64,64)
        self.encode18 = conv_bn_relu(64,64)
        '''
        self.decode1 = deconv_bn_relu(64,64)
        self.decode2 = deconv_bn_relu(64,64)
        self.decode3 = deconv_bn_relu(64,64)
        self.decode4 = deconv_bn_relu(64,64)
        self.decode5 = deconv_bn_relu(64,64)
        self.decode6 = deconv_bn_relu(64,64)
        self.decode7 = deconv_bn_relu(64,64)
        self.decode8 = deconv_bn_relu(64,64)
        self.decode9 = deconv_bn_relu(64,64)
        self.decode10 = deconv_bn_relu(64,64)
        '''
        self.decode11 = deconv_bn_relu(64,64)
        self.decode12 = deconv_bn_relu(64,64)
        self.decode13 = deconv_bn_relu(64,64)
        self.decode14 = deconv_bn_relu(64,64)
        self.decode15 = deconv_bn_relu(64,64)
        self.decode16 = deconv_bn_relu(64,64)
        self.decode17 = deconv_bn_relu(64,64)
        '''
        self.decode11 = end_deconv(64,1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal(m.weight.data,a=0,mode = 'fan_in')
            elif classname.find('Linear')!=-1:
                nn.init.kaiming_normal(m.weight.data,a=0,mode = 'fan_in')
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(mean=0,std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant(m.bias.data,0.0)

    def forward(self, x):
        
        out1 = self.encode1(x)  
        out2 = self.encode2(out1)
        out3 = self.encode3(out2)
        e4 = self.encode4(out3)  
        e5 = self.encode5(e4)
        e6 = self.encode6(e5)
        e7 = self.encode7(e6)
        e8 = self.encode8(e7)  
        e9 = self.encode9(e8)
        e10 = self.encode10(e9)
        e11 = self.encode11(e10)
        '''
        e12 = self.encode12(e11)
        e13 = self.encode13(e12)
        e14 = self.encode14(e13)
        e15 = self.encode15(e14)
        e16 = self.encode16(e15)
        e17 = self.encode17(e16)
        e18 = self.encode18(e17)
        '''
        d1 = self.decode1(e11)
        d2 = self.decode2(d1)
        d3 = self.decode3(d2)
        d4 = self.decode4(d3)
        d5  = self.decode5(d4)
        d6 = self.decode6(d5)
        d7 = self.decode7(d6)
        d8 = self.decode8(d7)
        d9 = self.decode9(d8)
        d10 = self.decode10(d9)
        d11 = self.decode11(d10)
        '''
        d12 = self.decode12(d11)
        d13  = self.decode13(d12)
        d14 = self.decode14(d13)
        d15 = self.decode15(d14)
        d16 = self.decode16(d15)
        d17 = self.decode17(d16)
        d18 = self.decode18(d17)
        '''
        return d11+x


















