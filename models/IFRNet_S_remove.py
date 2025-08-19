import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp, get_robust_weight
from loss import *
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from torchinfo import summary
from torch.utils.data import DataLoader


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation=dilation, groups=in_channels, bias=bias),
        # Pointwise convolution
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
        nn.PReLU(out_channels)
    )


def depthwise_separable_conv(channels, bias = True):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, groups = channels, bias = bias ),
        nn.Conv2d(channels,channels, kernel_size = 1, stride = 1, padding = 0, bias = bias),
        nn.PReLU(channels)
    )



class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True, bottleneck_ratio=0.5):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels

        # Full feature map convs
        self.conv1 = depthwise_separable_conv(in_channels, bias)
        self.conv2 = depthwise_separable_conv(side_channels, bias)
        self.conv3 = depthwise_separable_conv(in_channels, bias)

        # Final bottlenecked depthwise separable conv
        bottleneck_channels = max(1, int(in_channels * bottleneck_ratio))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, 
                      padding=0, bias=bias),  # reduce channels
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, 
                      padding=1, groups=bottleneck_channels, bias=bias),  # depthwise
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, stride=1, 
                      padding=0, bias=bias)   # restore channels
        )

        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        # Side-channel refinement once (conv2)
        out[:, -self.side_channels:, :, :] = self.conv2(
            out[:, -self.side_channels:, :, :].clone()
        )

        out = self.conv3(out)

        # conv4 dropped → keeps original channels for last side slice

        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 24, 3, 2, 1), 
            convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(24, 36, 3, 2, 1), 
            convrelu(36, 36, 3, 1, 1)
        )

        self.pyramid4 = nn.Sequential(
            convrelu(36, 54, 3, 2, 1), 
            convrelu(54, 54, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)

        f4 = self.pyramid4(f2)
        return f1, f2, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(108+1, 54),  
            ResBlock(54, 12),
            nn.ConvTranspose2d(54, 36 + 4, 4, 2, 1)
        )
        
    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out



class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu((36 * 3) + 4, 36 * 3), 
            ResBlock(36 * 3, 12), 
            nn.ConvTranspose2d(36 * 3, 24 + 4, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu((24*3)+4, 24*3),  
            ResBlock((24*3), 12), 
            nn.ConvTranspose2d((24*3), 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def forward(self, img0, img1, embt, imgt, flow=None):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_

        f0_1, f0_2,  f0_4 = self.encoder(img0)
        f1_1, f1_2,  f1_4 = self.encoder(img1)
        ft_1, ft_2,  ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_2_ = out4[:, 4:]


        out2 = self.decoder2(ft_2_ , f0_2, f1_2, up_flow0_4 , up_flow1_4)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2))
        # loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(4.0 * resize(up_flow0_4, 4.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(4.0 * resize(up_flow1_4, 4.0) - flow[:, 2:4], weight=robust_weight1))
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, loss_rec, loss_geo, loss_dis



    def inference(self, img0, img1, embt, scale_factor=1.0):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)

        f0_1, f0_2, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_4 = self.encoder(img1_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out2 = self.decoder2(ft_3_, f0_2, f1_2, up_flow0_4, up_flow1_4)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
        up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0/scale_factor))
        up_res_1 = resize(up_res_1, scale_factor=(1.0/scale_factor))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred


