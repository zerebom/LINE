import torch
import torch.utils.data as data
from torchvision import transforms
import pathlib
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=True, weight_init=True):
        self.dropout = dropout
        super(double_conv, self).__init__()
        conv1_out = out_ch // 2

        if conv1_out < in_ch:
            conv1_out = in_ch

        self.dropout_layer = nn.Dropout3d(p=0.5)
        self.conv = nn.Sequential(
            # in,out,kernel_size,padding
            nn.Conv3d(in_ch, conv1_out, 3, padding=1),
            nn.BatchNorm3d(conv1_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

        if weight_init:
            self.conv.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            double_conv(in_ch, out_ch, dropout=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, init_weights=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
        if init_weights:
            self.up.apply(self.init_weights)

        self.conv = double_conv(int(in_ch * 3 / 4), out_ch, dropout=False)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose3d):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        diffY = x1.size()[4] - x2.size()[4]
        x2 = F.pad(x2, (diffZ // 2, int(diffZ / 2),
                        diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)

        return x


class UNet3D(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(UNet3D, self).__init__()
        self.in_ch = input_shape[1]
        self.inc = inconv(self.in_ch, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == "__main__":
    # unet=UNet3D([8, 16, 1, 48, 48],3)
    net_shape = [8, 1, 16, 48, 48]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet = UNet3D(net_shape, 3)
    unet.to(device)
    print(device)

    dummy_img = torch.rand(net_shape).to(device)

    output = unet(dummy_img)
    print('output:', output)
