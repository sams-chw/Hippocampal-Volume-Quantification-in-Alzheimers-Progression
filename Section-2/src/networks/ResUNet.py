#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:44:33 2020

@author: sams_chw
"""
import torch
import torch.nn as nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels, kernal_size=1, stride=1, padding=0):
    residual = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        nn.InstanceNorm2d(out_channels)
        )
    return residual

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels)
    )
    return conv

def up_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )
    return conv

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    offset = 0 if tensor_size % 2 == 0 else 1
    return tensor[:,:,delta:tensor_size - delta-offset,delta:tensor_size - delta-offset]

class UpBlock(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()

        self.activation = nn.LeakyReLU(inplace=True)
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.unet_block = up_conv(in_channels, out_channels)
        self.res_block = conv1x1(in_channels, out_channels)
        
    def forward(self, x):
        x0 = x0_list.pop()
        x = self.transpose(x)
        y = crop_tensor(x0, x)
        out = torch.cat([x, y], axis=1)
        x = self.unet_block(torch.cat([x, y], axis=1))
        out = self.res_block(out)
        x += out
        x = self.activation(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(inplace=True)

        self.unet_block = double_conv(in_channels, out_channels)
        self.res_block = conv1x1(in_channels, out_channels)
        
    def forward(self, image):
        out = image
        x = self.unet_block(image)
        out = self.res_block(out)
        x += out
        x = self.activation(x)
        x0_list.append(x)
        return x

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num_channels= 1024
        self.num_downs = 4
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.contract = self._make_layer()
        self.expand = self._make_layer(self.num_channels, self.num_downs, True)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        

    def _make_layer(self, num_channels=64, num_downs=5, up=False):
        layers_dict = OrderedDict()
        for k in range(num_downs):
            if up:
                in_channels = num_channels // 2**k
                out_channels = num_channels // 2**(k+1)
                up = UpBlock(in_channels, out_channels)
                layer = [up]
                layers_dict[f'layer{num_downs-k-1}'] = nn.Sequential(*layer)

            else:
                in_channels = 1 if k==0 else num_channels*2**(k-1)
                out_channels = 64 if k==0 else num_channels*2**(k)
                down = DownBlock(in_channels, out_channels)
                if k < num_downs-1:
                    pool = self.max_pool_2x2
                    layer = [down , pool]
                else:
                    layer = [down]
                
                layers_dict[f'layer{k}'] = nn.Sequential(*layer)
 
        return nn.Sequential(layers_dict)

    def forward(self, image):
        global x0_list
        x0_list = list()
        
        # encoder
        x = self.contract(image)

        del x0_list[self.num_downs]

        # decoder
        x = self.expand(x)

        del globals()['x0_list']
        
        #output
        out = self.out(x)
    
        return out

def UNet():
    return ResUNet()
