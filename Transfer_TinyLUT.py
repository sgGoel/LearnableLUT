#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:14:24 2023

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# USER PARAMS
UPSCALE = 4                  # upscaling factor
MODEL_PATH_X1_L = "./model_TinyLUT.pth"
Lut_path = './TinyLUT_LUT'

h = 0
nkernel = 3
clist = np.ones((1,16,1,1))

class XQuantize(Function):
    
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None
    
blist = np.zeros((nkernel**2,nkernel,nkernel))

for l in range(blist.shape[0]):
    blist[l][l//nkernel][l%nkernel] = 1

### A lightweight deep network ###
nkernel = 3

blist = np.zeros((nkernel**2,nkernel,nkernel))

for l in range(blist.shape[0]):
    blist[l][l//nkernel][l%nkernel] = 1  
    
class PointOneChannel(torch.nn.Module):
    def __init__(self):
        super(PointOneChannel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self,x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return XQuantize.apply(x)

class UpOneChannel(torch.nn.Module):
    def __init__(self):
        super(UpOneChannel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            
    def forward(self,x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.conv5(x)
        return XQuantize.apply(x)
    
class DepthWise(torch.nn.Module):
    def __init__(self):
        super(DepthWise, self).__init__()
        channle_ = 16
        " High "
        kernel11 = blist[0]
        kernel11 = torch.FloatTensor(kernel11).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight11 = nn.Parameter(data = kernel11, requires_grad=True)
        mask11 = blist[0]
        mask11 = torch.FloatTensor(mask11).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask11 = nn.Parameter(data = mask11, requires_grad=False)
        
        kernel21 = blist[1]
        kernel21 = torch.FloatTensor(kernel21).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight21 = nn.Parameter(data = kernel21, requires_grad=True)
        mask21 = blist[1]
        mask21 = torch.FloatTensor(mask21).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask21 = nn.Parameter(data = mask21, requires_grad=False)
        
        kernel31 = blist[2]
        kernel31 = torch.FloatTensor(kernel31).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight31 = nn.Parameter(data = kernel31, requires_grad=True)
        mask31 = blist[2]
        mask31 = torch.FloatTensor(mask31).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask31 = nn.Parameter(data = mask31, requires_grad=False)
        
        kernel41 = blist[3]
        kernel41 = torch.FloatTensor(kernel41).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight41 = nn.Parameter(data = kernel41, requires_grad=True)
        mask41 = blist[3]
        mask41 = torch.FloatTensor(mask41).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask41 = nn.Parameter(data = mask41, requires_grad=False)
        
        kernel51 = blist[4]
        kernel51 = torch.FloatTensor(kernel51).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight51 = nn.Parameter(data = kernel51, requires_grad=True)
        mask51 = blist[4]
        mask51 = torch.FloatTensor(mask51).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask51 = nn.Parameter(data = mask51, requires_grad=False)
        
        kernel61 = blist[5]
        kernel61 = torch.FloatTensor(kernel61).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight61 = nn.Parameter(data = kernel61, requires_grad=True)
        mask61 = blist[5]
        mask61 = torch.FloatTensor(mask61).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask61 = nn.Parameter(data = mask61, requires_grad=False)
                
        kernel71 = blist[6]
        kernel71 = torch.FloatTensor(kernel71).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight71 = nn.Parameter(data = kernel71, requires_grad=True)
        mask71 = blist[6]
        mask71 = torch.FloatTensor(mask71).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask71 = nn.Parameter(data = mask71, requires_grad=False)
        
        kernel81 = blist[7]
        kernel81 = torch.FloatTensor(kernel81).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight81 = nn.Parameter(data = kernel81, requires_grad=True)
        mask81 = blist[7]
        mask81 = torch.FloatTensor(mask81).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask81 = nn.Parameter(data = mask81, requires_grad=False)
        
        kernel91 = blist[8]
        kernel91 = torch.FloatTensor(kernel91).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight91 = nn.Parameter(data = kernel91, requires_grad=True)
        mask91 = blist[8]
        mask91 = torch.FloatTensor(mask91).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask91 = nn.Parameter(data = mask91, requires_grad=False)
        
        " Low "
        
        kernel111 = blist[0]
        kernel111 = torch.FloatTensor(kernel111).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight111 = nn.Parameter(data = kernel111, requires_grad=True)
        mask111 = blist[0]
        mask111 = torch.FloatTensor(mask111).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask111 = nn.Parameter(data = mask111, requires_grad=False)
        
        kernel211 = blist[1]
        kernel211 = torch.FloatTensor(kernel211).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight211 = nn.Parameter(data = kernel211, requires_grad=True)
        mask211 = blist[1]
        mask211 = torch.FloatTensor(mask211).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask211 = nn.Parameter(data = mask211, requires_grad=False)
        
        kernel311 = blist[2]
        kernel311 = torch.FloatTensor(kernel311).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight311 = nn.Parameter(data = kernel311, requires_grad=True)
        mask311 = blist[2]
        mask311 = torch.FloatTensor(mask311).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask311 = nn.Parameter(data = mask311, requires_grad=False)
        
        kernel411 = blist[3]
        kernel411 = torch.FloatTensor(kernel411).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight411 = nn.Parameter(data = kernel411, requires_grad=True)
        mask411 = blist[3]
        mask411 = torch.FloatTensor(mask411).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask411 = nn.Parameter(data = mask411, requires_grad=False)
        
        kernel511 = blist[4]
        kernel511 = torch.FloatTensor(kernel511).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight511 = nn.Parameter(data = kernel511, requires_grad=True)
        mask511 = blist[4]
        mask511 = torch.FloatTensor(mask511).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask511 = nn.Parameter(data = mask511, requires_grad=False)
        
        kernel611 = blist[5]
        kernel611 = torch.FloatTensor(kernel611).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight611 = nn.Parameter(data = kernel611, requires_grad=True)
        mask611 = blist[5]
        mask611 = torch.FloatTensor(mask611).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask611 = nn.Parameter(data = mask611, requires_grad=False)
                
        kernel711 = blist[6]
        kernel711 = torch.FloatTensor(kernel711).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight711 = nn.Parameter(data = kernel711, requires_grad=True)
        mask711 = blist[6]
        mask711 = torch.FloatTensor(mask711).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask711 = nn.Parameter(data = mask711, requires_grad=False)
        
        kernel811 = blist[7]
        kernel811 = torch.FloatTensor(kernel811).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight811 = nn.Parameter(data = kernel811, requires_grad=True)
        mask811 = blist[7]
        mask811 = torch.FloatTensor(mask811).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask811 = nn.Parameter(data = mask811, requires_grad=False)
        
        kernel911 = blist[8]
        kernel911 = torch.FloatTensor(kernel911).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight911 = nn.Parameter(data = kernel911, requires_grad=True)
        mask911 = blist[8]
        mask911 = torch.FloatTensor(mask911).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask911 = nn.Parameter(data = mask911, requires_grad=False)
       
    def forward(self, xh, xl, h):
        B,C,H,W = xh.size()
        
        x1 = F.conv2d(xh, self.weight11*self.mask11, padding=0, groups=C)
        x1 = x1.clamp(-128,127)
        
        x2 = F.conv2d(xh, self.weight21*self.mask21, padding=0, groups=C)
        x2 = x2.clamp(-128,127)
        
        x3 = F.conv2d(xh, self.weight31*self.mask31, padding=0, groups=C)
        x3 = x3.clamp(-128,127)
        
        x4 = F.conv2d(xh, self.weight41*self.mask41, padding=0, groups=C)
        x4 = x4.clamp(-128,127)
        
        x5 = F.conv2d(xh, self.weight51*self.mask51, padding=0, groups=C)
        x5 = x5.clamp(-128,127)
        
        x6 = F.conv2d(xh, self.weight61*self.mask61, padding=0, groups=C)
        x6 = x6.clamp(-128,127)
        
        x7 = F.conv2d(xh, self.weight71*self.mask71, padding=0, groups=C)
        x7 = x7.clamp(-128,127)
        
        x8 = F.conv2d(xh, self.weight81*self.mask81, padding=0, groups=C)
        x8 = x8.clamp(-128,127)
        
        x9 = F.conv2d(xh, self.weight91*self.mask91, padding=0, groups=C)
        x9 = x9.clamp(-128,127)
        
        xh = [XQuantize.apply(x1), XQuantize.apply(x2), XQuantize.apply(x3), XQuantize.apply(x4), XQuantize.apply(x5), XQuantize.apply(x6), XQuantize.apply(x7), XQuantize.apply(x8), XQuantize.apply(x9)]
       
        
        x11 = F.conv2d(xl, self.weight111*self.mask111, padding=0, groups=C)
        x11 = x11.clamp(-128,127)
        
        x21 = F.conv2d(xl, self.weight211*self.mask211, padding=0, groups=C)
        x21 = x21.clamp(-128,127)
        
        x31 = F.conv2d(xl, self.weight311*self.mask311, padding=0, groups=C)
        x31 = x31.clamp(-128,127)
        
        x41 = F.conv2d(xl, self.weight411*self.mask411, padding=0, groups=C)
        x41 = x41.clamp(-128,127)
        
        x51 = F.conv2d(xl, self.weight511*self.mask511, padding=0, groups=C)
        x51 = x51.clamp(-128,127)
        
        x61 = F.conv2d(xl, self.weight611*self.mask611, padding=0, groups=C)
        x61 = x61.clamp(-128,127)
        
        x71 = F.conv2d(xl, self.weight711*self.mask711, padding=0, groups=C)
        x71 = x71.clamp(-128,127)
        
        x81 = F.conv2d(xl, self.weight811*self.mask811, padding=0, groups=C)
        x81 = x81.clamp(-128,127)
        
        x91 = F.conv2d(xl, self.weight911*self.mask911, padding=0, groups=C)
        x91 = x91.clamp(-128,127)
        
        xhl = [XQuantize.apply(x11), XQuantize.apply(x21), XQuantize.apply(x31), XQuantize.apply(x41), XQuantize.apply(x51), XQuantize.apply(x61), XQuantize.apply(x71), XQuantize.apply(x81), XQuantize.apply(x91)]

        if h:
            return xh
        else:
            return xhl
        
class PointConv(torch.nn.Module):
    def __init__(self):
        super(PointConv, self).__init__()
        self.HConv = nn.ModuleList()
        self.LConv = nn.ModuleList()
        
        for i in range(16):
            self.HConv.append(PointOneChannel())
            self.LConv.append(PointOneChannel())
            
    def forward(self,xh,xl,h,s,l):
        if s:
            xhout = []
            xlout = []
            for i in range(16):
                xhout.append(XQuantize.apply((self.HConv[i](xh[:,i:i+1,:,:])).clamp(-128,127)))
                xlout.append(XQuantize.apply((self.LConv[i](xl[:,i:i+1,:,:])).clamp(-128,127)))
                
            if h:
                return xhout
            else:
                return xlout
        else:
            xhl = self.HConv[l](xh).clamp(-128,127)
            xll = self.LConv[l](xl).clamp(-128,127)
            if h:
                return xhl
            else:
                return xll
        
class UpConv(torch.nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.HConv = nn.ModuleList()
        self.LConv = nn.ModuleList()
        for i in range(16):
            self.HConv.append(UpOneChannel())
            self.LConv.append(UpOneChannel())
    def forward(self,xh,xl,h,s,l):
        if s:
            xhout = []
            xlout = []
            for i in range(16):
                xhout.append(XQuantize.apply((self.HConv[i](xh[:,i:i+1,:,:])).clamp(-128,127)))
                xlout.append(XQuantize.apply((self.LConv[i](xl[:,i:i+1,:,:])).clamp(-128,127)))
                
            if h:
                return xhout
            else:
                return xlout
        else:
            xhl = self.HConv[l](xh).clamp(-128,127)
            xll = self.LConv[l](xl).clamp(-128,127)
            if h:
                return xhl
            else:
                return xll
        

class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()
        self.depthconv1 = DepthWise()
        self.pointconv1 = PointConv()
        self.upconv = UpConv()
        self.upscale = upscale

        cl = torch.FloatTensor(clist*0.8)
        self.clip_hh1 = nn.Parameter(data = cl, requires_grad=False)
        self.clip_hl1 = nn.Parameter(data = cl, requires_grad=False)
        
        self.clip_hl2 = nn.Parameter(data = cl, requires_grad=False)
        self.clip_hh2 = nn.Parameter(data = cl, requires_grad=False)
        
    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4, rounding_mode='floor')
        xl_ = image.clone()
        xh_ = image.clone()
        xl_.data = xl.data
        xh_.data = xh.data
        return xl_.type(torch.float32), xh_.type(torch.float32)
    
    def forward(self, xh, xl, layer=1, h=True, depth=False, point=False, up=False, s=1, l=1):
        if layer == 1:
            '''
            layer 1
            '''
            if h :
                if depth:
                    xh = self.depthconv1(xh, xl, h)
                    return xh
                if point:       
                    xh = self.pointconv1(xh, xl, h, s, l)
                    return xh
            else:
                if depth:
                    xl = self.depthconv1(xh, xl, h)
                    return xl
                if point:       
                    xl = self.pointconv1(xh, xl, h, s, l)
                    return xl
        if up:
            if h:
                xh = self.upconv(xh, xl, h,s,l)
                return xh
            else:
                xl = self.upconv(xh, xl, h,s,l)
                return xl
    
model_SR = SRNet(upscale=UPSCALE).cuda()
model_SR = nn.DataParallel(model_SR,[0])

lm = torch.load('{}'.format(MODEL_PATH_X1_L))
model_SR.load_state_dict(lm)

if not os.path.isdir(Lut_path):
    os.mkdir(Lut_path)

### Extract input-output pairs
with torch.no_grad():
    model_SR.eval()
    
    if h == 1:
        '''Depth Layer'''
        base = torch.arange(0, 64, 1)
        L = base.size(0)
    
        # 2D input
        first_ = base.cuda().unsqueeze(1)
        first__ = torch.cat([first_, first_], 1)
        first___ = torch.cat([first__, first__], 1)
        first_8 = torch.cat([first___, first___], 1)
        first_9 = torch.cat([first_8, first_], 1)
        
        # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
        input_tensor = first_9.unsqueeze(1).unsqueeze(1).reshape(-1,1,3,3).float() - 32.0
    
        outputs = []
        for i in range(9):
            batch_output = model_SR(input_tensor, input_tensor, layer=1, h=True, depth=True, point=False, up=False)
            results = torch.round(torch.clamp(batch_output[i], -128,127)).cpu().data.numpy().astype(np.int8)
            outputs.append(results)

        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_L1_Depth_x{}_H6".format(UPSCALE), results_cat)

        '''Point Layer'''
        outputs = []
        for j in range(16):
            base = torch.arange(-32, 32)
        
            # 2D input
            first_ = XQuantize.apply(base.cuda() * lm['module.clip_hh1'][0,j,0,0].item()).unique() 
            first_ = torch.arange(first_.min(), first_.max() + 1)
            input_tensor_ = first_.unsqueeze(1).repeat(1, 16).unsqueeze(-1).unsqueeze(-1).float()     
        
            batch_output = model_SR(input_tensor_[:,j:j+1,:,:], input_tensor_[:,j:j+1,:,:], layer=1, h=True, depth=False, point=True, up=False, s=0, l=j)
            results = torch.round(batch_output).clamp(-128,127).cpu().data.numpy().astype(np.int8)
            outputs.append(results)
              
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_L1_Point_x{}_H6".format(UPSCALE), results_cat)
        
        '''Up Layer'''
        outputs = []
        for k in range(16):  
            base = torch.arange(-32, 32)
        
            # 2D input
            first_ = XQuantize.apply(base.cuda() * lm['module.clip_hh2'][0,k,0,0].item()).unique() 
            first_ = torch.arange(first_.min(), first_.max() + 1)
            input_tensor_ = first_.unsqueeze(1).repeat(1, 16).unsqueeze(-1).unsqueeze(-1).float()   
        
            batch_output = model_SR(input_tensor_[:,k:k+1,:,:], input_tensor_[:,k:k+1,:,:], layer=1, h=True, depth=False, point=False, up=True, s=0, l=k)
            results = torch.round(batch_output).clamp(-128,127).cpu().data.numpy().astype(np.int8)
            outputs.append(results)
            
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_UP_x{}_H6".format(UPSCALE), results_cat)
        
    elif h != 1:
        '''Depth Layer'''
        base = torch.arange(0, 4, 1)   # 0-16
        L = base.size(0)
    
        # 2D input
        first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
        first_ = base.cuda().unsqueeze(1)
        first__ = torch.cat([first_, first_], 1)
        first___ = torch.cat([first__, first__], 1)
        first_8 = torch.cat([first___, first___], 1)
        first_9 = torch.cat([first_8, first_], 1)
        
        # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
        input_tensor = first_9.unsqueeze(1).unsqueeze(1).reshape(-1,1,3,3).float() 
        
        outputs = []
        for i in range(9):
            batch_output = model_SR(input_tensor,input_tensor, layer=1, h=False, depth=True, point=False, up=False)
            results = torch.round(torch.clamp(batch_output[i], -128,127)).cpu().data.numpy().astype(np.int8)
            outputs.append(results)

        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_L1_Depth_x{}_L2".format(UPSCALE), results_cat)
        
        
        '''Point Layer'''
        outputs = []
        for j in range(16):
            base = torch.arange(0, 4)
        
            # 2D input
            first_ = XQuantize.apply(base.cuda() * lm['module.clip_hl1'][0,j,0,0].item()).unique() 
            first_ = torch.arange(first_.min(), first_.max() + 1)
            input_tensor_ = first_.unsqueeze(1).repeat(1, 16).unsqueeze(-1).unsqueeze(-1).float()     
        
            batch_output = model_SR(input_tensor_[:,j:j+1,:,:], input_tensor_[:,j:j+1,:,:], layer=1, h=False, depth=False, point=True, up=False, s=0, l=j)
            results = torch.round(torch.clamp(batch_output, -128,127)).cpu().data.numpy().astype(np.int8)
            outputs.append(results)
              
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_L1_Point_x{}_L2".format(UPSCALE), results_cat)
        
        
        '''Up Layer'''
        outputs = []
        for k in range(16): 
            base = torch.arange(0, 4)
        
            # 2D input
            first_ = XQuantize.apply(base.cuda() * lm['module.clip_hl2'][0,k,0,0].item()).unique() 
            first_ = torch.arange(first_.min(), first_.max() + 1)
            input_tensor_ = first_.unsqueeze(1).repeat(1, 16).unsqueeze(-1).unsqueeze(-1).float()   
            
            batch_output = model_SR(input_tensor_[:,k:k+1,:,:], input_tensor_[:,k:k+1,:,:], layer=1, h=False, depth=False, point=False, up=True, s=0, l=k)
            results = torch.round(torch.clamp(batch_output, -128, 127)).cpu().data.numpy().astype(np.int8)
            outputs.append(results)
            
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)
        np.save(Lut_path+"/Model_V_UP_x{}_L2".format(UPSCALE), results_cat)
        


