#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:24:06 2024

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import isdir
import glob
import os
from torch.autograd import Function

from utils import _load_img_array, _rgb2ycbcr

from test_utils import PSNR, cal_ssim, modcrop
import matplotlib.pyplot as plt

def show(lut,path,h):
    if h:
        x = np.arange(-32,31)
        his, cum = np.histogram(lut.flatten(), bins=63)
        fig = plt.figure(figsize=(6, 4), dpi=64)
    else:
        x = np.arange(0,3)
        his, cum = np.histogram(lut.flatten(), bins=3)
        fig = plt.figure(figsize=(6, 4), dpi=4)
    ax = fig.add_subplot(111)
    ax.bar(x, his, width=0.5, color='C0')
    ax.grid(axis='y', linestyle='--')
    ax.set_axisbelow(True)
    plt.savefig(path)
    plt.show()
    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

### USER PARAMS ###
EXP_NAME = "SR-LUT"
VERSION = "test_TinyLUT_S"
UPSCALE = 4     # upscaling factor

path = './TinyLUT_LUT'
MODEL_PATH_X1_L = "./model_TinyLUT.pth"

TEST_DIR = './val/manga109/'      # Test images
p_max=[0]
def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class Round(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.round(x)
        return out
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class XQuantize(Function):
    
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class WQuantize(Function):
    @staticmethod
    def forward(ctx, w, scale):
        w = w / scale
        w = torch.clamp(w, -128, 127)
        w = torch.round(w)
        return w
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs,None


class SRNet(nn.Module):
    def __init__(self,UPSCALE):
        super(SRNet, self).__init__()
        self.upscale = 4
        self.path = path
        lm = torch.load('{}'.format(MODEL_PATH_X1_L))
        self.hh1 = lm['module.clip_hh1']
        self.hl1 = lm['module.clip_hl1']
        self.hh2 = lm['module.clip_hh2']
        self.hl2 = lm['module.clip_hl2']

        "layer 1"
        LUT_PATH_X1_L_dw = self.path+"/Model_V_L1_Depth_x4_L2.npy"    
        LUT_PATH_X1_H_dw = self.path+"/Model_V_L1_Depth_x4_H6.npy"    
        LUT_X1_L_dw = np.load(LUT_PATH_X1_L_dw).astype(np.float32).reshape(-1, 1*16)  
        LUT_X1_H_dw = np.load(LUT_PATH_X1_H_dw).astype(np.float32).reshape(-1, 1*16)
        self.LUT_X1_vector_L_dw = torch.Tensor(LUT_X1_L_dw).float()
        self.LUT_X1_vector_H_dw = torch.Tensor(LUT_X1_H_dw).float()
        
        LUT_PATH_X1_L_pw = self.path+"/Model_V_L1_Point_x4_L2.npy"   
        LUT_PATH_X1_H_pw = self.path+"/Model_V_L1_Point_x4_H6.npy"   
        LUT_X1_L_pw = np.load(LUT_PATH_X1_L_pw).astype(np.float32).reshape(-1, 1*16)  
        LUT_X1_H_pw = np.load(LUT_PATH_X1_H_pw).reshape(-1, 1*16)
        self.LUT_X1_vector_L_pw = torch.Tensor(LUT_X1_L_pw).float()
        self.LUT_X1_vector_H_pw = torch.Tensor(LUT_X1_H_pw).float()
        
        LUT_PATH_X3_L = self.path+"/Model_V_UP_x4_L2.npy"        
        LUT_PATH_X3_H = self.path+"/Model_V_UP_x4_H6.npy"  
        LUT_X3_L = np.load(LUT_PATH_X3_L).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)
        LUT_X3_H = np.load(LUT_PATH_X3_H).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)
        self.LUT_X3_vector_L = torch.Tensor(LUT_X3_L).float()    
        self.LUT_X3_vector_H = torch.Tensor(LUT_X3_H).float()
        
    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4, rounding_mode='floor')
        xl_ = image.clone()
        xh_ = image.clone()
        xl_.data = xl.data
        xh_.data = xh.data
        return xl_.type(torch.float32), xh_.type(torch.float32)
    
    def InterpTorchBatch_X1_dw(self, weight, img_in, h, w, hl):
        if hl:
            img_in = img_in + 32.0
            L = 64
        else:
            img_in = img_in
            L = 4
        x_list = []
        idx = 0
        for i in range(3):
            for j in range(3):
                x_list.append(img_in[:,:,i:i+h,j:j+w].unsqueeze(-1) + idx * L)
                idx = idx + 1
        sz = img_in.shape[0] * img_in.shape[1] * (img_in.shape[2]-2) * (img_in.shape[3]-2) * 9  
        idxx = torch.cat(x_list, dim=-1).round().type(torch.int64)
        output = weight[ idxx.flatten() ]
        output = output.reshape(sz, -1)
        output = output.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2]-2, img_in.shape[3]-2, 9, 16)).clamp(-128,127)
        output_concat = Round.apply(output.sum(-2) / 9.0)
        return output_concat
    
    def InterpTorchBatch_pw_L(self, weight, img_in):
        out = []
        S = 0
        hl1 = self.hl1.unsqueeze(-1).permute(0,-1,2,3,1).cpu()
        for c in range(16):
            scale = hl1[0,0,0,0,c].item()
            base = torch.arange(0, 4)
            first_ = Round.apply(base * scale).unique()
            first_ = torch.arange(first_.min(), first_.max() + 1)
            idxx = img_in[:,:,:,:,c]
            idxx = Round.apply(idxx).flatten().unsqueeze(-1).type(torch.int64) 
            
            sz = img_in.shape[0] * img_in.shape[1] * img_in.shape[2] * img_in.shape[3] * 16  
            p0 = weight[S:S+first_.size(0), :][ idxx ]  
            p0 = p0.reshape(sz, -1) 
            out.append(p0.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], img_in.shape[3], 16)).clamp(-128,127))  
            S = S+first_.size(0)  
        out = Round.apply(sum(out) / 16.0)
        return out
    
    def InterpTorchBatch_pw_H(self, weight, img_in):  
        out = []  
        S = 0  
        hh1 = self.hh1.unsqueeze(-1).permute(0,-1,2,3,1).cpu()        
        for c in range(16):  
            scale = hh1[0,0,0,0,c].item()
            base = torch.arange(-32, 32)
            first_ = Round.apply(base * scale).unique()
            first_ = torch.arange(first_.min(), first_.max() + 1)
            img_in_ = img_in[:,:,:,:,c]
            idxx = img_in_ - first_.min()
            idxx = Round.apply(idxx).flatten().unsqueeze(-1).type(torch.int64) 
            
            sz = img_in.shape[0] * img_in.shape[1] * img_in.shape[2] * img_in.shape[3] * 16  
            p0 = weight[S:S+first_.size(0), :][ idxx ]  
            p0 = p0.reshape(sz, -1) 
            out.append(p0.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], img_in.shape[3], 16)).clamp(-128,127))  
            S = S+first_.size(0)  
        out = Round.apply(sum(out) / 16.0)
        return out  
    
    def InterpTorchBatch_X3(self, weight, img_in, hl):
        if hl:
            out = []
            S = 0
            hh2 = self.hh2.unsqueeze(-1).permute(0,-1,2,3,1).cpu()   
            for c in range(16):
                scale = hh2[0,0,0,0,c].item()
                base = torch.arange(-32, 32)
                first_ = Round.apply(base * scale).unique()
                first_ = torch.arange(first_.min(), first_.max() + 1)
                img_in_ = img_in[:,:,:,:,c]
                idxx = img_in_ - first_.min()
                idxx = Round.apply(idxx).flatten().unsqueeze(-1).type(torch.int64) 
            
                sz = img_in.shape[0] * img_in.shape[1] * img_in.shape[2] * img_in.shape[3] * 16  
                p0 = weight[S:S+first_.size(0), :][ idxx ]  
                p0 = p0.reshape(sz, -1) 
                out.append(p0.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], img_in.shape[3], 16)).clamp(-128,127))  
                S = S+first_.size(0)  
        else:
            out = []
            S = 0
            hl2 = self.hl2.unsqueeze(-1).permute(0,-1,2,3,1).cpu() 
            for c in range(16):
                scale = hl2[0,0,0,0,c].item()
                base = torch.arange(0, 4)
                first_ = Round.apply(base * scale).unique()
                first_ = torch.arange(first_.min(), first_.max() + 1)
                idxx = img_in[:,:,:,:,c]
                idxx = Round.apply(idxx).flatten().unsqueeze(-1).type(torch.int64) 
                
                sz = img_in.shape[0] * img_in.shape[1] * img_in.shape[2] * img_in.shape[3] * 16  
                p0 = weight[S:S+first_.size(0), :][ idxx ]  
                p0 = p0.reshape(sz, -1) 
                out.append(p0.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], img_in.shape[3], 16)).clamp(-128,127))  
                S = S+first_.size(0)  
        out = Round.apply(sum(out) / 16.0)
        out = torch.clamp(out, -128,127)
        return out
    
    def forward(self, x):
        
        B, C, H, W = x.size()
        x = x.reshape(B*C, 1, H, W)
        x_l, x_h = self.low_high(x)
        
        
        "layer 1"
        batch_X1_L_dw = self.InterpTorchBatch_X1_dw(self.LUT_X1_vector_L_dw, x_l, H-2, W-2, 0)
        batch_X1_H_dw = self.InterpTorchBatch_X1_dw(self.LUT_X1_vector_H_dw, x_h, H-2, W-2, 1) 
        
        out_X1l = (batch_X1_L_dw.permute(0,-1,2,3,1)[:,:,:,:,0] + x_l[:,:,2:,2:]).clamp(0,3)
        out_X1h = (batch_X1_H_dw.permute(0,-1,2,3,1)[:,:,:,:,0] + x_h[:,:,2:,2:]).clamp(-32,31) 
        
        out_X1l = Round.apply(out_X1l * self.hl1.cpu()).clamp(0, 3)
        out_X1h = Round.apply(out_X1h * self.hh1.cpu()).clamp(-32,31)
        
        out_X1_L_pw = self.InterpTorchBatch_pw_L(self.LUT_X1_vector_L_pw, out_X1l.unsqueeze(-1).permute(0,-1,2,3,1))
        out_X1_H_pw = self.InterpTorchBatch_pw_H(self.LUT_X1_vector_H_pw, out_X1h.unsqueeze(-1).permute(0,-1,2,3,1)) 
        
        
        out_X1l = (out_X1_L_pw.permute(0,-1,2,3,1)[:,:,:,:,0] + out_X1l).clamp(0,3)
        out_X1h = (out_X1_H_pw.permute(0,-1,2,3,1)[:,:,:,:,0] + out_X1h).clamp(-32,31) 

        out_X1l = Round.apply(out_X1l * self.hl2.cpu()).clamp(0, 3)
        out_X1h = Round.apply(out_X1h * self.hh2.cpu()).clamp(-32,31)
        "layer up"
        out_X3_L = self.InterpTorchBatch_X3(self.LUT_X3_vector_L, out_X1l.unsqueeze(-1).permute(0,-1,2,3,1), 0)
        out_X3_H = self.InterpTorchBatch_X3(self.LUT_X3_vector_H , out_X1h.unsqueeze(-1).permute(0,-1,2,3,1), 1)
        
        out_X3_L = (out_X3_L.permute(0,-1,2,3,1)[:,:,:,:,0] + out_X1l).clamp(-128,127)
        out_X3_H = (out_X3_H.permute(0,-1,2,3,1)[:,:,:,:,0] + out_X1h).clamp(-128,127) 
        
        out_X3_ = torch.clamp((out_X3_H + out_X3_L), -128, 127)
                
        out_X3_ = nn.PixelShuffle(4)(out_X3_)
        
        out_X3_ = out_X3_.reshape(B, -1, self.upscale*(H-2), self.upscale*(W-2))
        
        return out_X3_
    
tlist = []
model_SR = SRNet(UPSCALE=UPSCALE)

if not isdir('result/{}'.format(str(VERSION))):
        mkdir('result/{}'.format(str(VERSION)))

# Validation
with torch.no_grad():
    model_SR.eval()
    # Test for validation images
    files_gt = glob.glob(TEST_DIR + '/HR/*.png')
    files_gt.sort()
    files_lr = glob.glob(TEST_DIR + '/LR/*.png')
    files_lr.sort()

    psnrs = []
    ssims = []
    
    for ti, fn in enumerate(files_gt):
        # Load HR image
        tmp = _load_img_array(files_gt[ti])
        val_H = np.asarray(tmp).astype(np.float32)  # HxWxC
        # Load LR image
        tmp = _load_img_array(files_lr[ti])
        val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
        val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
        val_L = val_L[np.newaxis, ...]           # BxCxHxW

        val_L = Variable(torch.from_numpy(val_L.copy()))
        t1 = time.time()
        batch_S1 = model_SR(F.pad(val_L, (2,0,2,0), mode='reflect'))

        batch_S2 = model_SR(F.pad(torch.rot90(val_L, 1, [2,3]), (2,0,2,0), mode='reflect'))
        batch_S2 = torch.rot90(batch_S2, 3, [2,3])
    
        batch_S3 = model_SR(F.pad(torch.rot90(val_L, 2, [2,3]), (2,0,2,0), mode='reflect'))
        batch_S3 = torch.rot90(batch_S3, 2, [2,3])
    
        batch_S4 = model_SR(F.pad(torch.rot90(val_L, 3, [2,3]), (2,0,2,0), mode='reflect'))
        batch_S4 = torch.rot90(batch_S4, 1, [2,3])
    
        batch_S = ( torch.clamp(batch_S1,-128,127) + torch.clamp(batch_S2,-128,127) )
        batch_S += ( torch.clamp(batch_S3,-128,127) + torch.clamp(batch_S4,-128,127) )
        batch_S_hat = torch.clamp(batch_S,-128,127)
        t2 = time.time()
        tlist.append(t2-t1)

        image_out = (batch_S_hat + 128.0).cpu().data.numpy()
        image_out = np.transpose(image_out[0,:,:,:], [1, 2, 0])  # HxWxC
        
        # Save to file
        image_out = ((image_out )).astype(np.uint8)
        Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

        # PSNR on Y channel
        img_gt = (val_H + 128.0).astype(np.uint8)
        CROP_S = 4
        img_gt = modcrop(img_gt, 4)
        
        # img_target = image_out.copy()
        if img_gt.shape[0] < image_out.shape[0]:
            image_out = image_out[:img_gt.shape[0]]
        if img_gt.shape[1] < image_out.shape[1]:
            image_out = image_out[:, :img_gt.shape[1]]
            
        if img_gt.shape[0] > image_out.shape[0]:
            image_out = np.pad(image_out, ((0, img_gt.shape[0]-image_out.shape[0]),(0,0),(0,0)))
        if img_gt.shape[1] > image_out.shape[1]:
            image_out = np.pad(image_out, ((0,0),(0, img_gt.shape[-1]-image_out.shape[1]),(0,0)))
        
        p = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S)
        s = cal_ssim(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0])
        ssims.append(s)
        psnrs.append(p)

mean_psnr = str(round(np.mean(np.asarray(psnrs)),5))
mean_ssim = str(round(np.mean(np.asarray(ssims)),5))
print(mean_psnr)
print(mean_ssim)
        