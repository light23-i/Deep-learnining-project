# For plotting
import numpy as np
import matplotlib.pyplot as plt

# For utilities
import time, sys, os

# For conversion
import cv2
import opencv_transforms.transforms as TF
import dataloader

# For everything
import torch
import torch.nn as nn
import torchvision.utils as vutils

# For our model
import model
import torchvision.models

# To ignore warning
import warnings
warnings.simplefilter("ignore", UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
else:
    print("No gpu detected")
with torch.no_grad():
    netC2S = mymodels.Color2Sketch(pretrained=True)
    netC2S.eval()

# A : Edge, B : Color
# batch_size. number of cluster
batch_size = 1
ncluster = 9

# Validation 
print('Loading Validation data...', end=' ')
val_transforms = TF.Compose([
    TF.Resize(512),
    ])
val_imagefolder = dataloader.PairImageFolder('./dataset/val', val_transforms, netC2S, ncluster)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False)
print("Done!")
print("Validation data size : {}".format(len(val_imagefolder)))


# Test
print('Loading Test data...', end=' ')
test_transforms = TF.Compose([
    TF.Resize(512),
    ])
test_imagefolder = dataloader.GetImageFolder('./dataset/test', test_transforms, netC2S, ncluster)

test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=batch_size, shuffle=False)
print("Done!")
print("Test data size : {}".format(len(test_imagefolder)))

# Reference
print('Loading Reference data...', end=' ')
refer_transforms = TF.Compose([
    TF.Resize(512),
    ])
refer_imagefolder = dataloader.GetImageFolder('./dataset/reference', refer_transforms, netC2S, ncluster)
refer_loader = torch.utils.data.DataLoader(refer_imagefolder, batch_size=1, shuffle=False)
refer_batch = next(iter(refer_loader))


temp_batch_iter = iter(refer_loader)
print("Done!")
print("Reference data size : {}".format(len(refer_imagefolder)))
nc = 3 * (ncluster + 1)
netG = mymodels.Sketch2Color(nc=nc, pretrained=True).to(device) 

netG.eval()
temp_batch = next(temp_batch_iter)
with torch.no_grad():
    
    edge = temp_batch[0].to(device)
    real = temp_batch[1].to(device)
    reference = refer_batch[1].to(device)
    color_palette = refer_batch[2]
    input_tensor = torch.cat([edge.cpu()]+color_palette, dim=1).to(device)
    fake = netG(input_tensor)
    result = torch.cat((reference, edge, fake), dim=-1).cpu()
    output = vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu().permute(1,2,0).numpy()
    plt.imsave(arr=output, fname='result.jpg')
