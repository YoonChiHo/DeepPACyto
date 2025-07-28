import os, argparse, sys, shutil, warnings, glob
from datetime import datetime
import matplotlib.pyplot as plt
from math import log2, log10
import pandas as pd
import numpy as np
from collections import OrderedDict

from torchvision import transforms, utils
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from skimage import exposure, color, io, img_as_float, img_as_ubyte
#from skimage.util import view_as_windows, pad, montage
#from PIL import Image, ImageFilter
#import imagej

import data_loader as data
import models
from models_unet import *

import pytorch_fid.fid_score as fid_score


def paired_dataloader(args, phase='train'):
    transformed_dataset = data.Paired_Dataset_img(file_rt=args.dataset,
                                              phase = phase,
                                              img_size=args.patch_size,
                                              transform=data.Compose([data.ToTensor()])
                                              )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader


def test_dataloader(args, phase='test'):
    transformed_dataset = data.TEST_Dataset_img(file_rt=args.dataset,
                                              phase = phase,
                                              img_size=args.patch_size,
                                              transform=data.Compose([data.ToTensorTest()])
                                              )
    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return dataloader
def compute_p_snr(path_input, path_ref):
    MSE = nn.MSELoss()
    imgs_input = glob.glob(os.path.join(path_input, '*.tiff'))
    imgs_ref = glob.glob(os.path.join(path_ref, '*.tiff'))
    ave_psnr = 0
    for i in range(len(imgs_input)):
        img_input = torch.from_numpy(img_as_float(io.imread(imgs_input[i]).transpose(2, 1, 0)))               
        img_ref = torch.from_numpy(img_as_float(io.imread(imgs_ref[i]).transpose(2, 1, 0)))
        img_input = img_input[None, :]
        img_ref = img_ref[None, :]             
        mse = MSE(img_input, img_ref)               
        psnr = 10 * log10(1 / mse.item())
        ave_psnr += psnr
    ave_psnr = ave_psnr / len(imgs_input)
    return ave_psnr

def print_output(generator, dataloader_valid, device='cuda:0'):
    os.makedirs('output/print', exist_ok=True)
    os.makedirs('output/print/lr', exist_ok=True)
    os.makedirs('output/print/hr', exist_ok=True)
    os.makedirs('output/print/sr', exist_ok=True)
    with torch.no_grad(): 
        generator.eval()
        print("=> Printing sampled patches")
        for k, batch in enumerate(dataloader_valid):     
            input, target = batch['input'].to(device), batch['output'].to(device)
            imgs_input =input.float().to(device)
            prediction = generator(imgs_input)
            target = target.float()
            for i in range(target.shape[0]):
                utils.save_image(imgs_input[i], 'output/print/lr/{}_{}.tiff'.format(k, i))
                utils.save_image(target[i], 'output/print/hr/{}_{}.tiff'.format(k, i))
                utils.save_image(prediction[i], 'output/print/sr/{}_{}.tiff'.format(k, i))
            sys.stdout.write("\r ==> Batch {}/{}".format(k+1, len(dataloader_valid)))
        print("\n Computing FID score")
        fid = fid_score.calculate_fid_given_paths(('output/print/sr', 'output/print/hr'), 8, device, 2048)
        print("\n Computing PSNR")
        psnr = compute_p_snr('output/print/sr', 'output/print/hr')
        print("FID score: {}, PSNR: {}".format(fid, psnr))
    return fid, psnr

def print_test_output(generator, dataloader_valid, target_folder, device='cuda:0'):
    os.makedirs('output/test', exist_ok=True)
    os.makedirs('output/test/lr', exist_ok=True)
    os.makedirs('output/test/sr', exist_ok=True)
    with torch.no_grad(): 
        generator.eval()
        print("=> Printing sampled patches")
        for k, batch in enumerate(dataloader_valid):     
            input= batch['input'].to(device) 
            name= batch['name']
            imgs_input =input.float().to(device)
            prediction = generator(imgs_input)
            #target = target.float()
            for i in range(input.shape[0]):
                #utils.save_image(imgs_input[i], f'output/test/lr/{name[0]}')
                utils.save_image(prediction[i], f'{target_folder}/{name[0]}')
            sys.stdout.write("\r ==> Batch {}/{}".format(k+1, len(dataloader_valid)))
        #print("\n Computing FID score")
        #fid = fid_score.calculate_fid_given_paths(('output/print/sr', 'output/print/hr'), 8, 'cuda:0', 2048)
        #print("\n Computing PSNR")
        #psnr = compute_p_snr('output/print/sr', 'output/print/hr')
        #print("FID score: {}, PSNR: {}".format(fid, psnr))
    
def super_resolution(input, target):
    parser = argparse.ArgumentParser(description='Train WSISR on compressed TMA dataset')
    parser.add_argument('--istrain', default=False, type=bool) #
    parser.add_argument('--target_checkpoints', default='./checkpoints/superresolution.pth', type=str, help='Dataset folder name')
    parser.add_argument('--dataset', default='./dataset/', type=str, help='Dataset folder name')
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
    parser.add_argument('--patch_size', default=2048, type=int, help='Patch size')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--num-epochs', default=200, type=int, help='Number of epochs, more epochs are desired for GAN training')   #37h for 200e
    parser.add_argument('--g-lr', default=0.0001, type=float, help='Learning rate of the generator')
    parser.add_argument('--d-lr', default=0.00001, type=float, help='Learning rate of the descriminator')
    parser.add_argument('--percep-weight', default=0.01, type=float, help='GAN loss weight')
    parser.add_argument('--run-from', default=None, type=str, help='Load weights from a previous run, use folder name in [weights] folder')
    parser.add_argument('--gan', default=1, type=int, help='Use GAN')
    parser.add_argument('--num-critic', default=1, type=int, help='Iteration interval for training the descriminator') 
    parser.add_argument('--test-interval', default=50, type=int, help='Epoch interval for FID score testing')
    parser.add_argument('--print-interval', default=10, type=int, help='Epoch interval for output printing')
    parser.add_argument('--in-folder', default='low', type=str, help='Low NA image folder name')
    parser.add_argument('--out-folder', default='high', type=str, help='High NA image folder name')   
    parser.add_argument('--gpus', default='0', type=str, help='High NA image folder name')      
    args = parser.parse_args([])

    args.dataset = input
    warnings.filterwarnings('ignore')
    device = torch.device(f'cuda:{args.gpus}')
    tensor = torch.cuda.FloatTensor
    test_dataset = test_dataloader(args, 'test')
    #generator = models.Generator()
    #generator = ResNet(3, 3, 64, 'inorm', nblk=9)
    generator = UNet(3, 3, 64, 'inorm')
    generator.to(device)

    
    init_net(generator, init_type='normal', init_gain=0.02, gpu_ids=0)

    a_weight = torch.load(args.target_checkpoints, map_location='cpu')
    generator.load_state_dict(a_weight)
    os.makedirs(target, exist_ok=True)
    print_test_output(generator, test_dataset, target, device)
    
if __name__ == '__main__':
    main()

