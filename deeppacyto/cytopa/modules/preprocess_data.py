

# %%

from glob import glob
import os
import numpy as np
import re
from tqdm import tqdm
import math
from PIL import Image
import slideio

Image.MAX_IMAGE_PIXELS = 933120000

# P1W3 : s 2.1
# 20May990 : s 1.05
# HE : zoom2(LR), zoom0(HR)

def preprocess_pap(name, source_folder, target_folder, out_size=2048, out_resize=512, res=230, is_hr=False):
    os.makedirs(f'{target_folder}/PAP', exist_ok=True)
    if is_hr:
        os.makedirs(f'{target_folder}/HRPAP', exist_ok=True)
    print(f'\nData {name} : PAP Processing')
    # PAP
    pap_list = glob(f'{source_folder}/*Pap*.ndpi')
    
    sio_slide = slideio.open_slide(pap_list[0], "NDPI")
    sio_img_s0 = sio_slide.get_scene(0)
    
    size = sio_img_s0.size

    imtmp = Image.fromarray(sio_img_s0.read_block(size = (int(size[0] / 10), int(size[1] / 10))).astype(np.uint8))
    imtmp.save(f'{source_folder}/{name}_PAP.png')

    patch_size = int(out_size * 500 / res)
    print(f'PAP size: {size} Patch size {patch_size}, expacted # of patches {(math.ceil(size[1]/patch_size)-1)} x {(math.ceil(size[0]/patch_size)-1)}')
    # Crop
    crop_list = glob(f'{source_folder}/*PAP*fg*.png')
    if len(crop_list) > 0:
        crop_img =  Image.open(crop_list[0])
        crop_ratio = (crop_img.size[0] / size[0] , crop_img.size[1] / size[1])
    for iy in tqdm(range(math.ceil(size[1]/patch_size)-1)):
        for ix in range(math.ceil(size[0]/patch_size)-1):
            if ix == math.ceil(size[0]/patch_size)-1 - 1:
                x1 = size[0]-patch_size
            else:
                x1 = ix*patch_size
                
            if iy == math.ceil(size[1]/patch_size)-1 - 1:
                y1 = size[1]-patch_size
            else:
                y1 = iy*patch_size
            
            if len(crop_list) > 0:
                crop_ary = np.asarray(crop_img.crop((int(x1*crop_ratio[0]),int(y1*crop_ratio[1]),int((x1+patch_size)*crop_ratio[0]),int((y1+patch_size)*crop_ratio[1])))).copy()
                crop_ary[crop_ary>0] = 1
                if sum(crop_ary.reshape(-1)) < (patch_size*crop_ratio[0]*patch_size*crop_ratio[1])/100*50: # Under 50%
                    continue #skip

            if is_hr:
                im_hr = sio_img_s0.read_block((x1, y1,patch_size, patch_size), size = (out_size, out_size))
                out_hr = Image.fromarray(im_hr.astype(np.uint8))
                out_hr.save(f'{target_folder}/HRPAP/{name}_i{iy:03d}_{ix:03d}.png')
            else:
                im_lr = sio_img_s0.read_block((x1, y1,patch_size, patch_size), size = (out_resize, out_resize))
                out_lr = Image.fromarray(im_lr.astype(np.uint8))
                out_lr.save(f'{target_folder}/PAP/{name}_i{iy:03d}_{ix:03d}.png')

def preprocess_pa(name, source_folder, target_folder, out_size=2048, out_resize=512):
    os.makedirs(f'{target_folder}/PA', exist_ok=True)
    image_list = glob(f'{source_folder}/tiled_image.tif')
    merge_img = Image.open(image_list[0])

    merge_img.save(f'{source_folder}/{name}_PA.png')
    
    #------------------------------------------------------ Main Preprocess ------------------------------------------------------
    
    size = merge_img.size
    
    for iy in tqdm(range(math.ceil(size[1]/(out_size/2))-1)):
        for ix in range(math.ceil(size[0]/(out_size/2))-1):
            if ix == math.ceil(size[0]/(out_size/2))-1 - 1:
                x1 = size[0]-out_size
            else:
                x1 = ix*(out_size/2)
                
            if iy == math.ceil(size[1]/(out_size/2))-1 - 1:
                y1 = size[1]-out_size
            else:
                y1 = iy*(out_size/2)

            im_re = np.asarray(merge_img.crop((x1,y1,x1+out_size,y1+out_size)))
            im_re = (255-im_re) #reverse
            out = Image.fromarray(im_re.astype(np.uint8))
            
            out = out.resize((out_resize,out_resize))

            out.save(f'{target_folder}/PA/{name}_i{iy:03d}_{ix:03d}.png')
                