from cytopa.modules.preprocess_data import preprocess_pap, preprocess_pa
from cytopa.modules.virtual_staining import virtual_staining
from cytopa.modules.super_resolution import super_resolution
from cytopa.modules.analysis import analysis
import os
import argparse
import cv2
import numpy as np
import pandas as pd
from glob import glob

def visualize_wsi(data_path, result_path, resize_ratio=0.1, threshold=0.3):
    # CSV 파일 읽기
    csv_path = os.path.join(result_path, 'cell_counts.csv')
    df = pd.read_csv(csv_path)
    
    # 이미지 파일 목록 가져오기
    image_files = sorted(glob(os.path.join(data_path, '*.png')))
    
    # 첫 이미지로 패치 크기 확인
    first_img = cv2.imread(image_files[0])
    patch_h, patch_w = int(first_img.shape[0] * resize_ratio), int(first_img.shape[1] * resize_ratio)
    
    # i000_000 형식에서 최대 행/열 수 찾기
    max_row = max_col = 0
    for img_path in image_files:
        filename = os.path.basename(img_path)
        row = int(filename.split('_')[1][1:])+1
        col = int(filename.split('_')[2].split('.')[0])+1
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    
    # 50% 오버랩 고려한 전체 WSI 크기 계산
    wsi_h = int((max_row + 1) * patch_h * 0.5)
    wsi_w = int((max_col + 1) * patch_w * 0.5)
    
    # WSI 이미지와 알파 채널 초기화
    wsi_img = np.zeros((wsi_h, wsi_w, 3), dtype=np.uint8)
    alpha_channel = np.zeros((wsi_h, wsi_w), dtype=np.uint8)
    prob_map_area = np.zeros((wsi_h, wsi_w, 3), dtype=np.uint8)
    prob_map_count = np.zeros((wsi_h, wsi_w, 3), dtype=np.uint8)
    
    # 각 패치 처리
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # 행/열 위치 추출
        row = int(filename.split('_')[1][1:])
        col = int(filename.split('_')[2].split('.')[0])
        
        # print(f'processing {img_path}')
        # 이미지 위치 계산 (50% 오버랩 고려)
        y = int(row * patch_h * 0.5)
        x = int(col * patch_w * 0.5)
        
        # 이미지 읽기 및 리사이즈
        img = cv2.imread(img_path)
        img = cv2.resize(img, (patch_w, patch_h))
        
        # probability map 색상 계산
        cell_info = df[df['file_name'] == f'{base_name}.png'].iloc[0]
        tumor_count = cell_info['tumor_count']
        immune_count = cell_info['immune_count']
        tumor_avg_size = cell_info['tumor_avg_size']
        immune_avg_size = cell_info['immune_avg_size']
        
        # 개수와 평균 크기를 곱하여 총 면적 계산
        tumor_area = tumor_count * tumor_avg_size
        immune_area = immune_count * immune_avg_size
        total_area = tumor_area + immune_area
        
        # 면적 기반 probability map 색상 계산
        if total_area > 0:
            tumor_ratio = tumor_area / total_area
            
            if tumor_ratio <= threshold:
                color = np.array([255, 0, 0], dtype=np.uint8)  # 완전한 파랑
            elif tumor_ratio >= (1 - threshold):
                color = np.array([0, 0, 255], dtype=np.uint8)  # 완전한 빨강
            else:
                # threshold~(1-threshold) 구간을 0~1로 정규화
                normalized_ratio = (tumor_ratio - threshold) / (1 - 2*threshold)
                color = np.array([255 * (1-normalized_ratio), 0, 255 * normalized_ratio], dtype=np.uint8)
            
            color_patch_area = np.full((patch_h, patch_w, 3), color)
        else:
            color_patch_area = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)

        # 개수 기반 probability map 색상 계산
        total_count = tumor_count + immune_count
        if total_count > 0:
            tumor_ratio_count = tumor_count / total_count
            
            if tumor_ratio_count <= threshold:
                color = np.array([255, 0, 0], dtype=np.uint8)
            elif tumor_ratio_count >= (1 - threshold):
                color = np.array([0, 0, 255], dtype=np.uint8)
            else:
                normalized_ratio = (tumor_ratio_count - threshold) / (1 - 2*threshold)
                color = np.array([255 * (1-normalized_ratio), 0, 255 * normalized_ratio], dtype=np.uint8)
            
            color_patch_count = np.full((patch_h, patch_w, 3), color)
        else:
            color_patch_count = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        
        # 이미지와 probability map 합성
        wsi_img[y:y+patch_h, x:x+patch_w] = img
        prob_map_area[y:y+patch_h, x:x+patch_w] = color_patch_area
        prob_map_count[y:y+patch_h, x:x+patch_w] = color_patch_count
        alpha_channel[y:y+patch_h, x:x+patch_w] = 128  # 50% 투명도
    
    # 최종 이미지 합성 및 저장 (면적 기반)
    final_img_area = cv2.addWeighted(wsi_img, 1, prob_map_area, 0.5, 0)
    output_path_area = os.path.join(result_path, 'wsi_probability_map_area.png')
    cv2.imwrite(output_path_area, final_img_area)
    
    # 최종 이미지 합성 및 저장 (개수 기반)
    final_img_count = cv2.addWeighted(wsi_img, 1, prob_map_count, 0.5, 0)
    output_path_count = os.path.join(result_path, 'wsi_probability_map_count.png')
    cv2.imwrite(output_path_count, final_img_count)
    
    return final_img_area, final_img_count
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_img', default='pa') #pap, pa
    parser.add_argument('--target_mode', default=None) #lr, lrwcell, hr, hrwcell
    parser.add_argument('--name', default='7207') #7207, 5748, 5912-1,5912-2, 6016
    parser.add_argument('--aug', action='store_true') #7207, 5748, 5912-1,5912-2, 6016
    opt = parser.parse_args()

    source_folder = f'dataset/original_data/{opt.name}'
    target_folder = f'dataset/cropped_data/{opt.name}'
    result_folder = f'results/{opt.name}'

    if opt.target_img == 'pa':
        # Preprocessing for PA data (reverse / crop)
        print(f'##################################')
        print(f'Starting Preprocessing {opt.name}')
        preprocess_pa(opt.name, source_folder, target_folder)
        
        # Virtual staining and Super resolution for PA data
        print(f'##################################')
        print(f'Starting Virtualstaining {opt.name}')
        virtual_staining(f'{target_folder}/PA', f'{target_folder}/VPAP')

        print(f'##################################')
        print(f'Starting Superresolution {opt.name}')
        super_resolution(f'{target_folder}/VPAP', f'{target_folder}/SRVPAP')
        # Cell/Cluster Detection
        print(f'##################################')
        print(f'Starting Analysis {opt.name}')
        if opt.target_mode == 'lr' or opt.target_mode is None:
            analysis(f'{target_folder}/VPAP',f'{result_folder}/VPAP_wocell', pretrained=False, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/VPAP', f'{result_folder}/VPAP_wocell_aug' if opt.aug else f'{result_folder}/VPAP_wocell')
        if opt.target_mode == 'hr':
            analysis(f'{target_folder}/SRVPAP',f'{result_folder}/SRVPAP_wocell', pretrained=False, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/SRVPAP', f'{result_folder}/SRVPAP_wocell_aug' if opt.aug else f'{result_folder}/SRVPAP_wocell')
        if opt.target_mode == 'lrwcell' or opt.target_mode is None:
            analysis(f'{target_folder}/VPAP',f'{result_folder}/VPAP_wcell', pretrained=True, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/VPAP', f'{result_folder}/VPAP_wcell_aug' if opt.aug else f'{result_folder}/VPAP_wcell')
        if opt.target_mode == 'hrwcell':
            analysis(f'{target_folder}/SRVPAP',f'{result_folder}/SRVPAP_wcell', pretrained=True, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/SRVPAP', f'{result_folder}/SRVPAP_wcell_aug' if opt.aug else f'{result_folder}/SRVPAP_wcell')


    elif opt.target_img == 'pap': #Directly Analysis
        # 1. HR PAP Analysis
        preprocess_pap(opt.name, source_folder, target_folder, is_hr=True)
        if opt.target_mode == 'hr' or opt.target_mode is None:
            analysis(f'{target_folder}/HRPAP',f'{result_folder}/HRPAP_wocell', pretrained=False, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/HRPAP', f'{result_folder}/HRPAP_wocell_aug' if opt.aug else f'{result_folder}/HRPAP_wocell')
        if opt.target_mode == 'hrwcell':
            analysis(f'{target_folder}/HRPAP',f'{result_folder}/HRPAP_wcell', pretrained=True, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/HRPAP', f'{result_folder}/HRPAP_wcell_aug' if opt.aug else f'{result_folder}/HRPAP_wcell')
        # 2. LR PAP Analysis
        preprocess_pap(opt.name, source_folder, target_folder, is_hr=False)
        if opt.target_mode == 'lr' or opt.target_mode is None:
            analysis(f'{target_folder}/PAP',f'{result_folder}/PAP_wocell', pretrained=False, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/PAP', f'{result_folder}/PAP_wocell_aug' if opt.aug else f'{result_folder}/PAP_wocell')
        if opt.target_mode == 'lrwcell' or opt.target_mode is None:
            analysis(f'{target_folder}/PAP',f'{result_folder}/PAP_wcell', pretrained=True, aug_types=["original","rotate90", "hflip", "vflip"] if opt.aug else ["original"])
            # visualize_wsi(f'{target_folder}/PAP', f'{result_folder}/PAP_wcell_aug' if opt.aug else f'{result_folder}/PAP_wcell')

        # # 2. LR PAP to SR PAP and Analysis
        # preprocess_pap(opt.name, source_folder, target_folder, is_hr=False)
        # super_resolution(f'{target_folder}/PAP', f'{target_folder}/SRPAP')
        # analysis(f'{target_folder}/SRPAP',f'{result_folder}/SRPAP', pretrained=False)
