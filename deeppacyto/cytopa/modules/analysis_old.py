# Import Python Standard Library dependencies
from functools import partial
from glob import glob

# Import utility functions
from cjm_psl_utils.core import download_file
from cjm_pil_utils.core import resize_img
from cjm_pytorch_utils.core import tensor_to_pil, get_torch_device, set_seed, move_data_to_device

# Import PIL for image manipulation
from PIL import Image

# Import PyTorch dependencies
import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2  as transforms

# Import Mask R-CNN
from models.base import model_initialize

import argparse
import os
from utils import get_custom_colors, sliding_window_crop_tensor, combine_patches
from test_cell import cell_prediction
import json
from pathlib import Path

from datasets import CustomDataset_test
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

import torch

def compute_dice(pred, gt):
    pred = pred.bool()
    gt = gt.bool()
    intersection = (pred & gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    return 2.0 * intersection / union if union > 0 else 1.0
def evaluate_maskrcnn(predictions, groundtruth, iou_threshold=0.5, cell_threshold=0.5):
    """
    Evaluate Mask R-CNN results including boxes, classes, and masks.
    """
    all_ious, all_dices = [], []
    all_pred_classes, all_gt_classes = [], []
    unmatched_gt_labels = []  # Track GT labels with no matching predictions

    unique_labels = [1, 2]  # Define possible classes

    # Initialize metrics
    precision = np.full(len(unique_labels), -1.0)
    recall = np.full(len(unique_labels), -1.0)
    f1_score = np.full(len(unique_labels), -1.0)
    tp_list = np.zeros(len(unique_labels), dtype=int)
    fp_list = np.zeros(len(unique_labels), dtype=int)
    fn_list = np.zeros(len(unique_labels), dtype=int)

    for pred, gt in zip(predictions, groundtruth):
        pred_boxes = pred['boxes']
        pred_classes = pred['labels']
        pred_masks = torch.stack([
            torch.where(mask >= cell_threshold, 1, 0).to(torch.bool)
            for mask in pred['masks']
        ])
        
        gt_boxes = gt['boxes']
        gt_classes = gt['labels']
        gt_masks = gt['masks']

        # Filter groundtruth and prediction to ignore class 0
        gt_valid_indices = gt_classes > 0
        pred_valid_indices = pred_classes > 0

        gt_boxes = gt_boxes[gt_valid_indices]
        gt_classes = gt_classes[gt_valid_indices]
        gt_masks = gt_masks[gt_valid_indices]

        pred_boxes = pred_boxes[pred_valid_indices]
        pred_classes = pred_classes[pred_valid_indices]
        pred_masks = pred_masks[pred_valid_indices]

        # Compute IoU between GT and predictions
        ious = compute_iou_matrix(gt_boxes, pred_boxes)
        gt_matched = set()  # To track matched GT
        pred_matched = set()  # To track matched predictions

        for gt_idx, gt_label in enumerate(gt_classes):
            # Find matching prediction with IoU >= threshold
            matching_pred_indices = np.where(ious[gt_idx] >= iou_threshold)[0]
            if len(matching_pred_indices) > 0:
                # Take the best match (highest IoU)
                best_pred_idx = matching_pred_indices[np.argmax(ious[gt_idx, matching_pred_indices])]
                pred_label = pred_classes[best_pred_idx]

                # Update true positives if labels match, else false negatives
                if pred_label == gt_label:
                    tp_list[gt_label - 1] += 1
                else:
                    fn_list[gt_label - 1] += 1
                    fp_list[pred_label - 1] += 1

                gt_matched.add(gt_idx)
                pred_matched.add(best_pred_idx)

                # Compute Dice for matched pair
                dice = compute_dice(pred_masks[best_pred_idx], gt_masks[gt_idx])
                all_dices.append(dice)
            else:
                unmatched_gt_labels.append(gt_label)

        # # not All Labeled Dataset 
        # # False positives for unmatched predictions
        # for pred_idx, pred_label in enumerate(pred_classes):
        #     if pred_idx not in pred_matched:
        #         fp_list[pred_label - 1] += 1

        # False negatives for unmatched GTs
        for gt_idx, gt_label in enumerate(gt_classes):
            if gt_idx not in gt_matched:
                fn_list[gt_label - 1] += 1
                all_dices.append(0)  # Dice for unmatched GT is 0

        all_ious.extend(ious.flatten())

    # Calculate precision, recall, and F1 for each label
    for idx, label in enumerate(unique_labels):
        tp, fp, fn = tp_list[idx], fp_list[idx], fn_list[idx]
        if tp + fp + fn > 0:  # Only calculate if the class is present
            precision[idx] = tp / (tp + fp + 1e-9)
            recall[idx] = tp / (tp + fn + 1e-9)
            f1_score[idx] = 2 * (precision[idx] * recall[idx]) / (precision[idx] + recall[idx] + 1e-9)

    return {
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'mean_dice': np.mean(all_dices) if all_dices else 0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp_list': tp_list,
        'fp_list': fp_list,
        'fn_list': fn_list,
    }

def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of bounding boxes."""
    ious = []
    for box1 in boxes1:
        ious.append([compute_iou(box1, box2) for box2 in boxes2])
    return np.array(ious)

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two boxes."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def cluster_prediction(model, model_cell, input_tensor, cluster_crop_size, cluster_seg_size, cell_crop_size, overlap=0, cluster_threshold=0.5, cluster_iou_threshold=0.2, cell_threshold=0.5, cell_iou_threshold=0.2, cell_info=None, device=None):

    # Step 1: Generate patches
    patches, coords = sliding_window_crop_tensor(input_tensor, cluster_crop_size, overlap)

    # Step 2: Process each patch
    if model_cell != None:
        patches_results_cell = [[] for _ in range(input_tensor.shape[0])]
    patches_results = [[] for _ in range(input_tensor.shape[0])]
    for patch in patches:
        # Resize the patch to the input size
        resized_patch = F.interpolate(patch, size=(cluster_seg_size, cluster_seg_size), mode="bilinear", align_corners=False)
        
        # Model prediction
        with torch.no_grad():  # Ensure no gradients are calculated
            if model_cell != None:
                #output_cell = model_cell(input_tensor)
                output_cell = cell_prediction(model_cell, resized_patch, cell_crop_size, overlap=0, threshold=cell_threshold, iou_threshold=cell_iou_threshold)
                
                for bs in range(len(output_cell)):
                    # Extract bboxes, scores, and labels
                    scores = output_cell[bs]['scores'].cpu()
                    scores_mask = scores > cell_threshold
                    bboxes = output_cell[bs]['boxes'][scores_mask].cpu()
                    scores_filtered = scores[scores_mask]
                    labels_filtered = output_cell[bs]['labels'][scores_mask].cpu()
                    masks_filtered = output_cell[bs]['masks'][scores_mask].cpu()

                    # Collect results for this patch
                    patches_results_cell[bs].append((bboxes, scores_filtered, labels_filtered, masks_filtered))
                for output_cell_ in output_cell:
                    output_cell_['masks'] = torch.stack([
                        torch.where(mask >= cell_threshold, 1, 0).to(torch.bool)
                        for mask in output_cell_['masks']
                    ])
                    output_cell_['masks'] = output_cell_['masks'].to(device)
                    output_cell_['boxes'] = output_cell_['boxes'].to(device)
                    output_cell_['scores'] = output_cell_['scores'].to(device)
                    output_cell_['labels'] = output_cell_['labels'].to(device)
                    output_cell_['cell_info'] = cell_info # Additional value for normalization
            else:
                output_cell = None
            model_output = model(resized_patch, output_cell)

        for bs in range(len(model_output)):
            # Extract bboxes, scores, and labels
            scores = model_output[bs]['scores'].cpu()
            scores_mask = scores > cluster_threshold
            bboxes = model_output[bs]['boxes'][scores_mask].cpu()
            scores_filtered = scores[scores_mask]
            labels_filtered = model_output[bs]['labels'][scores_mask].cpu()
            masks_filtered = model_output[bs]['masks'][scores_mask].cpu()
            if model_cell != None:
                cells_filtered = model_output[bs]['cells'][scores_mask].cpu()
                # Collect results for this patch
                patches_results[bs].append((bboxes, scores_filtered, labels_filtered, masks_filtered, cells_filtered))
            else:
                # Collect results for this patch
                patches_results[bs].append((bboxes, scores_filtered, labels_filtered, masks_filtered))

        # Clear GPU memory for the current patch
        del resized_patch, model_output  # Free temporary variables
        torch.cuda.empty_cache()  # Explicitly clear GPU memory
    if model_cell != None:
        # Step 3: Combine results
        combined_model_output_cell = combine_patches(
            patches_results_cell, coords, cluster_crop_size, input_tensor.shape[-1], iou_threshold=cluster_iou_threshold
        )
    else:
        combined_model_output_cell = None
    # Step 3: Combine results
    combined_model_output = combine_patches(
        patches_results, coords, cluster_crop_size, input_tensor.shape[-1], iou_threshold=cluster_iou_threshold
    )
    return combined_model_output, combined_model_output_cell

# if __name__ == "__main__":
def analysis(input, target, pretrained=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,       default='analysis') 
    parser.add_argument('--target',     type=str,       default='cluster') 
    parser.add_argument('--dataset',    type=str,       default='./dataset') 
    parser.add_argument('--gt',         type=str,       default=None) 
    parser.add_argument('--result_dir', type=str,       default='test') 
    parser.add_argument('--input_size', type=int,       default=2048) 
    parser.add_argument('--cluster_crop_size',  type=int,       default=2048) 
    parser.add_argument('--cluster_seg_size',  type=int,       default=2048) 
    parser.add_argument('--crop_size',  type=int,       default=256) 
    parser.add_argument('--seed',       type=int,       default=0) 
    parser.add_argument('--pretrained', type=int,       default=1) 
    parser.add_argument('--bs',         type=int,       default=1) 
    parser.add_argument('--num_workers',type=int,       default=1) 
    parser.add_argument('--cluster_threshold',  type=float,     default=0.05) #test 
    parser.add_argument('--cluster_iou_threshold',  type=float,     default=0.2) #test 
    parser.add_argument('--cell_threshold',  type=float,     default=0.5) #test 
    parser.add_argument('--cell_iou_threshold',  type=float,     default=0.2) #test 
    parser.add_argument('--test_samples', type=int,       default=100000) 
    parser.add_argument('--epochs', type=int,       default=0) 
    args = parser.parse_args([])

    args.dataset = input
    if pretrained:
        args.pretrained = 1
    else:
        args.pretrained = 0

    result_dir = target
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f'{result_dir}/cluster', exist_ok=True)

    if args.target == 'cluster' and args.pretrained:
        if args.epochs == 0:
            checkpoint = f'checkpoints/{args.project}/cluster_wcell.pth'
        else:
            checkpoint = f'checkpoints/{args.project}/cluster_wcell_e{args.epochs:04d}.pth'
        os.makedirs(f'{result_dir}/cell', exist_ok=True)
    else:
        if args.epochs == 0:
            checkpoint = f'checkpoints/{args.project}/cluster.pth'
        else:
            checkpoint = f'checkpoints/{args.project}/cluster_e{args.epochs:04d}.pth'
    # Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
    set_seed(args.seed)
        
    if args.pretrained and args.target == 'cluster':
        pretrained_path = f'checkpoints/{args.project}/cell.pth'
        pretrained_feature_path = f'checkpoints/{args.project}/cell.json'
    else:
        pretrained_path = None
        pretrained_feature_path = None
    device = get_torch_device()
    dtype = torch.float32
    device, dtype
    # Get the class names for the predicted label indices
    class_names = ['background', 'immune', 'tumor']

    # Set the name of the font file
    font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'
    # Download the font file
    download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
    draw_bboxes = partial(draw_bounding_boxes, fill=False, width=1, font=font_file, font_size=10)

    model = model_initialize(num_classes=len(class_names), device=device, dtype=dtype)
    model.load_state_dict(torch.load(checkpoint), strict=False)
    model.device = device
    if args.target == 'cluster' and args.pretrained:
        model.name = f'{args.target}_wcell'
    else:
        model.name = args.target
    # Set the model to evaluation mode
    model.eval();

    # Load pretrained Cell Detection Model
    if pretrained_path != None:
        model_cell = model_initialize(num_classes=2, device=device, dtype=dtype)
        model_cell.load_state_dict(torch.load(pretrained_path), strict=False)
        # Freeze the parameters to prevent updates
        for param in model_cell.parameters():
            param.requires_grad = False
        model_cell.eval()
    else:
        model_cell=None

    if model_cell != None:
        if Path(pretrained_feature_path).exists():
            print(f"Loading data from {pretrained_feature_path}...")
            with open(pretrained_feature_path, 'r') as f:
                cell_info = json.load(f)
        else:
            err
    else:
        cell_info = None
    
    test_dataset, class_names = CustomDataset_test(args.dataset, args.gt, test_sz=args.input_size)
    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': args.bs,  # Batch size for data loading
        'num_workers': args.num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,  # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': lambda batch: tuple(zip(*batch)),
    }
    test_dataloader = DataLoader(test_dataset, **data_loader_params, shuffle=False)
    print(f'Number of batches in test DataLoader:{len(test_dataloader)}')

    # 엑셀로 저장할 컬럼을 준비합니다.
    metrics = ['name', 'mean_iou', 'mean_dice', 'class1_precision', 'class1_recall', 'class1_f1_score', 'class1_tp', 'class1_fp', 'class1_fn', 'class2_precision', 'class2_recall', 'class2_f1_score', 'class2_tp', 'class2_fp', 'class2_fn']
    result_rows = []
    result_rows_patchinfo = []
    result_rows_patchinfo_nooverlap = []
    for batch_id, (inputs, targets, name) in enumerate(test_dataloader):
        inputs = torch.stack(inputs).to(device)

        print(f'Processing {name[0]}')
        # Open the test file
        # Make a prediction with the model
        with torch.no_grad():
            combined_outputs, combined_outputs_cell = cluster_prediction(model, model_cell, inputs, args.cluster_crop_size, args.cluster_seg_size, args.crop_size, cluster_threshold=args.cluster_threshold, cluster_iou_threshold=args.cluster_iou_threshold, cell_threshold=args.cell_threshold, cell_iou_threshold=args.cell_iou_threshold, cell_info=cell_info, device=device)
        if targets[0] != None:
            # Evaluation
            result = evaluate_maskrcnn(combined_outputs, targets, args.cell_threshold)
            result_rows.append([name[0], 
                result['mean_iou'],
                result['mean_dice'],
                result['precision'][0],  # precision은 리스트 형태이므로 첫 번째 값을 사용
                result['recall'][0],      # recall도 마찬가지
                result['f1_score'][0],
                result['tp_list'][0],
                result['fp_list'][0],
                result['fn_list'][0],
                result['precision'][1],  # precision은 리스트 형태이므로 첫 번째 값을 사용
                result['recall'][1],      # recall도 마찬가지
                result['f1_score'][1],
                result['tp_list'][1],
                result['fp_list'][1],
                result['fn_list'][1],
                ])     # f1_score도 마찬가지
            print(result)
        # Step 4: Annotate the original image
        pred_bboxes = BoundingBoxes(combined_outputs[0]['boxes'], format='xyxy', canvas_size=(args.input_size, args.input_size))
        pred_labels = [class_names[int(label)] for label in combined_outputs[0]['labels']]

        # Generate colors for annotation
        colors = get_custom_colors(len(class_names))
        int_colors = [tuple(int(c * 255) for c in color) for color in colors]
        pred_colors = [int_colors[i] for i in [class_names.index(label) for label in pred_labels]]

        # # Annotate the test image
        if len(combined_outputs[0]['masks']) == 0:
            pred_masks = combined_outputs[0]['masks'].to(torch.bool)  # 빈 리스트를 반환하거나 기본값 설정
        else:
            pred_masks = torch.stack([
                torch.where(mask >= args.cluster_threshold, 1, 0).to(torch.bool)
                for mask in combined_outputs[0]['masks']
            ])
        annotated_tensor = draw_segmentation_masks(image=inputs[0], masks=pred_masks, alpha=0.3, colors=pred_colors)
    
        # Annotate the test image with the predicted labels and bounding boxes
        annotated_tensor = draw_bboxes(
            image=annotated_tensor,
            boxes=pred_bboxes,
            # labels=[f"{label}\n{prob*100:.2f}%\nEA:{num:.2f}\nAREA:{area:.5f}" for label, prob, num, area in zip(pred_labels, combined_outputs[0]['scores'], combined_outputs[0]['cells'][:,0], combined_outputs[0]['cells'][:,1])],
            colors=pred_colors
        )
        if model_cell == None:
            cluster_area =  [mask.sum() for mask in pred_masks]
            for idx, (l, a) in enumerate(zip(pred_labels, cluster_area)):
                result_rows_patchinfo.append({
                    "Input Name": name[0],
                    "Index": idx,
                    "Label": l,
                    "Cluster Area": int(a),
                })
                if int(name[0].split('_')[1][1:])%2 == 0 and int(name[0].split('_')[2][:3])%2 == 0:
                    result_rows_patchinfo_nooverlap.append({
                        "Input Name": name[0],
                        "Index": idx,
                        "Label": l,
                        "Cluster Area": int(a),
                    })
        else:
            cluster_area =  [mask.sum() for mask in pred_masks]
            cell_ea = combined_outputs[0]['cells'][:, 0]
            cell_area = combined_outputs[0]['cells'][:, 1]
            for idx, (l, a, ce, ca) in enumerate(zip(pred_labels, cluster_area, cell_ea, cell_area)):
                result_rows_patchinfo.append({
                    "Input Name": name[0],
                    "Index": idx,
                    "Label": l,
                    "Cluster Area": int(a),
                    "Cell EA": int(ce),
                    "Cell Area": int(ca),
                })
                if int(name[0].split('_')[1][1:])%2 == 0 and int(name[0].split('_')[2][:3])%2 == 0:
                    result_rows_patchinfo_nooverlap.append({
                        "Input Name": name[0],
                        "Index": idx,
                        "Label": l,
                        "Cluster Area": int(a),
                        "Cell EA": int(ce),
                        "Cell Area": int(ca),
                    })

        # Save the image using the desired extension
        print(name[0])
        tensor_to_pil(annotated_tensor).save(f"{result_dir}/cluster/{name[0]}")

        if model_cell != None:
            class_names_cell = ['background', 'cellular']
                
            # Step 4: Annotate the original image
            pred_bboxes = BoundingBoxes(combined_outputs_cell[0]['boxes'], format='xyxy', canvas_size=(args.input_size, args.input_size))
            pred_labels = [class_names_cell[int(label)] for label in combined_outputs_cell[0]['labels']]

            # Generate colors for annotation
            colors = get_custom_colors(len(class_names_cell))
            int_colors = [tuple(int(c * 255) for c in color) for color in colors]
            pred_colors = [int_colors[i] for i in [class_names_cell.index(label) for label in pred_labels]]

            #pred_masks = torch.concat([Mask(torch.where(mask >= args.threshold, 1, 0), dtype=torch.bool) for mask in combined_outputs_cell[0]['masks']])
            pred_masks = torch.stack([
                torch.where(mask >= args.cell_threshold, 1, 0).to(torch.bool)
                for mask in combined_outputs_cell[0]['masks']
            ])
            annotated_tensor = draw_segmentation_masks(image=inputs[0], masks=pred_masks, alpha=0.3, colors=pred_colors)
            # Annotate the test image with the predicted labels and bounding boxes
            annotated_tensor = draw_bboxes(
                image=annotated_tensor,
                boxes=pred_bboxes,
                #labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, combined_outputs_cell[0]['scores'])],
                #labels=[f"{label}%" for label in pred_labels],
                colors=pred_colors
            )
            # Save the image using the desired extension
            tensor_to_pil(annotated_tensor).save(f"{result_dir}/cell/{name[0]}")

    df = pd.DataFrame(result_rows_patchinfo)
    output_path = f"{result_dir}/cluster_patches.xlsx"
    df.to_excel(output_path, index=False)

    df = pd.DataFrame(result_rows_patchinfo_nooverlap)
    output_path = f"{result_dir}/cluster_patches_nooverlap.xlsx"
    df.to_excel(output_path, index=False)

    if targets[0] != None:
        df = pd.DataFrame(result_rows, columns=metrics)
        mean_vals = ['mean', 
            df['mean_iou'].mean(),
            df['mean_dice'].mean(),
            df['class1_precision'].mean(),  
            df['class1_recall'].mean(),      
            df['class1_f1_score'].mean(),
            df['class1_tp'].mean(),
            df['class1_fp'].mean(),
            df['class1_fn'].mean(),
            df['class2_precision'].mean(), 
            df['class2_recall'].mean(),     
            df['class2_f1_score'].mean(),
            df['class2_tp'].mean(),
            df['class2_fp'].mean(),
            df['class2_fn'].mean(),
            ]     # f1_score도 마찬가지
        result_rows.append(mean_vals)  
        df = pd.DataFrame(result_rows, columns=metrics)
        df.to_excel(f"{result_dir}/cluster_score.xlsx", index_label='evaluation_run')
