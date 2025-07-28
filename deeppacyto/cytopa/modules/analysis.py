from cytopa.trainer import *
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
# from cytopa.visualizer import Visualizer
from detectron2.structures import Instances
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import ColorMode
import torch
from detectron2.structures import Boxes
import torchvision.ops as ops
import os
import numpy as np
import sys
import json
from pycocotools import mask as coco_mask
import pandas as pd
from PIL import Image
from collections import defaultdict

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    args.config_file = 'cytopa/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.IN_CHANS = 3
    cfg.TEST.DETECTIONS_PER_IMAGE = args.det_per_image
    if len(args.thing_classes) >=2:
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(args.thing_classes)
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        # cfg.INPUT.IMAGE_SIZE = 2048
        # cfg.MODEL.MaskDINO.DEC_LAYERS = 6
        # cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 4
    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(args.thing_classes)

    # # TTA(Test Time Augmentation) 설정 추가
    # cfg.TEST.AUG.ENABLED = True
    # cfg.TEST.AUG.FLIP = True

    cfg.DATASETS.TRAIN = ("cell_train",)
    cfg.DATASETS.TEST = ('cell_test',)
    cfg.OUTPUT_DIR = f'{args.output_dir}'
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.MaskDINO.PRETRAINED_MODEL = args.pretrained_model
    cfg.MODEL.MaskDINO.PRETRAINED_WEIGHT = args.pretrained_weight
    cfg.MODEL.MaskDINO.PRETRAINED_MODE = args.pretrained_mode
    # cfg.INPUT.MIN_SIZE_TEST = 2048  # 테스트 이미지의 최소 크기
    # cfg.INPUT.MAX_SIZE_TEST = 2048  # 테스트 이미지의 최대 크기
    # cfg.SOLVER.IMS_PER_BATCH = 1  # 메모리 문제가 발생할 경우 배치 크기 조정
    # cfg.MODEL.FPN.FPN_ON = True  # 큰 이미지를 처리하기 위한 FPN 설정
    # cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False  # 큰 이미지를 처리하기 위한 FPN 설정
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg

def sliding_window_crop(image, crop_size=512, overlap=0.5):
    """이미지를 겹치는 패치로 분할"""
    h, w = image.shape[:2]
    step = int(crop_size * (1 - overlap))
    patches = []
    coords = []

    for top in range(0, h - crop_size + 1, step):
        for left in range(0, w - crop_size + 1, step):
            patch = image[top:top + crop_size, left:left + crop_size]
            patches.append(patch)
            coords.append((left, top))

    # 오른쪽과 아래쪽 가장자리 처리
    if (h - crop_size) % step != 0:
        for left in range(0, w - crop_size + 1, step):
            patch = image[h - crop_size:h, left:left + crop_size]
            patches.append(patch)
            coords.append((left, h - crop_size))
    if (w - crop_size) % step != 0:
        for top in range(0, h - crop_size + 1, step):
            patch = image[top:top + crop_size, w - crop_size:w]
            patches.append(patch)
            coords.append((w - crop_size, top))

    return patches, coords

def combine_predictions(predictions_list, coords, original_size, crop_size=512):
    """패치별 예측을 하나의 마스크로 통합하며 객체 정보 보존"""
    h, w = original_size
    combined_instances = {
        'pred_boxes': [],
        'scores': [],
        'pred_classes': [],
        'pred_masks': []
    }
    
    for pred_instances, (left, top) in zip(predictions_list, coords):
        # 예측된 객체가 없는 경우 건너뛰기
        if len(pred_instances) == 0:
            continue
            
        # bbox 좌표 조정
        boxes = pred_instances.pred_boxes.tensor
        shifted_boxes = boxes.clone()
        shifted_boxes[:, [0, 2]] += left  # x 좌표 조정
        shifted_boxes[:, [1, 3]] += top   # y 좌표 조정
        
        # 마스크 조정
        masks = pred_instances.pred_masks
        shifted_masks = torch.zeros((masks.shape[0], h, w), device=masks.device)
        shifted_masks[:, top:top + crop_size, left:left + crop_size] = masks
        
        # 정보 저장
        combined_instances['pred_boxes'].append(shifted_boxes)
        combined_instances['scores'].append(pred_instances.scores)
        combined_instances['pred_classes'].append(pred_instances.pred_classes)
        combined_instances['pred_masks'].append(shifted_masks)
    
    # 예측된 객체가 전혀 없는 경우 빈 인스턴스 반환
    if not combined_instances['pred_boxes']:
        empty_instances = Instances(image_size=(h, w))
        empty_instances.pred_boxes = Boxes(torch.zeros((0, 4)))
        empty_instances.scores = torch.zeros(0)
        empty_instances.pred_classes = torch.zeros(0, dtype=torch.long)
        empty_instances.pred_masks = torch.zeros((0, h, w))
        return empty_instances
    
    # 모든 패치의 결과 합치기
    combined_instances['pred_boxes'] = torch.cat(combined_instances['pred_boxes'], dim=0)
    combined_instances['scores'] = torch.cat(combined_instances['scores'], dim=0)
    combined_instances['pred_classes'] = torch.cat(combined_instances['pred_classes'], dim=0)
    combined_instances['pred_masks'] = torch.cat(combined_instances['pred_masks'], dim=0)
    
    # NMS 적용
    keep_indices = ops.nms(
        combined_instances['pred_boxes'],
        combined_instances['scores'],
        iou_threshold=0.1
    )
    
    # NMS 결과로 필터링
    for key in combined_instances:
        combined_instances[key] = combined_instances[key][keep_indices]
    
    # Instances 객체 생성
    final_instances = Instances(image_size=(h, w))
    final_instances.pred_boxes = Boxes(combined_instances['pred_boxes'])
    final_instances.scores = combined_instances['scores']
    final_instances.pred_classes = combined_instances['pred_classes']
    final_instances.pred_masks = combined_instances['pred_masks']
    
    return final_instances

def evaluate_predictions_fgonly(predictions, annotations, metadata):
    """예측 결과를 평가하는 함수"""
    from detectron2.evaluation import COCOEvaluator
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    evaluator = COCOEvaluator(
        "cell_test",
        tasks=("bbox", "segm"),
        output_dir="./eval_results"
    )
    evaluator.reset()
    
    # # 데이터 형식 수정
    # for pred, anno in zip(predictions, annotations):
    #     # 입력 형식에 맞게 데이터 구성
    #     evaluator.process(
    #         [{"image_id": anno['image_id']}],  # inputs
    #         [pred]  # predictions
    #     )
    
    
    for pred, anno in zip(predictions, annotations):
        # 원본 예측 결과 복사
        filtered_pred = copy.deepcopy(pred)
        instances = filtered_pred["instances"]
        
        # 라벨링된 영역 마스크 생성 (모든 GT 마스크의 합집합)
        h, w = instances.image_size
        labeled_region = torch.zeros((h, w), dtype=torch.bool)
        
        
        # 라벨링된 영역 마스크 생성 (모든 GT 마스크의 합집합)
        labeled_region = torch.zeros_like(instances.pred_masks[0], dtype=torch.bool)
        for ann in anno['annotations']:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    # RLE 형식
                    gt_mask = torch.tensor(coco_mask.decode(ann['segmentation']).astype(bool))
                else:
                    # 폴리곤 형식
                    rles = coco_mask.frPyObjects(ann['segmentation'], h, w)
                    rle = coco_mask.merge(rles)
                    gt_mask = torch.tensor(coco_mask.decode(rle).astype(bool))
                labeled_region |= gt_mask
        
      
        # 각 예측 마스크와 라벨링된 영역의 교집합 비율 계산
        keep_indices = []
        for idx, mask in enumerate(instances.pred_masks):
            # float 마스크를 bool로 변환
            bool_mask = mask > 0.5  # threshold를 0.5로 설정
            intersection = (bool_mask & labeled_region).sum()
            mask_area = bool_mask.sum()
            if mask_area > 0:
                overlap_ratio = intersection.float() / mask_area
                # 예측의 50% 이상이 라벨링된 영역과 겹치는 경우만 유지
                if overlap_ratio >= 0.5:
                    keep_indices.append(idx)
        
        
        # 필터링된 인스턴스 생성
        keep_indices = torch.tensor(keep_indices)
        if len(keep_indices) > 0:
            new_instances = Instances(instances.image_size)
            new_instances.pred_boxes = instances.pred_boxes[keep_indices]
            new_instances.scores = instances.scores[keep_indices]
            new_instances.pred_classes = instances.pred_classes[keep_indices]
            new_instances.pred_masks = instances.pred_masks[keep_indices]
            filtered_pred["instances"] = new_instances
        else:
            # 유지할 예측이 없는 경우 빈 인스턴스 생성
            new_instances = Instances(instances.image_size)
            new_instances.pred_boxes = instances.pred_boxes[:0]
            new_instances.scores = instances.scores[:0]
            new_instances.pred_classes = instances.pred_classes[:0]
            new_instances.pred_masks = instances.pred_masks[:0]
            filtered_pred["instances"] = new_instances
        
        # 평가 수행
        evaluator.process(
            [{"image_id": anno['image_id']}],  # inputs
            [filtered_pred]  # filtered predictions
        )
    return evaluator.evaluate()

def evaluate_predictions(predictions, annotations, metadata):
    """예측 결과를 평가하는 함수"""
    from detectron2.evaluation import COCOEvaluator
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    evaluator = COCOEvaluator(
        "cell_test",
        tasks=("bbox", "segm"),
        output_dir="./eval_results"
    )
    evaluator.reset()
    
    # 데이터 형식 수정
    for pred, anno in zip(predictions, annotations):
        # 입력 형식에 맞게 데이터 구성
        evaluator.process(
            [{"image_id": anno['image_id']}],  # inputs
            [pred]  # predictions
        )
    
    return evaluator.evaluate()

def apply_nms(instances, iou_threshold=0.1):
    """인스턴스에 NMS를 적용하는 헬퍼 함수"""
    keep_indices = ops.nms(
        instances.pred_boxes.tensor,
        instances.scores,
        iou_threshold=iou_threshold
    )
    
    return instances[keep_indices]

def apply_augmentation(image, aug_type):
    """이미지에 augmentation 적용"""
    if aug_type == "original":
        return image
    elif aug_type == "rotate90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_type == "hflip":
        return cv2.flip(image, 1)  # 1은 수평 플립
    elif aug_type == "vflip":
        return cv2.flip(image, 0)  # 0은 수직 플립

def reverse_augmentation(instances, aug_type, image_size):
    """예측 결과의 augmentation 되돌리기"""
    h, w = image_size
    if aug_type == "original":
        return instances
    
    # 마스크 변환
    masks = instances.pred_masks
    boxes = instances.pred_boxes.tensor
    
    if aug_type == "rotate90":
        # 마스크 회전 되돌리기
        masks = torch.rot90(masks, k=-1, dims=(-2, -1))
        # 박스 좌표 변환
        new_boxes = boxes.clone()
        new_boxes[:, 0] = h - boxes[:, 3]  # x1
        new_boxes[:, 1] = boxes[:, 0]      # y1
        new_boxes[:, 2] = h - boxes[:, 1]  # x2
        new_boxes[:, 3] = boxes[:, 2]      # y2
        boxes = new_boxes
        
    elif aug_type == "hflip":
        # 마스크 수평 플립
        masks = torch.flip(masks, [-1])
        # 박스 좌표 변환
        new_boxes = boxes.clone()
        new_boxes[:, 0] = w - boxes[:, 2]  # x1
        new_boxes[:, 2] = w - boxes[:, 0]  # x2
        boxes = new_boxes
        
    elif aug_type == "vflip":
        # 마스크 수직 플립
        masks = torch.flip(masks, [-2])
        # 박스 좌표 변환
        new_boxes = boxes.clone()
        new_boxes[:, 1] = h - boxes[:, 3]  # y1
        new_boxes[:, 3] = h - boxes[:, 1]  # y2
        boxes = new_boxes
    
    # 변환된 결과로 새로운 인스턴스 생성
    new_instances = Instances(image_size=(h, w))
    new_instances.pred_boxes = Boxes(boxes)
    new_instances.scores = instances.scores
    new_instances.pred_classes = instances.pred_classes
    new_instances.pred_masks = masks
    
    return new_instances

def process_single_image(predictor, image, crop_size, conf_threshold, forced_lr=False, aug_types=None):
    """단일 이미지에 대한 모든 augmentation 처리 및 결과 통합"""
    original_h, original_w = image.shape[:2]
    h, w = original_h, original_w
    all_instances = []
    
    if forced_lr:
        # 이미지 크기를 1/4로 축소
        h, w = h // 4, w // 4
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        if crop_size > 0:
            crop_size = crop_size // 4
    
    for aug_type in aug_types:
        aug_image = apply_augmentation(image, aug_type)
        aug_h, aug_w = aug_image.shape[:2]
        
        if crop_size > 0:
            patches, coords = sliding_window_crop(aug_image, crop_size=crop_size, overlap=0)
            patch_predictions = []
            
            for patch in patches:
                outputs = predictor(patch)
                instances = outputs["instances"].to("cpu")
                confident_detections = instances[instances.scores > conf_threshold]
                patch_predictions.append(confident_detections)
            
            combined_output = combine_predictions(patch_predictions, coords, (aug_h, aug_w), crop_size=crop_size)
        else:
            outputs = predictor(aug_image)
            combined_output = outputs["instances"].to("cpu")
            combined_output = combined_output[combined_output.scores > conf_threshold]
        
        # Augmentation 되돌리기
        restored_instances = reverse_augmentation(combined_output, aug_type, (h, w))
        all_instances.append(restored_instances)
    
    # 모든 예측 결과 통합
    combined_instances = {
        'pred_boxes': [],
        'scores': [],
        'pred_classes': [],
        'pred_masks': []
    }
    
    for instances in all_instances:
        combined_instances['pred_boxes'].append(instances.pred_boxes.tensor)
        combined_instances['scores'].append(instances.scores)
        combined_instances['pred_classes'].append(instances.pred_classes)
        combined_instances['pred_masks'].append(instances.pred_masks)
    
    # 모든 결과 합치기
    combined_instances['pred_boxes'] = torch.cat(combined_instances['pred_boxes'], dim=0)
    combined_instances['scores'] = torch.cat(combined_instances['scores'], dim=0)
    combined_instances['pred_classes'] = torch.cat(combined_instances['pred_classes'], dim=0)
    combined_instances['pred_masks'] = torch.cat(combined_instances['pred_masks'], dim=0)
    
    # 최종 NMS 적용
    keep_indices = ops.nms(
        combined_instances['pred_boxes'],
        combined_instances['scores'],
        iou_threshold=0.1
    )
    
    # 최종 결과 생성
    final_instances = Instances(image_size=(h, w))
    final_instances.pred_boxes = Boxes(combined_instances['pred_boxes'][keep_indices])
    final_instances.scores = combined_instances['scores'][keep_indices]
    final_instances.pred_classes = combined_instances['pred_classes'][keep_indices]
    final_instances.pred_masks = combined_instances['pred_masks'][keep_indices]
    
    if forced_lr:
        # 마스크를 원본 크기로 확장
        resized_masks = torch.nn.functional.interpolate(
            final_instances.pred_masks.unsqueeze(1).float(),
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1) > 0.5  # 이진 마스크로 변환
        
        # 박스 좌표를 원본 크기로 확장
        scale_x = original_w / w
        scale_y = original_h / h
        scaled_boxes = final_instances.pred_boxes.tensor.clone()
        scaled_boxes[:, [0, 2]] *= scale_x  # x 좌표 스케일링
        scaled_boxes[:, [1, 3]] *= scale_y  # y 좌표 스케일링
        
        # 새로운 인스턴스 생성
        scaled_instances = Instances(image_size=(original_h, original_w))
        scaled_instances.pred_boxes = Boxes(scaled_boxes)
        scaled_instances.scores = final_instances.scores
        scaled_instances.pred_classes = final_instances.pred_classes
        scaled_instances.pred_masks = resized_masks
        
        return scaled_instances
    
    return final_instances

def save_masks_as_png(masks, classes, save_dir, base_name):
    """마스크를 PNG 형식으로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 하나의 마스크 이미지 생성 (H, W)
    combined = np.zeros(masks.shape[1:], dtype=np.uint8)
    for i, (mask, cls) in enumerate(zip(masks, classes)):
        # 각 인스턴스에 고유한 값 할당 (1부터 시작)
        combined[mask > 0] = i + 1
    
    # PNG로 저장
    combined_path = os.path.join(save_dir, f'{base_name}.png')
    cv2.imwrite(combined_path, combined)
    
    # 클래스 정보 별도 저장
    classes_path = os.path.join(save_dir, f'{base_name}_classes.npy')
    np.save(classes_path, classes)

def load_masks_from_png(save_dir, base_name):
    """PNG 형식의 마스크를 불러오기"""
    # 마스크 불러오기
    combined_path = os.path.join(save_dir, f'{base_name}.png')
    combined = cv2.imread(combined_path, cv2.IMREAD_UNCHANGED)
    
    # 각 인스턴스 마스크 분리
    unique_values = np.unique(combined)[1:]  # 0 제외
    masks = np.array([(combined == val) for val in unique_values], dtype=bool)
    
    # 클래스 정보 불러오기
    classes_path = os.path.join(save_dir, f'{base_name}_classes.npy')
    classes = np.load(classes_path)
    
    return masks, classes

def run(args):
    data_dir = args.data_dir
    # 기존 등록 해제 후 재등록
    if "cell_test" in DatasetCatalog:
        DatasetCatalog.remove("cell_test")
    DatasetCatalog.register("cell_test", lambda: np.load(data_dir, allow_pickle=True))
    
    balloon_metadata = MetadataCatalog.get("cell_test")
    if len(args.thing_classes) > 1:
        balloon_metadata.set(thing_classes=args.thing_classes,
                            thing_colors=[[255, 0, 0],[0, 0, 255]])  # 빨간색과 파란색 지정
    else:
        balloon_metadata.set(thing_classes=args.thing_classes, thing_colors=[[255, 0, 0]])
    args.resume = True

    cfg = setup(args)
    print("Command cfg:", cfg)

    predictor = DefaultPredictor(cfg)

    print(data_dir)
    ds_dict = np.load(data_dir, allow_pickle=True)
    rst = []
    predictions = []  # 평가를 위한 예측 결과 저장
    k = 0
    cell_counts = []  # 세포 수와 평균 크기를 저장할 리스트
    for ix, d in tqdm(enumerate(ds_dict)):
        # if '7207_i007_005.png' not in d["file_name"]:
        #     continue
        # if ix > 2:  #debug
        #     break
        if k >= args.vis_samples:
            break
        im = cv2.imread(d["file_name"])
        # 새로운 process_single_image 함수 사용
        combined_output = process_single_image(predictor, im, args.crop_size, args.conf_threshold, args.lr, args.aug_types)
        
        # 파일 이름과 클래스별 세포 수 계산
        file_name = d["file_name"]
        pred_classes = combined_output.pred_classes.numpy()
        tumor_count = np.sum(pred_classes == 0)  # tumor는 클래스 0
        immune_count = np.sum(pred_classes == 1)  # immune는 클래스 1
        
        # 마스크 크기 계산
        masks = combined_output.pred_masks.numpy()
        
        # 클래스별 마스크 크기 계산
        tumor_masks = masks[pred_classes == 0] if len(pred_classes) > 0 else np.array([])
        immune_masks = masks[pred_classes == 1] if len(pred_classes) > 0 else np.array([])
        
        # 각 클래스의 평균 마스크 크기 계산 (픽셀 수)
        tumor_avg_size = np.mean([mask.sum() for mask in tumor_masks]) if len(tumor_masks) > 0 else 0
        immune_avg_size = np.mean([mask.sum() for mask in immune_masks]) if len(immune_masks) > 0 else 0
        
        cell_counts.append({
            'file_name': os.path.basename(file_name),
            'tumor_count': tumor_count,
            'immune_count': immune_count,
            'tumor_avg_size': tumor_avg_size,
            'immune_avg_size': immune_avg_size
        })
        
        result_dict = {
            'pred_boxes': combined_output.pred_boxes.tensor.numpy(),
            'file_name': d['file_name'],
        }
        
        # 평가를 위한 예측 결과 저장
        if 'annotations' in d:  # annotation이 있는 경우에만
            prediction = {
                'image_id': d.get('image_id', k),
                'instances': combined_output
            }
            predictions.append(prediction)
        
        rst.append(result_dict)
        # 시각화 부분은 그대로 유지
        if k < args.vis_samples:
            v = Visualizer(im[:, :, ::-1],
                        metadata=balloon_metadata, 
                        scale=1 if args.crop_size == 512 else 4,
                        instance_mode=ColorMode.SEGMENTATION if len(args.thing_classes) > 1 else ColorMode.IMAGE_BW
            )
            combined_output.remove('scores')
            if len(args.thing_classes) == 1:
                combined_output.remove('pred_classes')
            out = v.draw_instance_predictions(combined_output)
            # 원본 파일 이름에서 확장자를 제외한 기본 이름 추출
            base_name = os.path.splitext(os.path.basename(d['file_name']))[0]
            os.makedirs(os.path.join(args.output_dir, 'img'), exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, 'img', 
                       f'{base_name}.png'), out.get_image()[:, :, ::-1])
            k += 1
        
        # 마스크를 PNG로 저장
        base_name = os.path.splitext(os.path.basename(d["file_name"]))[0]
        save_masks_as_png(
            combined_output.pred_masks.numpy(),
            combined_output.pred_classes.numpy(),
            os.path.join(args.output_dir, 'cluster_results'),
            f'{base_name}_cluster'
        )
    
    # 세포 수와 평균 크기 데이터를 DataFrame으로 변환
    df = pd.DataFrame(cell_counts)
    
    # 클래스 개수에 따라 평균 계산 방식 변경
    # print(df)
    if len(args.thing_classes) > 1:
        avg_row = pd.DataFrame([{
            'file_name': 'Average',
            'tumor_count': df['tumor_count'].mean(),
            'immune_count': df['immune_count'].mean(),
            'tumor_avg_size': df['tumor_avg_size'].mean(),
            'immune_avg_size': df['immune_avg_size'].mean()
        }])
    else:
        # 단일 클래스인 경우
        avg_row = pd.DataFrame([{
            'file_name': 'Average',
            'cell_count': df['cell_count'].mean() if 'cell_count' in df.columns else 0,
            'avg_size': df['avg_size'].mean() if 'avg_size' in df.columns else 0
        }])
    
    df = pd.concat([df, avg_row], ignore_index=True)
    
    # CSV 파일로 저장
    csv_output_path = os.path.join(args.output_dir,
                                 'cell_counts.csv')
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"세포 수 데이터가 저장되었습니다: {csv_output_path}")

def run_cell(args):
    data_dir = args.data_dir
    # 기존 등록 해제 후 재등록
    if "cell_test" in DatasetCatalog:
        DatasetCatalog.remove("cell_test")
        # MetadataCatalog.remove("cell_test")  # 메타데이터도 함께 제거
    DatasetCatalog.register("cell_test", lambda: np.load(data_dir, allow_pickle=True))
    
    balloon_metadata = MetadataCatalog.get("cell_test")
    # 메타데이터가 이미 설정되어 있는지 확인
    if not hasattr(balloon_metadata, "thing_classes"):
        if len(args.thing_classes) > 1:
            balloon_metadata.set(thing_classes=args.thing_classes,
                                thing_colors=[[255, 0, 0],[0, 0, 255]])  # 빨간색과 파란색 지정
        else:
            balloon_metadata.set(thing_classes=args.thing_classes,
                                thing_colors=[[255, 0, 0]])
    args.resume = True

    cfg = setup(args)
    print("Command cfg:", cfg)

    predictor = DefaultPredictor(cfg)

    print(data_dir)
    ds_dict = np.load(data_dir, allow_pickle=True)
    rst = []
    predictions = []  # 평가를 위한 예측 결과 저장
    k = 0
    cell_counts = []  # 세포 수와 평균 크기를 저장할 리스트
    for ix, d in tqdm(enumerate(ds_dict)):  
        # if '7207_i007_005.png' not in d["file_name"]:
        #     continue
        # if ix > 2:  #debug
        #     break
        im = cv2.imread(d["file_name"])
        # 새로운 process_single_image 함수 사용
        combined_output = process_single_image(predictor, im, args.crop_size, args.conf_threshold, args.lr, args.aug_types)
        
        # 시각화 부분은 그대로 유지
        if k < args.vis_samples:
            v = Visualizer(im[:, :, ::-1],
                        metadata=balloon_metadata, 
                        scale=1 if args.crop_size == 512 else 4,
                        instance_mode=ColorMode.SEGMENTATION if len(args.thing_classes) > 1 else ColorMode.IMAGE_BW
            )
            if len(args.thing_classes) == 1:
                combined_output.remove('scores')
                combined_output.remove('pred_classes')
            out = v.draw_instance_predictions(combined_output)
            # 원본 파일 이름에서 확장자를 제외한 기본 이름 추출
            base_name = os.path.splitext(os.path.basename(d['file_name']))[0]
            os.makedirs(os.path.join(args.output_dir, 'img_cell'), exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, 'img_cell', 
                       f'{base_name}.png'), out.get_image()[:, :, ::-1])
            k += 1
        
        # 마스크를 PNG로 저장
        base_name = os.path.splitext(os.path.basename(d["file_name"]))[0]
        save_masks_as_png(
            combined_output.pred_masks.numpy(),
            np.zeros(len(combined_output.pred_masks)),  # cell의 경우 단일 클래스
            os.path.join(args.output_dir, 'cell_results'),
            f'{base_name}_cell'
        )

def create_test_dataset_dicts(image_dir):
    """테스트 이미지만을 위한 dataset_dicts 생성"""
    dataset_dicts = []
    
    # 이미지 파일들을 순회
    for img_file in os.listdir(image_dir):
        if not img_file.endswith('.png'):
            continue
            
        # 이미지 경로
        image_path = os.path.join(image_dir, img_file)
            
        # 이미지 크기 얻기
        img = Image.open(image_path)
        width, height = img.size
        
        record = {
            "file_name": image_path,
            "image_id": len(dataset_dicts),
            "height": height,
            "width": width,
            "annotations": []  # 테스트용이므로 어노테이션은 비어있음
        }
        
        dataset_dicts.append(record)
    
    return dataset_dicts

def run_test_only(base_dir):
    """테스트 데이터셋만 생성"""
    # 데이터셋 생성
    test_dataset_dicts = create_test_dataset_dicts(base_dir)
    
    # 데이터셋 저장
    np.save(f"{base_dir}/processed_dataset.npy", test_dataset_dicts)
    
    print(f"Saved {len(test_dataset_dicts)} records to test dataset at input/processed_dataset.npy")

def analyze_detections(args, min_seg_size=10):
    """
    cluster와 cell detection 결과를 분석하여 엑셀로 저장하는 함수
    
    Args:
        args: 설정 인자
        min_seg_size: 최소 segmentation 크기 (픽셀 단위)
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    
    # 결과를 저장할 데이터프레임을 위한 리스트
    results_data = []
    
    # 데이터셋 불러오기
    ds_dict = np.load(args.data_dir, allow_pickle=True)
    
    for d in tqdm(ds_dict, desc="분석 중"):
        file_name = os.path.basename(d["file_name"])
        base_name = os.path.splitext(file_name)[0]
        
        # cluster와 cell 마스크 로드
        try:
            cluster_masks, cluster_classes = load_masks_from_png(
                os.path.join(args.output_dir, 'cluster_results'),
                f'{base_name}_cluster'
            )
            cell_masks, _ = load_masks_from_png(
                os.path.join(args.output_dir, 'cell_results'),
                f'{base_name}_cell'
            )
        except Exception as e:
            print(f"Warning: {file_name}의 결과를 찾을 수 없습니다. 에러: {e}")
            continue
        
        # 각 cluster instance 분석
        for instance_id, (cluster_mask, cls) in enumerate(zip(cluster_masks, cluster_classes), 1):
            # cluster instance 크기 계산
            cluster_size = cluster_mask.sum()
            
            # 최소 크기 필터링
            if cluster_size < min_seg_size:
                continue
            
            # cluster와 겹치는 cell 분석
            included_cells = 0
            total_cell_area = 0
            
            for cell_mask in cell_masks:
                # cell과 cluster의 겹침 영역 계산
                overlap = (cluster_mask & cell_mask).sum()
                cell_size = cell_mask.sum()
                
                # cell이 cluster와 50% 이상 겹치는 경우
                if overlap / cell_size >= 0.5:
                    included_cells += 1
                    total_cell_area += cell_size
            
            # 결과 저장
            instance_info = {
                'file_name': file_name,
                'instance_id': instance_id,
                'cluster_type': 'tumor' if cls == 0 else 'immune',
                'cluster_size': float(cluster_size),  # 1) cluster instance 크기
                'cell_count': included_cells,         # 2) 포함된 cell 수
                'avg_cell_size': float(total_cell_area / included_cells) if included_cells > 0 else 0  # 3) 평균 cell 크기
            }
            
            results_data.append(instance_info)
    
    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(results_data)
    excel_path = os.path.join(args.output_dir, 'instance_analysis.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"분석 결과가 저장되었습니다: {excel_path}")
    
def analysis(input, target, pretrained=False, aug_types=["original"]):
    # Preprocess
    targert_name = '241120_XCUT2_1024t1024_sal01_best'
    image_dir = "dataset/241120_XCUT2_1024t1024_sal01_best"  # 테스트 이미지가 있는 폴더 경로
    run_test_only(input)

    
    class Args:
        def __init__(self):
            self.EVAL_FLAG = 1
            self.data_dir = f'{input}/processed_dataset.npy'
            self.output_dir = target
            self.det_per_image = 1000
            self.device = 'cuda'
            self.conf_threshold = 0.1
            self.vis_samples = 100000
            self.thing_classes = ['tumor', 'immune']
            self.pretrained_model = None
            self.pretrained_weight = 0.5
            self.pretrained_mode = None
            self.crop_size = 0
            self.lr = False
            self.opts = []
            self.resume = True
            self.config_file = './cytopa/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
            self.aug_types = aug_types
    args = Args()
    
    # crop_size 설정
    if 'SR' in input or 'HR' in input:
        args.crop_size = 1024
    else:
        args.crop_size = 256
        
    # pretrained 모델 설정
    if pretrained:
        args.model_weights = 'checkpoints/analysis.pth'
        args.pretrained_model = 'checkpoints/analysis.pth'
        args.pretrained_weight = 0.5
        args.pretrained_mode = 'weight'
    else:
        args.model_weights = 'checkpoints/analysis.pth'

    run(args)
    

    if pretrained:
        # crop_size 설정
        if 'SR' in input or 'HR' in input:
            args.crop_size = 512
        else:
            args.crop_size = 128
        args.model_weights = 'checkpoints/analysis.pth'
        args.pretrained_model = None
        args.pretrained_mode = None
        args.thing_classes = ['nucleus']
        args.aug_types = ['original']
        run_cell(args)
        
    # 결과 분석 실행
    analyze_detections(args)
