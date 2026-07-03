import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS, TRANSFORMS

import mmdet.detseg_utils.datasets  # noqa: F401,E402
import mmdet.detseg_utils.models  # noqa: F401,E402
from mmdet.detseg_utils.datasets import RoadAnomalyDataset  # noqa: E402


@TRANSFORMS.register_module(force=True)
class EnsureCustomEntities(BaseTransform):
    def transform(self, results):
        results['custom_entities'] = True
        return results


@DATASETS.register_module(force=True)
class RoadAnomalyJPGDataset(RoadAnomalyDataset):
    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        stem = os.path.splitext(image_name)[0]
        data_info = {
            'img_path': os.path.join(self.data_root, 'frames', image_name),
            'reduce_zero_label': False,
            'seg_map_path': os.path.join(
                self.data_root, 'frames', f'{stem}.labels',
                'labels_semantic.png'),
            'seg_fields': [],
            'text': self.metainfo['classes'],
        }
        return self.pipeline(data_info)


DATASETS_CFG = {
    'fs_lostfound': dict(
        name='FS_LostFound',
        type='FSLostAndFoundDataset',
        data_root='data/FS_LostFound',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'road_anomaly': dict(
        name='RoadAnomaly',
        type='RoadAnomalyDataset',
        data_root='data/RoadAnomaly'),
    'road_anomaly_jpg': dict(
        name='RoadAnomaly_jpg',
        type='RoadAnomalyJPGDataset',
        data_root='data/RoadAnomaly_jpg_official'),
    'smiyc_anomaly': dict(
        name='SMIYC-AnomalyTrack',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_RoadAnomalyTrack',
        img_suffix='.jpg',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'smiyc_obstacle': dict(
        name='SMIYC-ObstacleTrack',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_ObstacleTrack',
        img_suffix='.webp',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
}


def build_dataset(dataset_key, root):
    spec = DATASETS_CFG[dataset_key].copy()
    data_root = os.path.join(root, spec.pop('data_root'))
    spec.pop('name')
    pipeline = [
        dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
        dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=False, with_seg=True),
        dict(type='EnsureCustomEntities'),
        dict(type='ConcatPrompt'),
        dict(
            type='PackDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'scale_factor', 'text', 'custom_entities',
                'tokens_positive')),
    ]
    cfg = dict(data_root=data_root, pipeline=pipeline, test_mode=True)
    cfg.update(spec)
    return DATASETS.build(cfg)


def build_model(config, checkpoint, device):
    cfg = Config.fromfile(config)
    init_default_scope('mmdet')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu', strict=False)
    model.to(device)
    model.eval()
    return model


def bbox_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def component_coverage(mask, boxes):
    if len(boxes) == 0:
        return 0.0
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    height, width = mask.shape
    best = 0.0
    for x1, y1, x2, y2 in boxes.round().astype(np.int64):
        x1 = int(np.clip(x1, 0, width))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height))
        y2 = int(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        best = max(best, float(mask[y1:y2, x1:x2].sum()) / area)
    return best


def anomaly_mask(gt_map):
    gt = gt_map.astype(np.int64)
    return ((gt == 1) | (gt == 2)).astype(np.uint8)


def evaluate_one(model, dataset_key, args):
    dataset = build_dataset(dataset_key, args.root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate)

    images = 0
    images_with_anomaly = 0
    total_components = 0
    total_anomaly_pixels = 0.0
    weighted_coverage = 0.0
    recalls_iou = {0.1: 0, 0.3: 0, 0.5: 0}
    recalls_cov = {0.5: 0, 0.8: 0}
    best_scores = []
    pred_box_counts = []

    for batch in loader:
        with torch.no_grad():
            outputs = model.test_step(batch)

        for data_sample in outputs:
            images += 1
            gt_map = data_sample.gt_sem_seg.sem_seg.squeeze(0).cpu().numpy()
            mask = anomaly_mask(gt_map)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)

            pred_instances = data_sample.pred_instances
            boxes = pred_instances.bboxes.detach().cpu().numpy()
            scores = pred_instances.scores.detach().cpu().numpy()
            pred_box_counts.append(float(len(boxes)))

            sample_components = 0
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < args.min_area:
                    continue
                sample_components += 1
                total_components += 1
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                comp_box = np.array([x, y, x + w, y + h], dtype=np.float32)
                ious = bbox_iou(comp_box, boxes)
                best_iou = float(ious.max()) if len(ious) else 0.0
                for thr in recalls_iou:
                    recalls_iou[thr] += int(best_iou >= thr)

                comp_mask = labels == label_id
                coverage = component_coverage(comp_mask, boxes)
                for thr in recalls_cov:
                    recalls_cov[thr] += int(coverage >= thr)
                total_anomaly_pixels += area
                weighted_coverage += area * coverage

                if len(scores) and len(ious):
                    best_scores.append(float(scores[int(ious.argmax())]))

            if sample_components:
                images_with_anomaly += 1

        print(
            f'{DATASETS_CFG[dataset_key]["name"]}: processed '
            f'{images}/{len(dataset)}',
            flush=True)
        if args.max_samples and images >= args.max_samples:
            break

    denom = max(total_components, 1)
    return OrderedDict(
        dataset=DATASETS_CFG[dataset_key]['name'],
        images=images,
        images_with_anomaly=images_with_anomaly,
        components=total_components,
        pred_boxes_per_image=round(float(np.mean(pred_box_counts)), 2)
        if pred_box_counts else 0.0,
        recall_iou_0_1=round(recalls_iou[0.1] / denom * 100, 2),
        recall_iou_0_3=round(recalls_iou[0.3] / denom * 100, 2),
        recall_iou_0_5=round(recalls_iou[0.5] / denom * 100, 2),
        recall_cov_0_5=round(recalls_cov[0.5] / denom * 100, 2),
        recall_cov_0_8=round(recalls_cov[0.8] / denom * 100, 2),
        pixel_coverage=round(weighted_coverage / max(total_anomaly_pixels, 1.0) * 100, 2),
        mean_best_box_score=round(float(np.mean(best_scores)), 4)
        if best_scores else 0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument(
        '--config',
        default='configs/detseg/detseg_swin-b_coco.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS_CFG))
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--min-area', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default='work_dirs/box_recall/results.json')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(args.root)
    model = build_model(args.config, args.checkpoint, args.device)
    results = []
    for dataset_key in args.datasets:
        if dataset_key not in DATASETS_CFG:
            raise KeyError(f'Unknown dataset: {dataset_key}')
        results.append(evaluate_one(model, dataset_key, args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
