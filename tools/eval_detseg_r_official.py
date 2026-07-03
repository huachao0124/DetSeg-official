import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms.base import BaseTransform
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from sklearn.metrics import auc, average_precision_score, roc_curve
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS, TRANSFORMS

import mmdet.detseg_utils.datasets  # noqa: F401,E402
import mmdet.detseg_utils.metrics  # noqa: F401,E402
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
    'fs_lostfound_m2a': dict(
        name='FS_LostFound/M2A',
        type='FSLostAndFoundDataset',
        data_root='data/FS_LostFound',
        score_dir='other_score_results/score_results_m2a/FS_LostFound',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'fs_lostfound_rba': dict(
        name='FS_LostFound/RbA',
        type='FSLostAndFoundDataset',
        data_root='data/FS_LostFound',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/FS_LostFound',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'fs_static_entropy': dict(
        name='FS_Static/Entropy',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rpl_static_entropy/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_logit_distance': dict(
        name='FS_Static/LogitDistance',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rpl_static_logit_distance/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_mae_features': dict(
        name='FS_Static/MAEFeatures',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rpl_static_mae_features/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_m2a': dict(
        name='FS_Static/M2A',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_m2a/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_rba': dict(
        name='FS_Static/RbA',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_rba_ood': dict(
        name='FS_Static/RbA-OoD',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rba_ood_map_coco/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_rpl_corocl': dict(
        name='FS_Static/RPL+CoroCL',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_rpl_corocl/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_uno_ade': dict(
        name='FS_Static/UNO-ADE',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_uno_ade/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'fs_static_uno_synthetic': dict(
        name='FS_Static/UNO-Synthetic',
        type='FSLostAndFoundDataset',
        data_root='data/FS_Static',
        score_dir='other_score_results/score_results_uno_synthetic/FS_Static',
        data_prefix=dict(img_path='original', seg_map_path='labels')),
    'road_anomaly_m2a': dict(
        name='RoadAnomaly/M2A',
        type='RoadAnomalyDataset',
        data_root='data/RoadAnomaly',
        score_dir='other_score_results/score_results_m2a/RoadAnomaly'),
    'road_anomaly_rba': dict(
        name='RoadAnomaly/RbA',
        type='RoadAnomalyJPGDataset',
        data_root='data/RoadAnomaly_jpg_official',
        score_dir='other_score_results/score_results_rba_swin_b_1dl_official_jpg/RoadAnomaly_jpg'),
    'smiyc_anomaly_m2a': dict(
        name='SMIYC-AnomalyTrack/M2A',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_RoadAnomalyTrack',
        score_dir='other_score_results/score_results_m2a/dataset_RoadAnomalyTrack',
        img_suffix='.jpg',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'smiyc_anomaly_rba': dict(
        name='SMIYC-AnomalyTrack/RbA',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_RoadAnomalyTrack',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/dataset_RoadAnomalyTrack',
        img_suffix='.jpg',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'smiyc_obstacle_m2a': dict(
        name='SMIYC-ObstacleTrack/M2A',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_ObstacleTrack',
        score_dir='other_score_results/score_results_m2a/dataset_ObstacleTrack',
        img_suffix='.webp',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
    'smiyc_obstacle_rba': dict(
        name='SMIYC-ObstacleTrack/RbA',
        type='SMIYCDataset',
        data_root='data/SMIYC/dataset_ObstacleTrack',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/dataset_ObstacleTrack',
        img_suffix='.webp',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')),
}


REFINE_EVALS = OrderedDict(
    fs_lostfound_rpl_corocl=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/RPL+CoroCL',
        score_dir='other_score_results/score_results_rpl_corocl/FS_LostFound'),
    road_anomaly_rpl_corocl=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/RPL+CoroCL',
        score_dir='other_score_results/score_results_rpl_corocl/RoadAnomaly'),
    smiyc_anomaly_rpl_corocl=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/RPL+CoroCL',
        score_dir='other_score_results/score_results_rpl_corocl/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_rpl_corocl=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/RPL+CoroCL',
        score_dir='other_score_results/score_results_rpl_corocl/dataset_ObstacleTrack'),
    fs_lostfound_pebal=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/PEBAL',
        score_dir='other_score_results/score_results_pebal/FS_LostFound'),
    road_anomaly_pebal=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/PEBAL',
        score_dir='other_score_results/score_results_pebal/RoadAnomaly'),
    smiyc_anomaly_pebal=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/PEBAL',
        score_dir='other_score_results/score_results_pebal/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_pebal=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/PEBAL',
        score_dir='other_score_results/score_results_pebal/dataset_ObstacleTrack'),
    fs_lostfound_rba_ood=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/RbA OoD',
        score_dir='other_score_results/score_results_rba_ood_map_coco/FS_LostFound'),
    road_anomaly_rba_ood=dict(
        source='road_anomaly_rba',
        name='RoadAnomaly/RbA OoD',
        score_dir='other_score_results/score_results_rba_ood_map_coco/RoadAnomaly_jpg'),
    smiyc_anomaly_rba_ood=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/RbA OoD',
        score_dir='other_score_results/score_results_rba_ood_map_coco/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_rba_ood=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/RbA OoD',
        score_dir='other_score_results/score_results_rba_ood_map_coco/dataset_ObstacleTrack'),
    fs_lostfound_uno_ade=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/UNO-ADE',
        score_dir='other_score_results/score_results_uno_ade/FS_LostFound'),
    road_anomaly_uno_ade=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/UNO-ADE',
        score_dir='other_score_results/score_results_uno_ade/RoadAnomaly'),
    smiyc_anomaly_uno_ade=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/UNO-ADE',
        score_dir='other_score_results/score_results_uno_ade/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_uno_ade=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/UNO-ADE',
        score_dir='other_score_results/score_results_uno_ade/dataset_ObstacleTrack'),
    fs_lostfound_uno_synthetic=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/UNO-Synthetic',
        score_dir='other_score_results/score_results_uno_synthetic/FS_LostFound'),
    road_anomaly_uno_synthetic=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/UNO-Synthetic',
        score_dir='other_score_results/score_results_uno_synthetic/RoadAnomaly'),
    smiyc_anomaly_uno_synthetic=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/UNO-Synthetic',
        score_dir='other_score_results/score_results_uno_synthetic/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_uno_synthetic=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/UNO-Synthetic',
        score_dir='other_score_results/score_results_uno_synthetic/dataset_ObstacleTrack'),
    smiyc_anomaly_uno=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/UNO',
        score_dir='other_score_results/score_results_uno/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_uno=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/UNO',
        score_dir='other_score_results/score_results_uno/dataset_ObstacleTrack'),
)


def install_eval_spec(dataset_key):
    if dataset_key in DATASETS_CFG:
        return
    spec = REFINE_EVALS[dataset_key]
    dataset_cfg = DATASETS_CFG[spec['source']].copy()
    dataset_cfg['name'] = spec['name']
    dataset_cfg['score_dir'] = spec['score_dir']
    DATASETS_CFG[dataset_key] = dataset_cfg


def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= 0.95)[0]
    if len(valid) == 0:
        return 1.0
    return float(np.min(fpr[valid]))


def compute_metrics(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return OrderedDict(
        AUPRC=round(float(average_precision_score(labels, scores)) * 100, 2),
        **{
            'FPR@95TPR': round(fpr_at_95_tpr(scores, labels) * 100, 2),
            'AUROC': round(float(auc(fpr, tpr)) * 100, 2),
        },
    )


def build_dataset(dataset_key, root):
    spec = DATASETS_CFG[dataset_key].copy()
    score_dir = os.path.join(root, spec.pop('score_dir'))
    data_root = os.path.join(root, spec.pop('data_root'))
    spec.pop('name')
    pipeline = [
        dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
        dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=False, with_seg=True),
        dict(type='EnsureCustomEntities'),
        dict(type='ConcatPrompt'),
        dict(type='GetAnomalyScoreMap', data_path=score_dir),
        dict(
            type='PackDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'anomaly_score_map', 'scale_factor', 'flip', 'flip_direction',
                'text', 'custom_entities', 'tokens_positive')),
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


def collect_valid(score_map, gt_map):
    if score_map.shape != gt_map.shape:
        score = torch.as_tensor(score_map, dtype=torch.float32)[None, None]
        score = F.interpolate(
            score, size=gt_map.shape, mode='bilinear',
            align_corners=False).squeeze().numpy()
    else:
        score = score_map

    gt = gt_map.astype(np.int64)
    valid = gt != 255
    labels = np.zeros_like(gt, dtype=np.uint8)
    labels[(gt == 1) | (gt == 2)] = 1
    return score[valid].reshape(-1), labels[valid].reshape(-1)


def evaluate_one(model, dataset_key, args, baseline_lookup):
    install_eval_spec(dataset_key)
    spec = DATASETS_CFG[dataset_key]
    score_dir = os.path.join(args.root, spec['score_dir'])
    if not os.path.isdir(score_dir):
        return OrderedDict(
            dataset=spec['name'],
            status='missing_score_dir',
            score_dir=spec['score_dir'],
        )

    dataset = build_dataset(dataset_key, args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate)

    compute_baseline = spec['name'] not in baseline_lookup
    base_scores, base_labels = [], []
    refined_scores, refined_labels = [], []
    processed = 0

    for batch in loader:
        with torch.no_grad():
            outputs = model.test_step(batch)

        for data_sample in outputs:
            score_map = data_sample.metainfo['anomaly_score_map']
            refined = data_sample.anomaly_scores.data.squeeze(0).detach().cpu().numpy()
            gt_map = data_sample.gt_sem_seg.sem_seg.squeeze(0).cpu().numpy()

            if compute_baseline:
                scores, labels = collect_valid(score_map, gt_map)
                base_scores.append(scores)
                base_labels.append(labels)

            scores, labels = collect_valid(refined, gt_map)
            refined_scores.append(scores)
            refined_labels.append(labels)
            processed += 1

        print(f'{spec["name"]}: processed {processed}/{len(dataset)}', flush=True)
        if args.max_samples and processed >= args.max_samples:
            break

    refined_scores = np.concatenate(refined_scores)
    refined_labels = np.concatenate(refined_labels)
    if compute_baseline:
        base_scores = np.concatenate(base_scores)
        base_labels = np.concatenate(base_labels)
        baseline = compute_metrics(base_scores, base_labels)
    else:
        baseline = baseline_lookup[spec['name']]

    return OrderedDict(
        dataset=spec['name'],
        samples=processed,
        baseline=baseline,
        detseg_r=compute_metrics(refined_scores, refined_labels),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument(
        '--config',
        default='configs/detseg/detseg-r_swin-b_external.py')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/detseg_swin-b_coco_1node_l40s/iter_38038.pth')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS_CFG))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--baseline-json', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out', default='work_dirs/detseg_r_eval/official.json')
    return parser.parse_args()


def load_baseline_lookup(path):
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    return {
        row['dataset']: row['baseline']
        for row in rows
        if row.get('status') is None and 'baseline' in row
    }


def main():
    args = parse_args()
    os.chdir(args.root)
    model = build_model(args.config, args.checkpoint, args.device)
    out_path = Path(args.out)
    baseline_lookup = load_baseline_lookup(args.baseline_json)
    results = []
    done = set()
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding='utf-8'))
        done = {row['dataset'] for row in results if row.get('status') is None}

    for dataset_key in args.datasets:
        if dataset_key not in DATASETS_CFG and dataset_key not in REFINE_EVALS:
            raise KeyError(f'Unknown dataset: {dataset_key}')
        install_eval_spec(dataset_key)
        dataset_name = DATASETS_CFG[dataset_key]['name']
        if dataset_name in done:
            print(f'{dataset_name}: skipping existing result', flush=True)
            continue
        result = evaluate_one(model, dataset_key, args, baseline_lookup)
        results.append(result)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
