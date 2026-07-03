import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from mmengine.registry import init_default_scope

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_detseg_r import (  # noqa: E402
    DATASETS_CFG,
    REFINE_EVALS,
    build_dataset,
    collect_valid,
    compute_metrics,
)


SCORE_EVALS = OrderedDict(
    fs_lostfound_m2a=dict(
        source='fs_lostfound_m2a',
        name='FS_LostFound/M2A',
        score_dir='other_score_results/score_results_m2a/FS_LostFound'),
    fs_lostfound_rba=dict(
        source='fs_lostfound_rba',
        name='FS_LostFound/RbA',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/FS_LostFound'),
    fs_static_entropy=dict(
        source='fs_static_entropy',
        name='FS_Static/Entropy',
        score_dir='other_score_results/score_results_rpl_static_entropy/FS_Static'),
    fs_static_logit_distance=dict(
        source='fs_static_logit_distance',
        name='FS_Static/LogitDistance',
        score_dir='other_score_results/score_results_rpl_static_logit_distance/FS_Static'),
    fs_static_mae_features=dict(
        source='fs_static_mae_features',
        name='FS_Static/MAEFeatures',
        score_dir='other_score_results/score_results_rpl_static_mae_features/FS_Static'),
    fs_static_m2a=dict(
        source='fs_static_entropy',
        name='FS_Static/M2A',
        score_dir='other_score_results/score_results_m2a/FS_Static'),
    fs_static_rba=dict(
        source='fs_static_entropy',
        name='FS_Static/RbA',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/FS_Static'),
    fs_static_rba_ood=dict(
        source='fs_static_entropy',
        name='FS_Static/RbA-OoD',
        score_dir='other_score_results/score_results_rba_ood_map_coco/FS_Static'),
    fs_static_rpl_corocl=dict(
        source='fs_static_entropy',
        name='FS_Static/RPL+CoroCL',
        score_dir='other_score_results/score_results_rpl_corocl/FS_Static'),
    fs_static_uno_ade=dict(
        source='fs_static_entropy',
        name='FS_Static/UNO-ADE',
        score_dir='other_score_results/score_results_uno_ade/FS_Static'),
    fs_static_uno_synthetic=dict(
        source='fs_static_entropy',
        name='FS_Static/UNO-Synthetic',
        score_dir='other_score_results/score_results_uno_synthetic/FS_Static'),
    road_anomaly_m2a=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/M2A',
        score_dir='other_score_results/score_results_m2a/RoadAnomaly'),
    road_anomaly_rba=dict(
        source='road_anomaly_rba',
        name='RoadAnomaly/RbA official jpg',
        score_dir='other_score_results/score_results_rba_swin_b_1dl_official_jpg/RoadAnomaly_jpg'),
    road_anomaly_rba_webp=dict(
        source='road_anomaly_m2a',
        name='RoadAnomaly/RbA webp',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/RoadAnomaly'),
    smiyc_anomaly_m2a=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/M2A',
        score_dir='other_score_results/score_results_m2a/dataset_RoadAnomalyTrack'),
    smiyc_anomaly_rba=dict(
        source='smiyc_anomaly_rba',
        name='SMIYC-AnomalyTrack/RbA',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/dataset_RoadAnomalyTrack'),
    smiyc_anomaly_uno=dict(
        source='smiyc_anomaly_m2a',
        name='SMIYC-AnomalyTrack/UNO',
        score_dir='other_score_results/score_results_uno/dataset_RoadAnomalyTrack'),
    smiyc_obstacle_m2a=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/M2A',
        score_dir='other_score_results/score_results_m2a/dataset_ObstacleTrack'),
    smiyc_obstacle_rba=dict(
        source='smiyc_obstacle_rba',
        name='SMIYC-ObstacleTrack/RbA',
        score_dir='other_score_results/score_results_rba_swin_b_1dl/dataset_ObstacleTrack'),
    smiyc_obstacle_uno=dict(
        source='smiyc_obstacle_m2a',
        name='SMIYC-ObstacleTrack/UNO',
        score_dir='other_score_results/score_results_uno/dataset_ObstacleTrack'),
)


for key, spec in REFINE_EVALS.items():
    SCORE_EVALS.setdefault(key, spec)


def install_eval_spec(key):
    spec = SCORE_EVALS[key]
    dataset_cfg = DATASETS_CFG[spec['source']].copy()
    dataset_cfg['name'] = spec['name']
    dataset_cfg['score_dir'] = spec['score_dir']
    DATASETS_CFG[key] = dataset_cfg


def evaluate_one(dataset_key, args):
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
    scores_all, labels_all = [], []
    processed = 0

    for idx in range(len(dataset)):
        item = dataset[idx]
        data_sample = item['data_samples']
        score_map = data_sample.metainfo['anomaly_score_map']
        gt_map = data_sample.gt_sem_seg.sem_seg.squeeze(0).cpu().numpy()
        scores, labels = collect_valid(score_map, gt_map)
        scores_all.append(scores)
        labels_all.append(labels)
        processed += 1

        if args.progress_interval and processed % args.progress_interval == 0:
            print(
                f'{spec["name"]}: processed {processed}/{len(dataset)}',
                flush=True)
        if args.max_samples and processed >= args.max_samples:
            break

    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    return OrderedDict(
        dataset=spec['name'],
        samples=processed,
        baseline=compute_metrics(scores_all, labels_all),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--datasets', nargs='+', default=list(SCORE_EVALS))
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--progress-interval', type=int, default=20)
    parser.add_argument(
        '--out',
        default='work_dirs/score_map_eval/baseline_scores.json')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(args.root)
    init_default_scope('mmdet')

    results = []
    for dataset_key in args.datasets:
        if dataset_key not in SCORE_EVALS:
            raise KeyError(f'Unknown score eval dataset: {dataset_key}')
        results.append(evaluate_one(dataset_key, args))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
