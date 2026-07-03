import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASETS = OrderedDict(
    fs_lostfound=dict(
        rel='FS_LostFound/images',
        suffixes=('.png',),
        out='FS_LostFound'),
    fs_static=dict(
        rel='FS_Static/original',
        suffixes=('.png',),
        out='FS_Static'),
    road_anomaly=dict(
        rel='RoadAnomaly/frames',
        frame_list='RoadAnomaly/frame_list.json',
        out='RoadAnomaly'),
    road_anomaly_jpg=dict(
        rel='RoadAnomaly_jpg_official/frames',
        frame_list='RoadAnomaly_jpg_official/frame_list.json',
        out='RoadAnomaly_jpg'),
    smiyc_anomaly=dict(
        rel='SMIYC/dataset_RoadAnomalyTrack/images',
        suffixes=('.jpg',),
        out='dataset_RoadAnomalyTrack'),
    smiyc_obstacle=dict(
        rel='SMIYC/dataset_ObstacleTrack/images',
        suffixes=('.webp',),
        out='dataset_ObstacleTrack'),
)


def list_images(data_root, dataset_key):
    spec = DATASETS[dataset_key]
    base = Path(data_root)
    if 'frame_list' in spec:
        frame_list = base / spec['frame_list']
        names = json.loads(frame_list.read_text(encoding='utf-8'))
        return [base / spec['rel'] / name for name in names]

    root = base / spec['rel']
    suffixes = spec['suffixes']
    return sorted(p for p in root.iterdir() if p.suffix.lower() in suffixes)


def load_image_tensor(path, mean, std):
    image = Image.open(path).convert('RGB')
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    return tensor


def strip_state_dict(state_dict):
    for key in ('model', 'model_state', 'state_dict'):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
    return state_dict


def build_rpl(repo, checkpoint, device):
    code_root = Path(repo) / 'rpl_corocl.code'
    sys.path.insert(0, str(code_root))
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        from config.config import config  # noqa: E402
        from model.network import Network  # noqa: E402
        from valid import compute_anomaly_score  # noqa: E402
    finally:
        os.chdir(cwd)

    model = Network(config.num_classes)
    state_dict = strip_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    def infer(image_tensor):
        with torch.no_grad():
            _, logits, _ = model(image_tensor[None].to(device))
            return compute_anomaly_score(logits, mode=config.measure_way)

    return infer, config.image_mean.astype(np.float32), config.image_std.astype(np.float32)


def build_pebal(repo, checkpoint, device):
    code_root = Path(repo) / 'code'
    sys.path.insert(0, str(code_root))
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        from config.config import config  # noqa: E402
        from model.network import Network  # noqa: E402
    finally:
        os.chdir(cwd)

    model = Network(config.num_classes, wide=True)
    state_dict = strip_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    def infer(image_tensor):
        with torch.no_grad():
            return model(image_tensor[None].to(device), output_anomaly=True)

    return infer, config.image_mean.astype(np.float32), config.image_std.astype(np.float32)


def build_model(args):
    if args.method == 'rpl_corocl':
        return build_rpl(args.repo, args.checkpoint, args.device)
    if args.method == 'pebal':
        return build_pebal(args.repo, args.checkpoint, args.device)
    raise KeyError(args.method)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['rpl_corocl', 'pebal'], required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    infer, mean, std = build_model(args)

    for dataset_key in args.datasets:
        if dataset_key not in DATASETS:
            raise KeyError(f'Unknown dataset: {dataset_key}')
        images = list_images(args.data_root, dataset_key)
        out_dir = Path(args.out_root) / DATASETS[dataset_key]['out']
        if args.dry_run:
            print(f'{dataset_key}: {len(images)} images -> {out_dir}')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(images):
            if args.max_samples and idx >= args.max_samples:
                break
            score_path = out_dir / f'{image_path.name}.npy'
            image_tensor = load_image_tensor(image_path, mean, std)
            score = infer(image_tensor).detach().cpu().numpy().astype(np.float32)
            np.save(score_path, score)
            print(
                f'{dataset_key}: {idx + 1}/{len(images)} {score_path}',
                flush=True)


if __name__ == '__main__':
    main()
