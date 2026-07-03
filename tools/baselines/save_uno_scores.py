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
        names = json.loads((base / spec['frame_list']).read_text(encoding='utf-8'))
        return [base / spec['rel'] / name for name in names]
    root = base / spec['rel']
    return sorted(p for p in root.iterdir() if p.suffix.lower() in spec['suffixes'])


def setup_cfg(repo, config_file, weights, device, opts):
    sys.path.insert(0, str(Path(repo)))
    from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
    from detectron2.config import get_cfg  # noqa: E402
    from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
    from detectron2.data import MetadataCatalog  # noqa: E402
    from mask2former import add_maskformer2_config  # noqa: E402,F401
    from train_net import Trainer  # noqa: E402

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', weights, 'MODEL.DEVICE', device, *opts])
    cfg.freeze()

    # Model construction queries training metadata. Keep official metadata values
    # such as Mapillary Vistas ignore_label=65 when they were already registered.
    for name in cfg.DATASETS.TRAIN:
        metadata = MetadataCatalog.get(name)
        try:
            metadata.ignore_label
        except AttributeError:
            metadata.set(ignore_label=255)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)
    model.eval()
    return cfg, model


def image_to_input(path, cfg):
    from detectron2.data import transforms as T  # noqa: E402

    image = np.asarray(Image.open(path).convert(cfg.INPUT.FORMAT))
    height, width = image.shape[:2]
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST)
    image = aug.get_transform(image).apply_image(image)
    tensor = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
    return {'image': tensor, 'height': height, 'width': width}


def uno_score(output):
    mask_pred = output['mask_pred'].sigmoid()
    mask_cls = output['mask_cls']
    probs = mask_cls.softmax(-1)
    s_no = probs[..., :-2]
    s_unc = probs
    s_x = -s_unc[..., -2] + s_no.max(1)[0]
    v = (mask_pred * s_x.view(-1, 1, 1)).sum(0)
    return (-v).detach().cpu().numpy().astype(np.float32)


def run_model(model, sample):
    if model.__class__.__name__ != 'MaskFormerJointFlow':
        return model([sample])[0]

    from detectron2.structures import ImageList  # noqa: E402

    images = ImageList.from_tensors([sample['image']],
                                    model.size_divisibility)
    image_sizes = images.image_sizes
    images = (images.tensor - model.pixel_mean) / model.pixel_std
    return model([sample], images, None, image_sizes)[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_key in args.datasets:
        if dataset_key not in DATASETS:
            raise KeyError(f'Unknown dataset: {dataset_key}')
        images = list_images(args.data_root, dataset_key)
        out_dir = Path(args.out_root) / DATASETS[dataset_key]['out']
        if args.dry_run:
            print(f'{dataset_key}: {len(images)} images -> {out_dir}')
            continue
    if args.dry_run:
        return

    cfg, model = setup_cfg(args.repo, args.config_file, args.weights, args.device,
                           args.opts)
    model.to(args.device)

    for dataset_key in args.datasets:
        images = list_images(args.data_root, dataset_key)
        out_dir = Path(args.out_root) / DATASETS[dataset_key]['out']
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_path in enumerate(images):
            if args.max_samples and idx >= args.max_samples:
                break
            sample = image_to_input(image_path, cfg)
            sample['image'] = sample['image'].to(args.device)
            with torch.no_grad():
                output = run_model(model, sample)
            score = uno_score(output)
            score_path = out_dir / f'{image_path.name}.npy'
            np.save(score_path, score)
            print(
                f'{dataset_key}: {idx + 1}/{len(images)} {score_path}',
                flush=True)


if __name__ == '__main__':
    main()
