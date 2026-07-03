# Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes

![DetSeg framework](docs/framework.png)

This is the official implementation of **"DetSeg: A Novel Paradigm for Road Anomaly Detection with Object-Level Understanding"** (ICCV 2025).

## 📋 Abstract

Recognizing out-of-distribution (OoD) objects on roads is crucial for safe driving. Most existing methods rely on segmentation models' uncertainty as anomaly scores, often resulting in false positives - especially at ambiguous regions like boundaries, where segmentation models inherently exhibit high uncertainty. Additionally, it is challenging to define a suitable threshold to generate anomaly masks, especially with the inconsistencies in predictions across consecutive frames.

We propose **DetSeg**, a novel paradigm that helps incorporate object-level understanding. DetSeg first detects all objects in the open world and then suppresses in-distribution (ID) bounding boxes, leaving only OoD proposals. These proposals can either help previous methods eliminate false positives (**DetSeg-𝓡**), or generate binary anomaly masks without complex threshold search when combined with a box-prompted segmentation module (**DetSeg-𝓢**).

Additionally, we introduce **vanishing point guided Hungarian matching (VPHM)** to smooth the prediction results within a video clip, mitigating abrupt variations of predictions between consecutive frames.

## ✨ Highlights

- 🚀 **Object-Level Understanding**: Leverages detection to suppress false positives at ambiguous regions
- 🎯 **Two Variants**: DetSeg-𝓡 (refine existing methods) & DetSeg-𝓢 (threshold-free segmentation)
- 📉 **Up to 37.45% FPR₉₅ reduction** compared to previous methods

## 🛠️ Installation

### Tested Environment

- Python 3.8
- PyTorch 2.1.0 + CUDA 11.8
- TorchVision 0.16.0
- MMCV 2.1.0
- MMEngine 0.10.7

### Step-by-step Installation

```bash
# Clone the repository
git clone https://github.com/huachao0124/DetSeg-official.git
cd DetSeg-official

# Create conda environment
conda create -n detseg python=3.8 -y
conda activate detseg

# Install PyTorch
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install OpenMMLab dependencies
pip install mmengine==0.10.7
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# Install DetSeg dependencies and editable package
pip install -r requirements.txt
pip install -v -e .
```

If the server cannot access external package indexes directly, use a PyPI
mirror for normal packages and a proxy for the PyTorch/OpenMMLab wheel URLs.

## 📁 Data Preparation

```
data/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017/
│   └── val2017/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── gtFine/
├── fishyscapes/
│   ├── LostAndFound/
│   └── Static/
├── road_anomaly/
└── segment_me_if_you_can/
    ├── dataset_AnomalyTrack/
    └── dataset_ObstacleTrack/
```

COCO is required for DetSeg detection-module fine-tuning. Cityscapes is used
for DetSeg-R internal segmentation-branch training. Fishyscapes, RoadAnomaly,
and SegmentMeIfYouCan are evaluation datasets.

## 📦 Model Zoo

Place all model files under `ckpts/`. The DetSeg checkpoint is tracked by Git
LFS; use `git lfs pull` after cloning, or download it from the link below.

Released checkpoint:

| Name | Source | Save As | Purpose |
| --- | --- | --- | --- |
| DetSeg detection module | [DetSeg][detseg-coco] | `ckpts/detseg_swin-b_coco_20260623-207453f5.pth` | COCO fine-tuned weights for DetSeg-R and DetSeg-S evaluation |

Upstream pretrained weights:

| Name | Source | Save As | Purpose |
| --- | --- | --- | --- |
| Swin-B ImageNet-22K | [Swin Transformer][swin-b-22k] | `ckpts/swin_base_patch4_window12_384_22k.pth` | Backbone initialization |
| MM-Grounding-DINO Swin-B pretrain-all | [OpenMMLab][mm-gdino-swin-b-all] | `ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth` | Detection-module initialization |

[detseg-coco]: https://media.githubusercontent.com/media/huachao0124/DetSeg-official/main/ckpts/detseg_swin-b_coco_20260623-207453f5.pth
[swin-b-22k]: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
[mm-gdino-swin-b-all]: https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth

## 🚀 Training

### DetSeg Detection Module

```bash
# Single GPU
python tools/train.py configs/detseg/detseg_swin-b_coco.py

# Multi-GPU (8 GPUs)
./tools/dist_train.sh configs/detseg/detseg_swin-b_coco.py 8
```

The COCO fine-tuning config trains DetSeg's object detector and validates its
universal-query proposals with class-agnostic proposal recall (`AR@100` and
`AR@300`).

### DetSeg-R Segmentation Branch

```bash
# Single GPU
python tools/train.py configs/detseg/detseg-r_swin-b_cityscapes.py

# Multi-GPU (8 GPUs)
./tools/dist_train.sh configs/detseg/detseg-r_swin-b_cityscapes.py 8
```

## 📊 Evaluation

### DetSeg-R with External Score Maps

Use this config to refine anomaly score maps from methods such as
Mask2Anomaly, RbA, or UNO:

```bash
python tools/test.py configs/detseg/detseg-r_swin-b_external.py \
  ckpts/detseg_swin-b_coco_20260623-207453f5.pth
```

By default, this evaluates Mask2Anomaly score maps on Fishyscapes Lost & Found:

```text
data/FS_LostFound/images/
data/FS_LostFound/labels_masks/
other_score_results/score_results_m2a/FS_LostFound/
```

For another dataset or external method, copy
`configs/detseg/detseg-r_swin-b_external.py` and edit the variables near the
top of the config: `test_dataset_type`, `test_data_root`, `test_img_path`,
`test_seg_map_path`, `score_dir`, and `use_unify_gt`. Set
`model.use_sam_refine=True` only for the SAM-mask refinement ablation. Set
`model.filter_by_road_overlap=True` only for the road-overlap ablation;
reproduced paper results use both defaults disabled.

### Released Checkpoint Results

Metrics are reported as `AUROC / AP / FPR95`.
Small differences from the paper numbers are expected because the released
checkpoint is retrained with the same config and subject to normal training
variance.

| Dataset | Method | Paper DetSeg-R | Released checkpoint |
| --- | --- | --- | --- |
| RoadAnomaly | M2A | 98.33 / 87.57 / 5.70 | 98.51 / 88.58 / 5.30 |
| RoadAnomaly | RbA | 97.76 / 86.12 / 11.12 | 98.52 / 89.93 / 7.76 |
| RoadAnomaly | RbA OoD | 98.76 / 90.11 / 3.92 | 99.02 / 92.74 / 2.73 |
| RoadAnomaly | RPL+CoroCL | 97.46 / 79.58 / 10.16 | 96.35 / 74.49 / 14.13 |
| RoadAnomaly | UNO | 99.02 / 93.89 / 1.94 | 99.19 / 94.63 / 1.86 |
| FS Static | M2A | 99.42 / 93.12 / 1.71 | 99.61 / 91.02 / 1.78 |
| FS Static | RbA | 98.84 / 80.89 / 4.38 | 99.34 / 84.85 / 2.70 |
| FS Static | RbA OoD | 99.44 / 90.02 / 2.62 | 99.81 / 96.38 / 0.40 |
| FS Static | RPL+CoroCL | 99.79 / 93.66 / 0.63 | 99.79 / 93.64 / 0.61 |
| FS Static | UNO | 99.69 / 97.72 / 0.11 | 99.74 / 97.93 / 0.07 |
| FS Lost & Found | M2A | 96.79 / 78.15 / 4.84 | 96.23 / 72.54 / 9.40 |
| FS Lost & Found | RbA | 97.38 / 69.39 / 6.53 | 97.16 / 68.81 / 9.80 |
| FS Lost & Found | RbA OoD | 99.06 / 77.79 / 3.87 | 97.13 / 73.78 / 5.43 |
| FS Lost & Found | RPL+CoroCL | 99.45 / 75.23 / 2.15 | 99.42 / 72.52 / 2.54 |
| FS Lost & Found | UNO | 99.15 / 87.01 / 1.95 | 99.04 / 85.95 / 5.11 |

Use `tools/eval_detseg_r_official.py` to regenerate this comparison after
placing the required datasets and baseline score maps in the paths listed in
that script.

### DetSeg-R with Internal Score Maps

```bash
python tools/test.py configs/detseg/detseg-r_swin-b_internal.py ckpt_path
```

### DetSeg-S

DetSeg-S prompts SAM with retained DetSeg boxes and evaluates binary masks:

```bash
python tools/test.py configs/detseg/detseg-s_swin-b.py ckpt_path
```

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@InProceedings{Zhu_2025_ICCV,
    author    = {Zhu, Huachao and Liu, Zelong and Sun, Zhichao and Zou, Yuda and Xia, Gui-Song and Xu, Yongchao},
    title     = {Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {8472-8481}
}
```

## 🙏 Acknowledgements

This project is built upon [MMDetection](https://github.com/open-mmlab/mmdetection), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM](https://github.com/facebookresearch/segment-anything). We thank the authors for their excellent work.

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).
