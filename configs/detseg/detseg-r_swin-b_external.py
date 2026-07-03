# Reproduction entry: evaluate DetSeg-R refinement on external anomaly score
# maps. Override dataset and score-map paths with --cfg-options when needed.
_base_ = './detseg-r_swin-b_internal.py'

model = dict(
    type='DetSegR',
    backbone=dict(out_indices=(1, 2, 3)),
    seg_decoder=None,
    use_sam_refine=False,
    sam_model_name='facebook/sam-vit-base',
    box_score_thr=0.2,
    strong_objectness_box=1.0,
    filter_by_road_overlap=False)

test_dataset_type = 'FSLostAndFoundDataset'
test_data_root = 'data/FS_LostFound'
test_img_path = 'images'
test_seg_map_path = 'labels_masks'
score_dir = 'other_score_results/score_results_m2a/FS_LostFound'
use_unify_gt = False

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=False, with_seg=True),
]
if use_unify_gt:
    test_pipeline.append(dict(type='UnifyGT', label_map={0: 0, 2: 1}))
test_pipeline.extend([
    dict(type='ConcatPrompt'),
    dict(type='GetAnomalyScoreMap', data_path=score_dir),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'anomaly_score_map', 'scale_factor', 'flip', 'flip_direction',
            'text', 'custom_entities')),
])

test_dataset = dict(
    _delete_=True,
    type=test_dataset_type,
    data_root=test_data_root,
    pipeline=test_pipeline,
    data_prefix=dict(img_path=test_img_path, seg_map_path=test_seg_map_path))

val_dataloader = dict(batch_size=1, num_workers=2, dataset=test_dataset)
test_dataloader = val_dataloader
val_evaluator = dict(type='AnomalyMetricLoad')
test_evaluator = val_evaluator
