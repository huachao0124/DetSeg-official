#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/code/DetSeg-official}"
BASELINES="${BASELINES:-/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/code/baselines}"
CONDA_SH="${CONDA_SH:-/root/conda/etc/profile.d/conda.sh}"
TRAIN_SESSION="${TRAIN_SESSION:-coco_train}"
WORK_DIR="${WORK_DIR:-work_dirs/detseg_swin-b_coco_1node_l40s}"
CONFIG="${CONFIG:-configs/detseg/detseg_swin-b_coco.py}"
SHM_CKPT="${SHM_CKPT:-/dev/shm/detseg_coco_final.pth}"
DEVICE="${DEVICE:-cuda:0}"
MIN_ITER="${MIN_ITER:-38000}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p nohup_logs work_dirs/score_map_eval work_dirs/box_recall work_dirs/detseg_r_eval

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

score_count() {
  find "$1" -type f -name '*.npy' 2>/dev/null | wc -l
}

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  log "waiting for tmux session $TRAIN_SESSION"
  while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

latest_ckpt="$(ls -1v "$WORK_DIR"/iter_*.pth | tail -n 1)"
latest_iter="$(basename "$latest_ckpt" | sed -E 's/^iter_([0-9]+)\.pth$/\1/')"
if [[ "$latest_iter" -lt "$MIN_ITER" ]]; then
  log "latest checkpoint $latest_ckpt is below MIN_ITER=$MIN_ITER"
  exit 1
fi
log "using checkpoint $latest_ckpt"
cp "$latest_ckpt" "$SHM_CKPT"

source "$CONDA_SH"

if [[ "$(score_count other_score_results/score_results_rpl_corocl)" -lt 200 ]]; then
  log "exporting RPL+CoroCL score maps"
  conda activate rpl_env
  python tools/baselines/save_energy_scores.py \
    --method rpl_corocl \
    --repo "$BASELINES/RPL" \
    --data-root data \
    --checkpoint "$BASELINES/_weights_try/rpl_corocl/rev3.pth" \
    --out-root other_score_results/score_results_rpl_corocl \
    --device "$DEVICE" \
    --datasets fs_lostfound road_anomaly smiyc_anomaly smiyc_obstacle
else
  log "skipping RPL+CoroCL score maps; found $(score_count other_score_results/score_results_rpl_corocl) files"
fi

conda activate uno_env
if [[ "$(score_count other_score_results/score_results_uno_ade)" -lt 200 ]]; then
  log "exporting UNO-ADE score maps"
  python tools/baselines/save_uno_scores.py \
    --repo "$BASELINES/Open-set-M2F" \
    --config-file "$BASELINES/Open-set-M2F/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs12_2k_city+vistas_uno.yaml" \
    --weights "$BASELINES/_weights_try/uno/uno_ade.pth" \
    --data-root data \
    --out-root other_score_results/score_results_uno_ade \
    --device "$DEVICE" \
    --datasets fs_lostfound road_anomaly smiyc_anomaly smiyc_obstacle
else
  log "skipping UNO-ADE score maps; found $(score_count other_score_results/score_results_uno_ade) files"
fi

if [[ "$(score_count other_score_results/score_results_uno_synthetic)" -lt 200 ]]; then
  log "exporting UNO-Synthetic score maps"
  python tools/baselines/save_uno_scores.py \
    --repo "$BASELINES/Open-set-M2F" \
    --config-file "$BASELINES/Open-set-M2F/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs12_2k_city+vistas_uno.yaml" \
    --weights "$BASELINES/_weights_try/uno/uno_synthetic.pth" \
    --data-root data \
    --out-root other_score_results/score_results_uno_synthetic \
    --device "$DEVICE" \
    --datasets fs_lostfound road_anomaly smiyc_anomaly smiyc_obstacle
else
  log "skipping UNO-Synthetic score maps; found $(score_count other_score_results/score_results_uno_synthetic) files"
fi

log "evaluating baseline score maps"
conda activate detseg
if [[ -f work_dirs/score_map_eval/baseline_scores_posttrain.json ]]; then
  log "skipping baseline score-map evaluation; output already exists"
else
  CUDA_VISIBLE_DEVICES="" python tools/eval_score_maps.py \
    --datasets \
    fs_lostfound_m2a fs_lostfound_rba fs_lostfound_rpl_corocl fs_lostfound_uno_ade fs_lostfound_uno_synthetic \
    road_anomaly_m2a road_anomaly_rba road_anomaly_rpl_corocl road_anomaly_uno_ade road_anomaly_uno_synthetic \
    smiyc_anomaly_m2a smiyc_anomaly_rba smiyc_anomaly_rpl_corocl smiyc_anomaly_uno smiyc_anomaly_uno_ade smiyc_anomaly_uno_synthetic \
    smiyc_obstacle_m2a smiyc_obstacle_rba smiyc_obstacle_rpl_corocl smiyc_obstacle_uno smiyc_obstacle_uno_ade smiyc_obstacle_uno_synthetic \
    --out work_dirs/score_map_eval/baseline_scores_posttrain.json
fi

log "evaluating final box recall"
if [[ -f work_dirs/box_recall/coco_final.json ]]; then
  log "skipping final box recall; output already exists"
else
  python tools/eval_box_recall.py \
    --root . \
    --config "$CONFIG" \
    --checkpoint "$SHM_CKPT" \
    --datasets fs_lostfound road_anomaly smiyc_anomaly smiyc_obstacle \
    --device "$DEVICE" \
    --out work_dirs/box_recall/coco_final.json
fi

log "evaluating DetSeg-R refinements"
if [[ -f work_dirs/detseg_r_eval/coco_final_posttrain.json ]]; then
  log "skipping DetSeg-R refinement evaluation; output already exists"
else
  python tools/eval_detseg_r.py \
    --root . \
    --config "$CONFIG" \
    --checkpoint "$SHM_CKPT" \
    --datasets \
    fs_lostfound_m2a fs_lostfound_rba fs_lostfound_rpl_corocl fs_lostfound_uno_ade fs_lostfound_uno_synthetic \
    road_anomaly_m2a road_anomaly_rba road_anomaly_rpl_corocl road_anomaly_uno_ade road_anomaly_uno_synthetic \
    smiyc_anomaly_m2a smiyc_anomaly_rba smiyc_anomaly_rpl_corocl smiyc_anomaly_uno smiyc_anomaly_uno_ade smiyc_anomaly_uno_synthetic \
    smiyc_obstacle_m2a smiyc_obstacle_rba smiyc_obstacle_rpl_corocl smiyc_obstacle_uno smiyc_obstacle_uno_ade smiyc_obstacle_uno_synthetic \
    --device "$DEVICE" \
    --out work_dirs/detseg_r_eval/coco_final_posttrain.json
fi

log "done"
