# Prepare SMIYC predictions for Codabench

The [SMIYC Codabench competition](https://www.codabench.org/competitions/5402/)
accepts saved pixel-wise anomaly scores for `AnomalyTrack-all` and
`ObstacleTrack-all`. Higher scores mean more anomalous pixels.

Use `tools/prepare_smiyc_submission.py` to check existing HDF5 prediction
archives and create one submission ZIP per method:

```bash
pip install numpy h5py
python tools/prepare_smiyc_submission.py \
  /path/to/detseg_results.zip /path/to/detsegv3_results.zip \
  --output-dir outputs/smiyc/ready
```

The tool accepts an optional outer `outputs/` directory and multiple
methods in the source archives. Each generated ZIP has this structure:

```text
anomaly_p/<method>/
├── AnomalyTrack-all/<frame ID>.hdf5
└── ObstacleTrack-all/<frame ID>.hdf5
```

Each HDF5 file must contain a self-contained floating-point `value` dataset.
Singleton axes such as `1 x H x W` are supported by the
[official evaluator](https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/evaluation.py).
The tool preserves the original HDF5 files byte for byte; it does not
normalize scores, apply thresholds, or run inference.

Validation checks include finite values, spatial dimensions, duplicate
files, cross-method frame IDs and dimensions, official split counts, and
archive round-trip hashes. Each ZIP has a separate `.manifest.json` with
per-file checksums and score ranges. Existing output files are not
overwritten. These checks do not replace comparison with the original
images or the official server evaluation.

The [upstream dataset definitions](https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/datasets/tracks.py)
expect 110 anomaly images, 327 obstacle test images, 30 night images, and
55 snowstorm images. Validation filtering is reported separately: the
current upstream code excludes prefixes `validation_19` through
`validation_29` while asserting 30 retained images. Some existing result
sets include `validation_20` and retain only 29 under this filter. The tool
records this mismatch as a warning and preserves the files unchanged;
confirm the intended split with the organizers if validation scoring is
requested.

## Submit and request leaderboard publication

1. Sign in to Codabench and register for competition 5402 using your access
   link if needed. Upload one method ZIP under **My Submissions**.
2. Save the submission ID. For each task, download **Output from scoring
   step → scoring_result.zip** and retain its `scores.json`.
3. Add verified scores to the appropriate tracks in
   [`assets/leaderboard.json`](https://github.com/SegmentMeIfYouCan/segment-me-if-you-can/blob/master/assets/leaderboard.json).
   Include an accurate method/configuration name, OoD-training declaration,
   paper link and code link. Use server scores, without replacing them
   with paper numbers or local validation results.
4. Submit a PR with the submission ID for organizer verification. Uploading
   to Codabench does not automatically update the official SMIYC website.

Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Beyond_Pixel_Uncertainty_Bounding_the_OoD_Objects_in_Road_Scenes_ICCV_2025_paper.html

Code: https://github.com/huachao0124/DetSeg-official

The competition configuration checked on 2026-09-29 allows 10 submissions
per participant in total and at most 10 per day. Check remaining quota
before uploading. Account credentials and competition access keys should
not be included in commits or public PRs.
