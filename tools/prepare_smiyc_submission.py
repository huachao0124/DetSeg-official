"""Validate saved SMIYC HDF5 predictions and make one upload ZIP per method.

Accepts archives rooted at anomaly_p/ or outputs/anomaly_p/. Prediction
files are preserved byte for byte. No inference or network upload occurs.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

import h5py
import numpy as np


TRACKS = ('AnomalyTrack-all', 'ObstacleTrack-all')
# Match the official DatasetObstacleTrack splits in datasets/tracks.py.
TEST_SCENES = {
    'curvy-street', 'one-way-street', 'gravel', 'greyasphalt', 'motorway',
    'paving', 'darkasphalt', 'darkasphalt2',
}
VALIDATION_EXCLUDED = tuple(f'validation_{i}' for i in (19, *range(21, 30)))


def identify_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe ZIP member: {name}')
    parts = path.parts
    if parts and parts[0] == 'outputs':
        parts = parts[1:]
    if len(parts) != 4 or parts[0] != 'anomaly_p':
        raise ValueError(f'Unexpected ZIP member: {name}')
    _, method, track, filename = parts
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', method):
        raise ValueError(f'Invalid method name: {method}')
    if track not in TRACKS or not filename.endswith('.hdf5'):
        raise ValueError(f'Unexpected prediction path: {name}')
    return method, track, filename


def check_coverage(entries):
    anomaly = [e for e in entries if e['track'] == TRACKS[0]]
    obstacle = [Path(e['filename']).stem for e in entries if e['track'] == TRACKS[1]]
    counts = dict(
        anomaly_all=len(anomaly),
        obstacle_test=sum(fid.split('_')[0] in TEST_SCENES for fid in obstacle),
        obstacle_night=sum(fid.startswith('driveway_') for fid in obstacle),
        obstacle_snowstorm=sum(fid.split('_')[0] in ('snowstorm1', 'snowstorm2')
                               for fid in obstacle),
        obstacle_validation=sum(fid.startswith('validation_') and
                                not fid.startswith(VALIDATION_EXCLUDED)
                                for fid in obstacle),
    )
    expected = dict(anomaly_all=110, obstacle_test=327, obstacle_night=30,
                    obstacle_snowstorm=55, obstacle_validation=30)
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError(f'Incomplete official splits: {counts}; expected {expected}')
    allowed = TEST_SCENES | {'driveway', 'snowstorm1', 'snowstorm2', 'validation'}
    if any(fid.split('_')[0] not in allowed for fid in obstacle):
        raise ValueError('Unrecognized obstacle scene')
    return counts


def validate_hdf5(blob, name):
    with h5py.File(io.BytesIO(blob), 'r') as handle:
        if set(handle) != {'value'}:
            raise ValueError(f'{name}: expected only the HDF5 dataset "value"')
        if not isinstance(handle.get('value', getlink=True), h5py.HardLink):
            raise ValueError(f'{name}: linked datasets are not supported')
        dataset = handle['value']
        if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
            raise ValueError(f'{name}: expected a self-contained dataset')
        if dataset.ndim < 2 or min(dataset.shape) <= 0:
            raise ValueError(f'{name}: expected a nonempty spatial score map')
        if not np.issubdtype(dataset.dtype, np.floating):
            raise ValueError(f'{name}: expected floating-point anomaly scores')
        # Official Evaluation applies np.squeeze before scoring.
        scores = np.squeeze(dataset[:])
        if scores.ndim != 2:
            raise ValueError(f'{name}: expected H x W after squeezing singleton axes')
        if not np.isfinite(scores).all():
            raise ValueError(f'{name}: NaN or infinity in prediction')
        return dict(shape=list(scores.shape), stored_shape=list(dataset.shape),
                    dtype=str(scores.dtype),
                    minimum=float(scores.min()), maximum=float(scores.max()),
                    constant=bool(scores.min() == scores.max()),
                    compression=dataset.compression,
                    sha256=hashlib.sha256(blob).hexdigest())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(inputs, output_dir):
    grouped = defaultdict(list)
    seen = set()
    for source in inputs:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                method, track, filename = identify_member(member.filename)
                identity = (method, track, filename)
                if identity in seen:
                    raise ValueError(f'Duplicate prediction: {identity}')
                seen.add(identity)
                grouped[method].append(dict(source=str(source.resolve()),
                    member=member.filename, track=track, filename=filename))
    if not grouped:
        raise ValueError('No predictions found')
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, entries in grouped.items():
        check_coverage(entries)
        for suffix in ('.zip', '.manifest.json'):
            if (output_dir / (method + suffix)).exists():
                raise FileExistsError(output_dir / (method + suffix))

    reference_shapes = {}
    reference_frames = None
    reports = []
    for method, entries in sorted(grouped.items()):
        frame_set = {(e['track'], e['filename']) for e in entries}
        if reference_frames is not None and frame_set != reference_frames:
            raise ValueError(f'{method}: frame IDs differ between methods')
        reference_frames = frame_set
        records = []
        with tempfile.TemporaryDirectory(dir=output_dir) as temp:
            temporary = Path(temp) / 'submission.zip'
            from contextlib import ExitStack
            with ExitStack() as stack:
                sources = {source: stack.enter_context(zipfile.ZipFile(source))
                           for source in {e['source'] for e in entries}}
                archive = stack.enter_context(zipfile.ZipFile(
                    temporary, 'w', compression=zipfile.ZIP_STORED))
                for index, entry in enumerate(sorted(entries, key=lambda e: (
                        e['track'], e['filename'])), 1):
                    blob = sources[entry['source']].read(entry['member'])
                    record = validate_hdf5(blob, entry['member'])
                    key = (entry['track'], entry['filename'])
                    shape = record['shape']
                    if key in reference_shapes and shape != reference_shapes[key]:
                        raise ValueError(f'{key}: image dimensions differ between methods')
                    reference_shapes[key] = shape
                    member = f'anomaly_p/{method}/{entry["track"]}/{entry["filename"]}'
                    archive.writestr(member, blob)
                    records.append(dict(path=member, **record))
                    if index % 50 == 0 or index == len(entries):
                        print(f'{method}: checked {index}/{len(entries)}', flush=True)
            # Read the finished archive and verify every prediction is unchanged.
            with zipfile.ZipFile(temporary) as archive:
                for record in records:
                    digest = hashlib.sha256(archive.read(record['path'])).hexdigest()
                    if digest != record['sha256']:
                        raise ValueError(f'Archive round-trip failed: {record["path"]}')
            output = output_dir / (method + '.zip')
            os.link(temporary, output)
        report = dict(
            method=method, archive=str(output.resolve()),
            archive_bytes=output.stat().st_size, sha256=sha256_file(output),
            tracks=dict(Counter(e['track'] for e in entries)),
            coverage=check_coverage(entries),
            dtype_counts=dict(Counter(r['dtype'] for r in records)),
            constant_maps=sum(r['constant'] for r in records),
            minimum=min(r['minimum'] for r in records),
            maximum=max(r['maximum'] for r in records),
            source_archives=sorted({e['source'] for e in entries}),
            validation='Format, finite values, official split counts, and cross-method '
                       'frame IDs/shapes checked; original images were not available.',
            files=records)
        with output.with_suffix('.manifest.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs, args.output_dir)


if __name__ == '__main__':
    main()
