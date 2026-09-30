"""CPU-only checks for the SMIYC upload format and failure handling."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import h5py
import numpy as np


SPEC = importlib.util.spec_from_file_location(
    'prepare_smiyc', Path(__file__).resolve().parents[1]
    / 'tools/prepare_smiyc_submission.py')
smiyc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smiyc)


def hdf5_blob(scores):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as handle:
        handle.create_dataset('value', data=scores)
    return buffer.getvalue()


class SubmissionTests(unittest.TestCase):
    def test_singleton_axis_and_negative_scores_are_preserved(self):
        scores = np.arange(6, dtype=np.float16).reshape(1, 2, 3) - 3
        result = smiyc.validate_hdf5(hdf5_blob(scores), 'frame')
        self.assertEqual(result['shape'], [2, 3])
        self.assertEqual(result['stored_shape'], [1, 2, 3])
        self.assertEqual(result['minimum'], -3)

    def test_invalid_scores(self):
        for scores in [np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                       np.zeros((2, 2, 3)), np.zeros((2, 3), dtype=np.int16)]:
            with self.subTest(shape=scores.shape, dtype=scores.dtype):
                with self.assertRaises(ValueError):
                    smiyc.validate_hdf5(hdf5_blob(scores), 'frame')

    def test_unexpected_paths(self):
        for path in ['../anomaly_p/M/AnomalyTrack-all/f.hdf5',
                     '/anomaly_p/M/AnomalyTrack-all/f.hdf5',
                     'anomaly_p/M/Other/f.hdf5', 'readme.txt']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                smiyc.identify_member(path)

    def test_packaging_round_trip_and_overwrite_rejection(self):
        blob = hdf5_blob(np.arange(6, dtype=np.float16).reshape(1, 2, 3))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.zip'
            members = [f'AnomalyTrack-all/frame{i}.hdf5' for i in range(110)]
            for scene, count in [('curvy-street', 327), ('driveway', 30),
                                 ('snowstorm1', 55), ('validation_fixture', 30)]:
                members.extend(f'ObstacleTrack-all/{scene}_{i}.hdf5'
                               for i in range(count))
            with zipfile.ZipFile(source, 'w') as archive:
                for member in members:
                    archive.writestr('outputs/anomaly_p/DetSeg/' + member, blob)
            with contextlib.redirect_stdout(io.StringIO()):
                report = smiyc.prepare([source], root / 'ready')[0]
            self.assertEqual(report['tracks'], {
                'AnomalyTrack-all': 110, 'ObstacleTrack-all': 442})
            with zipfile.ZipFile(root / 'ready/DetSeg.zip') as archive:
                self.assertEqual(len(archive.namelist()), 552)
                for name in archive.namelist():
                    self.assertTrue(name.startswith('anomaly_p/DetSeg/'))
                    self.assertEqual(archive.read(name), blob)
            with self.assertRaises(FileExistsError):
                smiyc.prepare([source], root / 'ready')
            with self.assertRaises(ValueError):
                smiyc.prepare([source, source], root / 'duplicates')


if __name__ == '__main__':
    unittest.main()
