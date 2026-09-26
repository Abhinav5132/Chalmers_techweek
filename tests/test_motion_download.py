"""Catalog provenance and atomic download behavior without network access."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from download_motions import download


class MotionDownloadTests(unittest.TestCase):
    def test_catalog_covers_source_and_preserves_legacy_ids(self):
        catalog = json.loads((ROOT / 'src/motion_catalog.json').read_text())
        clips = catalog['clips']
        self.assertEqual(len(clips), 61)
        self.assertEqual(len({c['id'] for c in clips}), 61)
        self.assertEqual(len({c['path'] for c in clips}), 61)
        mapping = {c['source_name']: c['id'] for c in clips}
        self.assertEqual(mapping['J_ShortDance16_JazzWalk'], 'walk')
        self.assertEqual(mapping['J_Dance0_StepTouch'], 'step_touch')
        self.assertEqual(mapping['B_BowKarate'], 'bow')
        self.assertEqual(catalog['license'], 'CC-BY-4.0')

    def test_failed_replacement_keeps_existing_file_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / 'walk.npz'
            path.write_bytes(b'original invalid recording')
            with patch('download_motions.urllib.request.urlopen', side_effect=OSError('offline')), patch('download_motions.time.sleep'):
                with self.assertRaises(OSError):
                    download({'id': 'walk', 'path': 'walk.npz'}, {'dataset':'exptech/g1-moves', 'revision':'test'}, directory)
            self.assertEqual(path.read_bytes(), b'original invalid recording')
            self.assertEqual(list(directory.iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
