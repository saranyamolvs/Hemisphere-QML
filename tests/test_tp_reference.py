"""TP reference-page mapping, streaming exports, and output safety."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from schlieren_qml.tp_reference import export_tp_montages


class TPReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.track_dir = self.root / 'tracking'
        self.track_dir.mkdir()
        self.tracks = self.track_dir / 'tp_tracks.csv'
        self.manifest = self.track_dir / 'tp_tracking_manifest.json'
        self.config = dict(image_directory=str(self.images), start=0, stride=1, limit=19)
        self.rows = []
        for i in range(19):
            data = (np.arange(32*48).reshape(32, 48) + i).astype(np.uint16)
            Image.fromarray(data).save(self.images / f'frame{i:03d}.tif')
            self.rows.append(dict(filename=f'frame{i:03d}.tif', sequence_frame_index=i,
                                  time_s=i/51032, x_px=23.5, y_px=10.5,
                                  score=.8, peak_margin=.1, accepted=i != 1,
                                  source='manual' if i == 0 else 'template'))
        self.write_inputs()
        self.output = self.root / 'reference'

    def write_inputs(self):
        with self.tracks.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest.write_text(json.dumps(dict(config=self.config, original_shape=[32, 48])))

    def test_pages_index_markers_partial_page_and_tiffs_are_unchanged(self):
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.images.iterdir()}
        with patch('schlieren_qml.tp_reference.read_image', wraps=__import__(
                'schlieren_qml.data', fromlist=['read_image']).read_image) as read:
            result = export_tp_montages(self.tracks, self.output, tile_width=96)
        self.assertEqual(read.call_count, 19)
        self.assertEqual(result['n_pages'], 2)
        self.assertEqual(result['n_frames'], 19)
        with Path(result['index_csv']).open() as handle:
            index = list(csv.DictReader(handle))
        self.assertEqual(len(index), 19)
        self.assertEqual(index[0]['source'], 'manual')
        self.assertEqual(index[1]['accepted'], 'False')
        self.assertEqual(index[15]['reference_tile'], '16')
        self.assertEqual(index[16]['reference_tile'], '1')
        self.assertEqual(index[18]['reference_tile'], '3')
        self.assertEqual(index[18]['filename'], 'frame018.tif')
        with Image.open(self.output / result['page_files'][0]) as page:
            self.assertEqual(page.size, (384, 28+4*(48+64)))
            x, y = int(float(index[0]['thumbnail_x_px'])), int(float(index[0]['thumbnail_y_px']))
            self.assertEqual(page.getpixel((x, 28+48+y)), (255, 165, 0))
            self.assertEqual(page.getpixel((96+x, 28+48+y)), (255, 0, 255))
        with Image.open(self.output / result['page_files'][1]) as page:
            # The final page has three frames; its fourth tile stays blank.
            self.assertEqual(page.getpixel((3*96+20, 28+48+20)), (32,32,32))
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.images.iterdir()})
        self.assertEqual(result['tracks_sha256'], hashlib.sha256(self.tracks.read_bytes()).hexdigest())

    def test_overwrite_removes_stale_pages_and_rejects_unrelated_files(self):
        export_tp_montages(self.tracks, self.output, tile_width=96)
        with self.assertRaises(FileExistsError):
            export_tp_montages(self.tracks, self.output, tile_width=96)
        self.rows = self.rows[:3]
        self.config['limit'] = 3
        self.write_inputs()
        result = export_tp_montages(self.tracks, self.output, tile_width=96, overwrite=True)
        self.assertEqual(result['n_pages'], 1)
        self.assertFalse((self.output / 'tp_reference_00002.png').exists())
        (self.output / 'unrelated.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            export_tp_montages(self.tracks, self.output, tile_width=96, overwrite=True)
        self.assertEqual((self.output / 'unrelated.txt').read_text(), 'keep')

    def test_input_sequence_geometry_and_destination_protections(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            export_tp_montages(self.tracks, self.images / 'marked', tile_width=96)
        with self.assertRaisesRegex(ValueError, 'source tracking'):
            export_tp_montages(self.tracks, self.track_dir, tile_width=96, overwrite=True)
        self.rows[1]['filename'] = 'frame000.tif'
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'order'):
            export_tp_montages(self.tracks, self.output, tile_width=96)
        self.assertFalse(self.output.exists())
        self.rows[1]['filename'] = 'frame001.tif'
        self.rows[1]['x_px'] = 999
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'Invalid TP'):
            export_tp_montages(self.tracks, self.output, tile_width=96)
        self.assertFalse(self.output.exists())

    def test_failed_reexport_preserves_previous_complete_export(self):
        export_tp_montages(self.tracks, self.output, tile_width=96)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        Image.fromarray(np.zeros((4,4), dtype=np.uint16)).save(self.images / 'frame018.tif')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            export_tp_montages(self.tracks, self.output, tile_width=96, overwrite=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertFalse(list(self.root.glob('.reference_building_*')))

    def test_category_csv_supports_explicit_tracking_manifest(self):
        category_dir = self.root / 'categories'
        category_dir.mkdir()
        states = category_dir / 'tp_states.csv'
        definition_path = category_dir / 'tp_category_definition.json'
        definition_path.write_text(json.dumps(dict(fit_partition='train',
            upper_boundary_y_px=8., lower_boundary_y_px=16.)))
        fields = list(self.rows[0]) + ['label']
        with states.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{**row, 'label': 'candidate_2'} for row in self.rows])
        result = export_tp_montages(states, self.output, manifest_path=self.manifest, tile_width=96)
        with Path(result['index_csv']).open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['label'], 'candidate_2')
        self.assertEqual(result['category_boundaries_y_px'], [8.,16.])
        with Image.open(self.output / result['page_files'][0]) as page:
            # Height doubles from 32 to 64 at width 96; pixel-center mapping.
            self.assertEqual(page.getpixel((4,28+48+16)), (0,255,255))
            self.assertEqual(page.getpixel((4,28+48+32)), (255,255,0))
            # TP marker remains visible above the boundary overlays.
            self.assertEqual(page.getpixel((47,28+48+21)), (255,165,0))
        definition_path.write_text(json.dumps(dict(fit_partition='train',
            upper_boundary_y_px=12., lower_boundary_y_px=16.)))
        with self.assertRaisesRegex(ValueError, 're-export categories'):
            export_tp_montages(states, self.root/'stale', manifest_path=self.manifest, tile_width=96)
        with self.assertRaisesRegex(ValueError, 'tile_width'):
            export_tp_montages(self.tracks, self.root/'invalid', tile_width=True)


if __name__ == '__main__':
    unittest.main()
