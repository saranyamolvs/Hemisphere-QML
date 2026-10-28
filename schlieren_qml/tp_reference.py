"""Streaming 4-by-4 TP reference-image exports without changing source TIFFs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from .data import read_image
from .triple_point import _outside_images, _sequence


_PAGE_PREFIX = 'tp_reference_'
_INDEX = 'tp_reference_index.csv'
_MANIFEST = 'tp_reference_manifest.json'
_ARTIFACT = 'tp_reference_montages'


def _fit_text(draw, text, width):
    """Bound the display label; the CSV always retains the complete filename."""
    text = str(text)
    while text and draw.textlength(text) > width:
        text = text[:-1]
    return text


def _check_destination(output, tracks, manifest, overwrite):
    if not isinstance(overwrite, bool):
        raise ValueError('overwrite must be boolean.')
    if output.is_symlink():
        raise ValueError('Montage output must not be a symlink.')
    for protected in (tracks, manifest):
        if output == protected or output in protected.parents:
            raise ValueError('Montage output must be separate from source tracking files.')
    if output.exists():
        if not output.is_dir():
            raise ValueError('Montage output must be a directory.')
        if not any(output.iterdir()):
            return
        if not overwrite:
            raise FileExistsError('Montage output exists; choose a new directory or enable overwrite.')
        previous = output / _MANIFEST
        if not previous.is_file() or previous.is_symlink():
            raise ValueError('Overwrite is allowed only for an existing TP montage export.')
        try:
            metadata = json.loads(previous.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('Existing montage export metadata is invalid.') from exc
        if metadata.get('artifact') != _ARTIFACT:
            raise ValueError('Overwrite is allowed only for an existing TP montage export.')
        expected = {_MANIFEST, _INDEX, *metadata.get('page_files', [])}
        if {p.name for p in output.iterdir()} != expected:
            raise ValueError('Montage output contains unrelated files; choose a new directory.')
        for path in output.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink > 1:
                raise ValueError('Montage output contains a file alias or directory.')


def export_tp_montages(tracks_csv, output_directory, *, manifest_path=None,
                       tile_width=384, overwrite=False):
    """Save all tracked frames as chronological 16-frame (4x4) PNG pages.

    ``tracks_csv`` can be tp_tracks.csv, or tp_states.csv with an explicit
    ``manifest_path`` pointing to the original tp_tracking_manifest.json.
    Source sensor coordinates stay in the index CSV; thumbnails use per-image
    min/max grayscale scaling for reference viewing only. Only one TIFF and one
    montage page are held at a time. Existing exports are replaced only after a
    complete new export has succeeded; stale pages from longer runs disappear.
    """
    if isinstance(tile_width, bool) or not isinstance(tile_width, int) or not 64 <= tile_width <= 4096:
        raise ValueError('tile_width must be an integer from 64 to 4096.')
    tracks = Path(tracks_csv).expanduser().resolve()
    source_manifest = (tracks.parent / 'tp_tracking_manifest.json' if manifest_path is None
                       else Path(manifest_path).expanduser().resolve())
    tracking = json.loads(source_manifest.read_text(encoding='utf-8'))
    config = tracking['config']
    paths, indices = _sequence(config)
    h, w = tracking['original_shape']
    if (isinstance(h, bool) or isinstance(w, bool) or not isinstance(h, int)
            or not isinstance(w, int) or h < 1 or w < 1):
        raise ValueError('Tracking manifest original_shape must contain positive image dimensions.')
    with tracks.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {'filename', 'sequence_frame_index', 'x_px', 'y_px', 'accepted', 'source'}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError('Tracking CSV needs unique columns with TP coordinates and provenance.')
        rows = list(reader)
    if len(rows) != len(paths):
        raise ValueError('Tracking CSV does not match the configured image sequence.')
    additions = ['reference_page', 'reference_tile', 'thumbnail_x_px', 'thumbnail_y_px']
    if set(additions).intersection(fields):
        raise ValueError('Input CSV already contains reference-export columns; use the original tracks.')
    for row, path, index in zip(rows, paths, indices):
        try:
            x, y = float(row['x_px']), float(row['y_px'])
            valid = (None not in row and row['filename'] == path.name
                     and int(row['sequence_frame_index']) == index
                     and row['accepted'] in {'True', 'False'}
                     and np.isfinite(x) and np.isfinite(y)
                     and 0 <= x <= w - 1 and 0 <= y <= h - 1)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('Invalid TP row or filename/frame order differs from the selected images.')
    raw_output = Path(output_directory).expanduser()
    if raw_output.is_symlink():
        raise ValueError('Montage output must not be a symlink.')
    output = _outside_images(raw_output, config)
    _check_destination(output, tracks, source_manifest, overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}_building_', dir=output.parent))
    backup = None
    try:
        thumb_h = max(1, int(round(h * tile_width / w)))
        header, page_header = 48, 28
        tile_h = header + thumb_h
        page_size = (4 * tile_width, page_header + 4 * tile_h)
        n_pages = (len(rows) + 15) // 16
        page_files = []
        with (staging / _INDEX).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + additions)
            writer.writeheader()
            for number, (row, path) in enumerate(zip(rows, paths)):
                slot = number % 16
                if slot == 0:
                    page_number = number // 16 + 1
                    page_name = f'{_PAGE_PREFIX}{page_number:05d}.png'
                    page = Image.new('RGB', page_size, '#202020')
                    draw = ImageDraw.Draw(page)
                    draw.text((6, 7), f'TP reference page {page_number}/{n_pages} | '
                              'Orange: accepted; magenta: needs review', fill='white')
                image = np.asarray(read_image(path), dtype=float)
                if image.shape != (h, w) or not np.all(np.isfinite(image)):
                    raise ValueError(f'Image dimensions changed or pixels are not finite: {path.name}')
                low, high = float(image.min()), float(image.max())
                if high > low:
                    gray = np.rint(255 * np.clip((image - low) / (high - low), 0, 1)).astype(np.uint8)
                else:
                    gray = np.zeros((h, w), dtype=np.uint8)
                thumbnail = Image.fromarray(gray).convert('RGB').resize(
                    (tile_width, thumb_h), resample=Image.Resampling.LANCZOS)
                tx = (float(row['x_px']) + .5) * tile_width / w - .5
                ty = (float(row['y_px']) + .5) * thumb_h / h - .5
                tx = float(np.clip(tx, 0, tile_width - 1))
                ty = float(np.clip(ty, 0, thumb_h - 1))
                color = '#FFA500' if row['accepted'] == 'True' else '#FF00FF'
                marker = ImageDraw.Draw(thumbnail)
                radius = max(3, int(round(tile_width / 64)))
                marker.line([(tx-radius, ty), (tx+radius, ty)], fill=color, width=2)
                marker.line([(tx, ty-radius), (tx, ty+radius)], fill=color, width=2)
                left, top = (slot % 4) * tile_width, page_header + (slot // 4) * tile_h
                status = 'accepted' if row['accepted'] == 'True' else 'NEEDS REVIEW'
                labels = [f"Frame {row['sequence_frame_index']} | {row['filename']}",
                          f"x={float(row['x_px']):.2f}, y={float(row['y_px']):.2f} px | {status}",
                          row.get('label', row['source'])]
                for line_no, label in enumerate(labels):
                    draw.text((left+4, top+3+14*line_no),
                              _fit_text(draw, label, tile_width-8), fill=color if line_no == 1 else 'white')
                page.paste(thumbnail, (left, top+header))
                writer.writerow({**row, 'reference_page': page_name, 'reference_tile': slot+1,
                                 'thumbnail_x_px': tx, 'thumbnail_y_px': ty})
                if slot == 15 or number == len(rows)-1:
                    page.save(staging / page_name, format='PNG')
                    page_files.append(page_name)
                    if page_number % 25 == 0 or page_number == n_pages:
                        print(f'TP reference pages: {page_number:,}/{n_pages:,}', flush=True)
        metadata = {'artifact': _ARTIFACT, 'n_frames': len(rows), 'n_pages': n_pages,
                    'frames_per_page': 16, 'columns': 4, 'rows': 4, 'tile_width': tile_width,
                    'thumbnail_height': thumb_h, 'tile_header_height': header,
                    'page_header_height': page_header, 'page_files': page_files,
                    'tracks_csv': str(tracks), 'tracks_sha256': hashlib.sha256(tracks.read_bytes()).hexdigest(),
                    'tracking_manifest': str(source_manifest),
                    'tracking_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                    'tracking_coordinates': 'Original sensor pixels; y increases downward.',
                    'display_conversion': 'Per-image min/max grayscale to 8-bit, resized for reference only.',
                    'marker_colors': {'orange': 'accepted', 'magenta': 'needs review'},
                    'original_tiffs_modified': False}
        (staging / _MANIFEST).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f'.{output.name}_previous_', dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return {**metadata, 'output_directory': str(output), 'index_csv': str(output / _INDEX),
                'manifest_json': str(output / _MANIFEST)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
