"""Verify the released ARCTIC clip mapping against trusted raw/prepared data.

Requires the original ARCTIC raw annotations and the corresponding prepared
HOIGPT motion features. Install the project with its preprocess extra.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rot6(values):
    values = np.asarray(values)
    return Rotation.from_rotvec(values.reshape(-1, 3)).as_matrix()[:, :, :2].reshape(len(values), -1)


def reconstruct(mano, obj, start, end):
    base = obj[start, 4:] / 1000
    left, right = mano['left'], mano['right']
    return np.concatenate([
        rot6(np.concatenate([left['rot'], left['pose']], axis=1)[start:end]),
        left['trans'][start:end] - base,
        rot6(np.concatenate([right['rot'], right['pose']], axis=1)[start:end]),
        right['trans'][start:end] - base,
        obj[start:end, :1],
        rot6(obj[start:end, 1:4]),
        obj[start:end, 4:] / 1000 - base,
    ], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/resources/arctic_local_snapshot/arctic_clips.json')
    parser.add_argument('--prepared-root', required=True, type=Path)
    parser.add_argument('--raw-root', required=True, type=Path,
                        help='ARCTIC raw_seqs directory containing sXX/*.mano.npy and *.object.npy')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('dataset') != 'arctic':
        raise ValueError('Expected the released ARCTIC schema_version=1 mapping')
    clips = manifest['clips']
    by_id = {clip['id']: clip for clip in clips}
    if len(by_id) != len(clips):
        raise ValueError('Duplicate IDs in manifest')
    split_ids = manifest['split_ids']
    for split, ordered in split_ids.items():
        path = args.prepared_root / (split + '.txt')
        if digest(path) != manifest['split_sha256'][split]:
            raise ValueError(f'{split}: split file hash differs from manifest')
        current = [s.strip() for s in path.read_text().splitlines() if s.strip()]
        if current != ordered or len(current) != len(set(current)):
            raise ValueError(f'{split}: ordered IDs differ from manifest')
        if set(current) != {clip['id'] for clip in clips if clip['split'] == split}:
            raise ValueError(f'{split}: clip assignments differ from split file')
    if sum(map(len, split_ids.values())) != len(clips):
        raise ValueError('Split files do not cover every clip exactly once')

    raw_root = args.raw_root.resolve()
    cached_source = None
    mano = obj = None
    largest_error = 0.0
    for count, clip in enumerate(clips, 1):
        name, source = clip['id'], clip['source']
        relative = Path(source)
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2:
            raise ValueError(f'{name}: invalid raw source path')
        path = (raw_root / relative).resolve()
        if not path.is_relative_to(raw_root):
            raise ValueError(f'{name}: source leaves raw dataset root')
        if relative.name.split('_', 1)[0] != name.split('_', 1)[1]:
            raise ValueError(f'{name}: object name does not match source')
        if source != cached_source:
            # Official ARCTIC MANO files are NumPy object dictionaries. Only
            # load raw annotations obtained from a trusted dataset source.
            mano = np.load(str(path) + '.mano.npy', allow_pickle=True).item()
            obj = np.load(str(path) + '.object.npy', allow_pickle=False)
            cached_source = source
        start, end = clip['start'], clip['end']
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(obj):
            raise ValueError(f'{name}: invalid frame interval')
        saved_path = args.prepared_root / 'new_joints' / (name + '.npy')
        if digest(saved_path) != clip['feature_sha256']:
            raise ValueError(f'{name}: saved feature SHA256 differs')
        saved = np.load(saved_path, allow_pickle=False)
        rebuilt = reconstruct(mano, obj, start, end)
        if saved.shape != rebuilt.shape or saved.shape[1] != 208:
            raise ValueError(f'{name}: feature shape differs')
        error = float(np.max(np.abs(saved - rebuilt)))
        largest_error = max(largest_error, error)
        if not np.isfinite(error) or error > manifest['verification']['feature_tolerance']:
            raise ValueError(f'{name}: feature mismatch, max absolute error {error}')
        caption = (args.prepared_root / 'texts' / (name + '.txt')).read_text().split('#', 1)[0].strip()
        if hashlib.sha256(caption.encode()).hexdigest() != clip['caption_sha256']:
            raise ValueError(f'{name}: caption digest differs')
        if count % 500 == 0:
            print(f'Verified {count}/{len(clips)}; max absolute feature error {largest_error:.3g}', flush=True)
    print(f'PASS: {len(clips)} ARCTIC clips; max absolute feature error {largest_error:.3g}')


if __name__ == '__main__':
    main()
