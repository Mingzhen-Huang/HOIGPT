"""Normalization and object-cache preparation without pretrained weights."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sample_id(value):
    if not isinstance(value, str) or not value or '/' in value or '\\' in value or any(c.isspace() or ord(c) < 32 for c in value) or value in ('.', '..'):
        raise ValueError(f'Invalid sample ID: {value!r}')
    if len(value.split('_')) < 2 or not value.split('_')[1]:
        raise ValueError(f'Sample ID must have the object in its second underscore component: {value}')
    return value


def split_ids(path):
    ids = [sample_id(line.strip()) for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate IDs in {path}')
    return ids


def new_outputs(paths):
    paths = [Path(p) for p in paths]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite {path}; select a new output directory')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def normalization(data_root, ids):
    """Streaming population moments followed by the original feature-group stds."""
    count = 0
    mean = np.zeros(208, dtype=np.float64)
    m2 = np.zeros(208, dtype=np.float64)
    for name in ids:
        motion = np.load(Path(data_root) / 'new_joints' / f'{sample_id(name)}.npy', allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 208 or not len(motion) or not np.isfinite(motion).all():
            raise ValueError(f'Invalid features for {name}')
        motion = motion.astype(np.float64)
        n = len(motion)
        local_mean = motion.mean(axis=0)
        delta = local_mean - mean
        m2 += ((motion - local_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if not count:
        raise ValueError('No frames for normalization')
    std = np.sqrt(np.maximum(m2 / count, 0))
    std[:96] = std[:96].mean()
    std[99:195] = std[99:195].mean()
    std[199:205] = std[199:205].mean()
    std[198] = 1.0
    locations = [96, 97, 98, 195, 196, 197, 205, 206, 207]
    std[locations] = std[locations].mean()
    if (std <= 0).any():
        raise ValueError('Zero normalization scale: use a sufficiently varied training split')
    return mean, std, count


def farthest_points(vertices, count, seed):
    """Deterministic new-cache option; not claimed to match historical point IDs."""
    if len(vertices) < count:
        raise ValueError(f'Mesh has {len(vertices)} vertices, fewer than {count} requested points')
    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(len(vertices), np.inf)
    used = np.zeros(len(vertices), dtype=bool)
    index = int(rng.integers(len(vertices)))
    for i in range(count):
        selected[i] = index
        used[index] = True
        distances = np.minimum(distances, ((vertices - vertices[index]) ** 2).sum(axis=1))
        distances[used] = -np.inf
        index = int(np.argmax(distances))
    return selected


def object_cache(mesh_root, dataset, names, indices=None, points=1024, seed=1234):
    import trimesh

    if dataset not in ('arctic', 'grab') or points < 1:
        raise ValueError('Expected arctic/grab and a positive point count')
    if indices is not None and (
        indices.get('schema_version') != 1
        or indices.get('dataset') != dataset
        or indices.get('vertex_order') != 'file_order_process_false'
        or indices.get('points_per_object') != points
    ):
        raise ValueError('Point snapshot must specify matching dataset, count and unprocessed file vertex order')
    root = Path(mesh_root)
    result = {key: {} for key in ('obj_pcs', 'obj_pc_normals', 'point_sets', 'obj_path')}
    result['vertex_order'] = 'file_order_process_false'
    result['object_name'] = sorted(names)
    if dataset == 'arctic':
        result['obj_pc_top'] = {}
    metadata = {}
    for name in sorted(names):
        if '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError(f'Invalid object name: {name}')
        relative = f'{name}/mesh.obj' if dataset == 'arctic' else f'{name}.ply'
        path = root / relative
        digest = sha256(path)
        record = indices['objects'][name] if indices is not None else None
        if record is not None and digest != record['mesh_sha256']:
            raise ValueError(f'{name}: mesh differs from the point-index snapshot (including vertex ordering)')
        # Processing merges/reorders vertices even with maintain_order=True.
        # Historical point indices and ARCTIC parts.json refer to file order.
        mesh = trimesh.load(path, maintain_order=True, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f'Expected a single triangular mesh: {path}')
        vertices = np.asarray(mesh.vertices).copy()
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f'{name}: invalid mesh vertices')
        if dataset == 'arctic':
            vertices /= 1000.0
        point_set = np.asarray(record['indices']) if record else farthest_points(vertices, points, seed)
        if not np.issubdtype(point_set.dtype, np.integer) or point_set.shape != (points,) or (point_set < 0).any() or (point_set >= len(vertices)).any() or len(np.unique(point_set)) != points:
            raise ValueError(f'{name}: invalid point indices')
        point_set = point_set.astype(np.int64)
        result['point_sets'][name] = point_set
        result['obj_pcs'][name] = vertices[point_set]
        result['obj_pc_normals'][name] = np.asarray(mesh.vertex_normals)[point_set].copy()
        result['obj_path'][name] = relative
        metadata[name] = {'mesh_sha256': digest, 'vertices': len(vertices)}
        if dataset == 'arctic':
            parts_path = root / name / 'parts.json'
            if record and sha256(parts_path) != record['parts_sha256']:
                raise ValueError(f'{name}: parts.json differs from snapshot')
            parts = np.asarray(json.loads(parts_path.read_text()))
            if parts.shape != (len(vertices),) or not np.isin(parts, [0, 1]).all():
                raise ValueError(f'{name}: expected one binary part label per vertex')
            result['obj_pc_top'][name] = (1 - parts)[point_set]
            metadata[name]['parts_sha256'] = sha256(parts_path)
    return result, metadata


def write_object_cache(path, cache, metadata):
    path = Path(path)
    provenance = path.with_suffix('.provenance.json')
    new_outputs([path, provenance])
    with path.open('wb') as stream:
        pickle.dump(cache, stream, protocol=4)
    provenance.write_text(json.dumps(metadata, indent=2) + '\n')
