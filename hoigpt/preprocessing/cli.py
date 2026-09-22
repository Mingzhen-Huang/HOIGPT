"""Command-line preparation of checkpoint-free HOIGPT resources."""

import argparse
import json
from pathlib import Path

import numpy as np

from .features import arctic_features, grab_features
from .resources import new_outputs, normalization, object_cache, sample_id, sha256, split_ids, write_object_cache


def inside(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError(f'Source must be relative to the raw dataset root: {relative}')
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'Source must be relative to the raw dataset root: {relative}')
    return candidate


def read_manifest(path, dataset):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema_version') != 1 or manifest.get('dataset') != dataset:
        raise ValueError('Expected schema_version=1 and the selected dataset in the manifest')
    clips = manifest['clips']
    if not clips:
        raise ValueError('Manifest has no clips')
    ids = set()
    for clip in clips:
        name = sample_id(clip['id'])
        if name in ids:
            raise ValueError(f'Duplicate clip ID: {name}')
        ids.add(name)
        if not isinstance(clip['source'], str) or not clip['source']:
            raise ValueError(f'{name}: missing raw source')
        caption = clip['caption']
        if not isinstance(caption, str) or not caption.strip() or any(c in caption for c in '#\n\r'):
            raise ValueError(f'{name}: caption must be one nonempty line without #')
        if 'tokens' in clip:
            tokens = clip['tokens']
            if not isinstance(tokens, str) or not tokens.split() or any(c in tokens for c in '#\n\r') or any('/' not in token for token in tokens.split()):
                raise ValueError(f'{name}: tokens must be a space-separated word/POS string')
        if clip.get('split') not in (None, 'train', 'val', 'test'):
            raise ValueError(f'{name}: split must be train, val, test, or omitted')
    return manifest


def convert(args):
    manifest = read_manifest(args.manifest, args.dataset)
    out = Path(args.output)
    split_names = sorted({c['split'] for c in manifest['clips'] if c.get('split')})
    paths = [out / 'source_manifest.json', out / 'preprocessing.json']
    paths += [out / f'{split}.txt' for split in split_names]
    for clip in manifest['clips']:
        paths.extend([out / 'new_joints' / f'{clip["id"]}.npy', out / 'raw_captions' / f'{clip["id"]}.txt'])
        if 'tokens' in clip:
            paths.append(out / 'texts' / f'{clip["id"]}.txt')
    new_outputs(paths)
    records = []
    for clip in manifest['clips']:
        source = inside(args.raw_root, clip['source'])
        if args.dataset == 'arctic':
            if source.name.split('_')[0] != clip['id'].split('_')[1]:
                raise ValueError(f'{clip["id"]}: object name differs from ARCTIC source sequence')
            mano_path = Path(str(source) + '.mano.npy')
            object_path = Path(str(source) + '.object.npy')
            # Official raw annotations use NumPy object dictionaries. Only load
            # trusted datasets: allow_pickle=True can execute pickle payloads.
            mano = np.load(mano_path, allow_pickle=True).item()
            objects = np.load(object_path, allow_pickle=False)
            features = arctic_features(mano, objects, clip['start'], clip['end'])
            hashes = {'mano_sha256': sha256(mano_path), 'object_sha256': sha256(object_path)}
        else:
            with np.load(source, allow_pickle=True) as raw:
                if str(raw['obj_name'].item()) != clip['id'].split('_')[1]:
                    raise ValueError(f'{clip["id"]}: object name does not match raw GRAB metadata')
                mano = {side: raw[key].item()['params'] for side, key in [('left', 'lhand'), ('right', 'rhand')]}
                obj = raw['object'].item()['params']
                objects = np.concatenate((obj['global_orient'], obj['transl']), axis=1)
            features = grab_features(mano, objects, clip['start'], clip['end'])
            hashes = {'source_sha256': sha256(source)}
        if features.shape[1] != 208 or not np.isfinite(features).all():
            raise ValueError(f'{clip["id"]}: invalid converted features')
        np.save(out / 'new_joints' / f'{clip["id"]}.npy', features)
        (out / 'raw_captions' / f'{clip["id"]}.txt').write_text(clip['caption'] + '\n')
        if 'tokens' in clip:
            (out / 'texts' / f'{clip["id"]}.txt').write_text(f'{clip["caption"]}#{clip["tokens"]}#0.0#0.0\n')
        records.append({'id': clip['id'], 'frames': len(features), **hashes})
        print(f'{clip["id"]}: {features.shape}')
    for split in split_names:
        (out / f'{split}.txt').write_text(''.join(c['id'] + '\n' for c in manifest['clips'] if c.get('split') == split))
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'preprocessing.json').write_text(json.dumps({
        'manifest_sha256': sha256(args.manifest), 'dataset': args.dataset,
        'end_convention': 'exclusive', 'temporal_resampling': 'none',
        'translation_origin': 'clip-start object' if args.dataset == 'arctic' else 'source-frame-0 object',
        'clips': records,
    }, indent=2) + '\n')


def arctic_manifest(args):
    """Original annotation parsing, with recorded deterministic traversal."""
    root = Path(args.descriptions)
    if args.sequence_order:
        sources = [line.strip() for line in Path(args.sequence_order).read_text().splitlines() if line.strip()]
        if len(sources) != len(set(sources)):
            raise ValueError('Duplicate source sequences in --sequence-order')
    else:
        sources = sorted(str(p.parent.relative_to(root)) for p in root.glob('*/*/description.txt'))
    clips = []
    for source in sources:
        entries = {}
        for line in (inside(root, source) / 'description.txt').read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            start, end = map(int, parts[0].split('-'))
            action = ' '.join(parts[1:-2]).removesuffix(',')
            hand = ' '.join(parts[-2:])
            if start not in entries or 'both' in hand:
                entries[start] = (min(start, end), max(start, end), [action], [hand])
        segments = list(entries.values())
        if args.augment_contiguous:
            ordered = sorted(segments, key=lambda row: row[0])
            groups = []
            for entry in ordered:
                if not groups or entry[0] > groups[-1][-1][1] + 5:
                    groups.append([])
                groups[-1].append(entry)
            segments = []
            for group in groups:
                for i in range(len(group)):
                    for j in range(i + 2, len(group) + 1):
                        subset = group[i:j]
                        segments.append((subset[0][0], subset[-1][1],
                                         sum((r[2] for r in subset), []), sum((r[3] for r in subset), [])))
                segments.extend(group)
        obj = Path(source).name.split('_')[0]
        for start, end, actions, hands in segments:
            if start == end:
                raise ValueError(f'Empty interval in {source}: {start}-{end}')
            caption = ', then '.join(f'{action} {obj} with {hand}' for action, hand in zip(actions, hands))
            clips.append({'id': f'{len(clips):06d}_{obj}', 'source': source,
                          'start': start, 'end': end, 'caption': caption})
    if not clips:
        raise ValueError('No ARCTIC description clips found')
    new_outputs([args.output])
    Path(args.output).write_text(json.dumps({'schema_version': 1, 'dataset': 'arctic',
        'status': 'new deterministic manifest; not certified as the paper ID mapping',
        'augment_contiguous': args.augment_contiguous, 'sequence_order': sources,
        'clips': clips}, indent=2) + '\n')
    print(f'Wrote {len(clips)} clips. No random split was generated; preserve this ID mapping.')


def tokenize(args):
    import spacy

    nlp = spacy.load(args.spacy_model)
    paths = sorted(Path(args.input).glob('*.txt'))
    if not paths:
        raise ValueError('No raw caption files found')
    out = Path(args.output)
    new_outputs([out / path.name for path in paths])
    for path in paths:
        lines = []
        for caption in path.read_text().splitlines():
            if '#' in caption:
                raise ValueError(f'{path}: already tokenized captions are not raw input')
            if not caption.strip():
                continue
            words = []
            for token in nlp(caption.replace('-', '')):
                if not token.text.isalpha():
                    continue
                word = token.lemma_ if token.pos_ in ('NOUN', 'VERB') and token.text != 'left' else token.text
                words.append(f'{word}/{token.pos_}')
            if not words:
                raise ValueError(f'{path}: caption has no word/POS tokens')
            lines.append(f'{caption}#{" ".join(words)}#0.0#0.0\n')
        (out / path.name).write_text(''.join(lines))
    print(f'Tokenized {len(paths)} files using {args.spacy_model} {nlp.meta.get("version")}')


def stats(args):
    ids = split_ids(args.split_file)
    out = Path(args.output)
    new_outputs([out / 'mean.npy', out / 'std.npy', out / 'normalization.json'])
    mean, std, count = normalization(args.data_root, ids)
    np.save(out / 'mean.npy', mean)
    np.save(out / 'std.npy', std)
    (out / 'normalization.json').write_text(json.dumps({'schema_version': 1,
        'method': 'population moments with original HOIGPT grouped standard deviations',
        'split_sha256': sha256(args.split_file), 'sequences': len(ids), 'frames': count,
        'mean': mean.tolist(), 'std': std.tolist()}, indent=2) + '\n')


def restore(args):
    snapshot = Path(args.snapshot)
    out = Path(args.output)
    normalization_data = json.loads((snapshot / 'normalization.json').read_text())
    split_paths = [snapshot / f'{name}.txt' for name in ('train', 'val', 'test') if (snapshot / f'{name}.txt').is_file()]
    seen = set()
    for path in split_paths:
        ids = set(split_ids(path))
        if seen & ids:
            raise ValueError('Snapshot splits overlap')
        seen |= ids
    mean, std = (np.asarray(normalization_data[key], dtype=np.float64) for key in ('mean', 'std'))
    if mean.shape != (208,) or std.shape != (208,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid snapshot normalization')
    new_outputs([out / p.name for p in split_paths] + [out / 'mean.npy', out / 'std.npy'])
    for path in split_paths:
        (out / path.name).write_bytes(path.read_bytes())
    np.save(out / 'mean.npy', mean)
    np.save(out / 'std.npy', std)
    print('Restored existing lists and normalization only. No missing split was invented.')


def objects(args):
    import trimesh

    if args.points < 1:
        raise ValueError('--points must be positive')
    indices = json.loads(Path(args.indices).read_text()) if args.indices else None
    if indices is not None and indices.get('dataset') != args.dataset:
        raise ValueError('Point-index snapshot dataset differs')
    if indices is not None:
        names = list(indices['objects'])
    else:
        root = Path(args.mesh_root)
        names = [p.parent.name for p in root.glob('*/mesh.obj')] if args.dataset == 'arctic' else [p.stem for p in root.glob('*.ply')]
    if not names:
        raise ValueError('No meshes found')
    cache, metadata = object_cache(args.mesh_root, args.dataset, names, indices, args.points, args.seed)
    write_object_cache(args.output, cache, {'dataset': args.dataset, 'points': args.points,
        'selection': 'snapshot' if indices else 'new deterministic farthest-point sampling',
        'vertex_order': 'file_order_process_false', 'trimesh_version': trimesh.__version__,
        'normals': 'trimesh vertex normals; not verified equivalent to historical cache normals',
        'seed': args.seed, 'objects': metadata})
    print(f'Built {len(names)} objects in {args.output}; mesh assets were not copied')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('convert', help='Convert manifest-selected raw ARCTIC/GRAB clips to 208 features')
    p.add_argument('--dataset', choices=['arctic', 'grab'], required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=convert)
    p = commands.add_parser('arctic-manifest', help='Record clip IDs from ARCTIC description annotations')
    p.add_argument('--descriptions', required=True)
    p.add_argument('--sequence-order', help='One relative subject/sequence per line; otherwise sorted order is used')
    p.add_argument('--augment-contiguous', action='store_true')
    p.add_argument('--output', required=True)
    p.set_defaults(run=arctic_manifest)
    p = commands.add_parser('tokenize-text', help='Apply original spaCy word/POS caption processing')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--spacy-model', default='en_core_web_sm')
    p.set_defaults(run=tokenize)
    p = commands.add_parser('stats', help='Calculate normalization from an explicitly selected split')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-file', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=stats)
    p = commands.add_parser('restore-snapshot', help='Restore available split lists and normalization; not raw ID mapping')
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=restore)
    p = commands.add_parser('object-cache', help='Build object point cache from separately obtained meshes')
    p.add_argument('--dataset', choices=['arctic', 'grab'], required=True)
    p.add_argument('--mesh-root', required=True)
    p.add_argument('--indices', help='Optional historical point-index snapshot with mesh checksums')
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output', required=True)
    p.set_defaults(run=objects)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
