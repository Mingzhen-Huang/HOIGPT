# Data resources and preprocessing

This guide documents the available data-preparation code and metadata. The
[Final ARCTIC models](../releases/arctic-final/README.md) provide the paired
dual-codebook tokenizer and Stage 3 language-model weights, with matching source,
normalization, configuration and verification records.

## Resource overview

| Resource | Provided support | Inputs and conventions |
| --- | --- | --- |
| ARCTIC raw clip conversion | CPU converter preserving the original 208-feature formulas | Explicit manifests map source sequences and frame intervals to clip IDs |
| ARCTIC annotation parsing | Manifest builder with optional contiguous-clip augmentation | Sorted sequence order or an explicitly supplied historical order |
| GRAB raw clip conversion | CPU converter preserving pose/translation conventions | Explicit contact intervals and split assignments in the input manifest |
| Caption preprocessing | spaCy lemmatization and word/POS conversion | Supply annotations and record the spaCy model version used |
| Normalization | Grouped-standard-deviation computation and released ARCTIC values | Use checkpoint-matched arrays; select the fit population explicitly for new statistics |
| ARCTIC split metadata | 5,423 training and 111 validation IDs | Local prepared-data IDs; provide the test split required by your evaluation protocol |
| Object conditioning | Mesh-to-cache builder and recorded ARCTIC point indices | Match mesh hashes and vertex order; preserve checkpoint-specific normal caches |
| GRAB split metadata | Explicit manifest input | Supply dataset-specific clip and split assignments |
| Stage 1 / Stage 3 checkpoints | [Final models](../releases/arctic-final/README.md) through Git LFS | Use the paired codebook, vocabulary and archived source |
| PointNet / evaluator weights | PointNet file and evaluator state embedded in Stage 3 | Full evaluation uses matching data and evaluator configuration |

These tools do not require a trained model or CUDA to extract the pose features.
They do not vendor MANO assets or raw dataset files. Obtain data under the relevant
dataset terms; only use trusted raw NumPy annotation dictionaries and pickle caches.

## Installation and entry point

From a source checkout or unpacked source distribution:

```bash
python -m pip install -e '.[preprocess]'
python scripts/prepare_data.py --help
```

For captions that do not already have verified word/POS tokens:

```bash
python -m spacy download en_core_web_sm
```

Outputs are never overwritten. Choose a new output directory if any target file
already exists. An interrupted/failed conversion may leave partial diagnostic
outputs; inspect them and use a fresh directory for the next run.

## Explicit clip manifests

Both converters take a JSON manifest with `schema_version: 1`, `dataset`, and a
`clips` list. See the [ARCTIC example](../assets/resources/examples/arctic_clips.json)
and [GRAB example](../assets/resources/examples/grab_clips.json).

Each clip specifies:

- `id`: output name without an extension; its second underscore-separated
  component must be the object name, as required by the training data loader.
- `source`: path relative to the raw root. ARCTIC uses `subject/sequence` without
  the `.mano.npy` suffix; GRAB uses `subject/sequence.npz`.
- `start`, `end`: zero-based frame interval **[start, end)**, not seconds.
- `caption`: the caption for this extracted clip.
- Optional `tokens`: already prepared `word/POS` tokens, separated by spaces.
- Optional `split`: `train`, `val`, or `test` from a verified split assignment.
  No split is inferred or randomized if omitted.

The converter saves `source_manifest.json` and `preprocessing.json` with raw-file
SHA-256 hashes, lengths, and conventions. Keep these with the generated data and
eventual checkpoints. Assign related/overlapping clips at the raw-sequence or
subject level according to your evaluation protocol; do not independently shuffle
augmented clips and call it the paper split.

### ARCTIC

Expected raw files include:

```text
RAW_SEQS/s01/box_use_02.mano.npy
RAW_SEQS/s01/box_use_02.object.npy
```

```bash
python scripts/prepare_data.py convert --dataset arctic \
  --raw-root /path/to/arctic_data/data/raw_seqs \
  --manifest assets/resources/examples/arctic_clips.json \
  --output /path/to/new_prepared_arctic
```

To create a **new** manifest from your local HOIGPT-style description annotations:

```bash
python scripts/prepare_data.py arctic-manifest \
  --descriptions /path/to/description \
  --output /path/to/arctic_clips.json
```

The description tree is `subject/sequence/description.txt`; lines look like
`42-62 grasp, both hands`. The original deduplication by start frame and preference
for both-hand descriptions are retained. `--augment-contiguous` enables the
original combinations of adjacent events separated by at most five frames.
The recovered scripts contain both augmented and unaugmented variants; select
the one matching your source dataset, not merely the filename of a paper config.

Default sequence order is sorted. `--sequence-order file.txt` accepts an explicit
ordered list of `subject/sequence` entries when a historical order is known. The
order is recorded in the manifest. **Do not apply the bundled numeric-ID split
snapshot to a freshly numbered manifest without verifying the ID mapping.**

### GRAB

The converter reads `lhand`/`rhand` annotation dictionaries containing `params`
with 45-dimensional `fullpose`, 3-dimensional `global_orient`, and `transl`, plus
`object.params.global_orient`/`transl`. Unexpected shapes fail explicitly.

```bash
python scripts/prepare_data.py convert --dataset grab \
  --raw-root /path/to/grab \
  --manifest /path/to/verified_grab_clips.json \
  --output /path/to/new_prepared_grab
```

The original GRAB script used contact-based clip boundaries and an inclusive
end frame. Supply those contact intervals in the manifest and convert an
inclusive `[start, end]` to `[start, end + 1)`. Keep each clip's split assignment
with its source manifest so the segmentation and evaluation protocol stay aligned.

### Feature conventions retained from the original scripts

| Feature | Columns | Convention |
| --- | --- | --- |
| Left hand | 0–95 / 96–98 | Six-dimensional rotations for 16 joints / translation |
| Right hand | 99–194 / 195–197 | Same convention |
| Object | 198 / 199–204 / 205–207 | Articulation / six-dimensional rotation / translation |

Six-dimensional rotations are the first **two columns** of the rotation matrix,
flattened row-wise, using the original `hoigpt.lib.utils.rot` functions. Do not
replace this with a different library's row-based 6D convention.

ARCTIC object translations are converted from millimeters to meters; hand
translations are already in meters. All translations are relative to the object's
position at the **clip start**. GRAB translations are in meters and retain the
object origin at **source frame 0**, as in the original script; its articulation
feature is zero. The converters do not resample frames. Existing training/token
loaders separately apply stride 4 to motions longer than 400 frames.

The optional legacy joint/vertex visualization files are not needed by the
published training loaders and are not generated by feature extraction.

## Captions and normalization

Conversion writes captions to `raw_captions/`. If `tokens` are supplied in a clip,
it also writes the training-ready `texts/<id>.txt` line. Otherwise run:

```bash
python scripts/prepare_data.py tokenize-text \
  --input /path/to/prepared/raw_captions --output /path/to/prepared/texts
```

This preserves the original hyphen removal, alphabetic-token filtering, and
noun/verb lemmatization except the word `left`. Output uses
`caption#word/POS word/POS#0.0#0.0`. Pin and record the spaCy model version for an
exact experiment; a different model version may change tokens and POS tags.

For **new training**, choose the training population explicitly:

```bash
python scripts/prepare_data.py stats --data-root /path/to/prepared \
  --split-file /path/to/prepared/train.txt --output /path/to/new_statistics
```

This produces `mean.npy`, `std.npy`, and a normalization provenance JSON. The
population variance and feature-group scaling follow the original `mean_var.py`:
shared scale per hand rotation block, shared object rotation scale, articulation
scale 1, and a shared translation scale. The old script pooled every feature file;
the release requires an explicit list to avoid silently using held-out data.
For existing weights, use their original normalization rather than recomputing it.

## Existing ARCTIC snapshot

[`assets/resources/arctic_local_snapshot/`](../assets/resources/arctic_local_snapshot/)
contains the local train/val lists, float64 normalization values,
mesh-dependent point indices and source checksums. Supply motions, captions,
meshes and MANO assets in the prepared-data workspace. Download model weights
from the [Final release](../releases/arctic-final/README.md).

```bash
python scripts/prepare_data.py restore-snapshot \
  --snapshot assets/resources/arctic_local_snapshot \
  --output /path/to/new_snapshot_directory
```

This restores the saved train/val lists and `mean.npy`/`std.npy`.
Provide `test.txt` according to your evaluation protocol. The saved lists fix
the local split produced by the original unseeded shuffle; preserve these lists
and their clip-ID mapping when using the released weights. Treat the supplied
normalization arrays as fixed checkpoint assets. For newly computed statistics,
record the selected fit population with the new arrays.

## Object meshes and point cache

Obtain the matching licensed meshes separately. ARCTIC expects
`meshes/<object>/mesh.obj` with per-vertex `parts.json`; GRAB expects
`meshes/<object>.ply`. Object point clouds are generated in meters.

For the recorded ARCTIC point selection:

```bash
python scripts/prepare_data.py object-cache --dataset arctic \
  --mesh-root /path/to/prepared/meshes \
  --indices assets/resources/arctic_local_snapshot/object_points.json \
  --output /path/to/new_cache/arctic.pkl
```

Mesh and part-label hashes must match the snapshot before indices are applied.
Meshes are loaded with `process=False` to preserve file vertex order. The default
mesh-processing step can reorder vertices even with `maintain_order=True`, making
both sampled indices and per-vertex part labels incorrect.
The cache contains the keys used by `hoigpt.lib.models.object`: object names,
point positions/normals, vertex indices, relative mesh paths, and ARCTIC part masks.
The metadata records the selection method, mesh hashes, vertex-order convention,
and trimesh version. As with other pickle caches, load only trusted output.
New caches carry a `vertex_order` marker which the training and geometry loaders
honor. Unmarked historical caches retain the previous loader behavior; rebuilding
a cache can therefore also change the full-mesh/part-label alignment. Keep the
cache version with the experiment instead of silently replacing an old one.

For all 11 ARCTIC objects in this snapshot, file-order indices reproduce the
historical point coordinates within `2e-8` meters. The cache builder computes
normals with trimesh, producing values different from the historical cache.
Retain the original normal cache for checkpoint-specific comparisons and keep
the cache version alongside the experiment.

For **new** data, omit `--indices` and choose `--points 1024 --seed 1234` for
deterministic farthest-point sampling. Record this sampling choice and its cache
with the experiment. Place the resulting `arctic.pkl` or `grab.pkl` at the
prepared dataset root and provide the corresponding licensed meshes separately.

## Validation coverage

- ARCTIC feature conversion matched the original formula element-for-element on
  a real raw clip, including clip-start translation rebasing.
- GRAB feature conversion matched the original formula on synthetic annotation
  dictionaries.
- Restored split lists and float64 normalization arrays match the saved local assets.
- Checks cover grouped normalization, caption processing, frame bounds and
  overwrite protection.
- Reconstructed object point coordinates match the recorded selection. Use the
  historical normal cache for checkpoints that depend on those normal values.

## Final model checkpoints

Download the [Final ARCTIC models](../releases/arctic-final/README.md) with Git LFS.
The release includes the dual-codebook tokenizer, Stage 3 language model,
PointNet initialization file, normalization arrays, matching source and loading
verification. Its SHA256 manifest identifies the exact paired weights.

The published metrics use the documented local ARCTIC validation protocol. Keep
the split, normalization, object point selection, tokenizer architecture and
vocabulary, Stage 3 weights, and evaluator configuration consistent. Evaluator
parameters are preserved as `metrics.*` in the Stage 3 checkpoint; the full
evaluation entry point also needs the matching evaluator file and prepared data.
Old single-codebook checkpoints are not interchangeable with this release.
