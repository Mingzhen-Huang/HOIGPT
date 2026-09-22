# Changelog

Notable changes to the HOIGPT code release are documented here.

## Unreleased

- Added checkpoint-free ARCTIC/GRAB clip-to-feature converters and a deterministic
  ARCTIC annotation-to-manifest command, preserving explicit sample IDs and frames.
- Added original caption processing and grouped normalization, plus object-cache
  reconstruction from separately obtained meshes and recorded point indices.
- Preserved mesh file vertex order for point indices/part labels; documented
  the unresolved difference between recomputed and historical object normals.
  Training/geometry loaders honor the new cache marker without changing the
  mesh-loading behavior of unmarked historical caches.
- Organized existing ARCTIC split lists, normalization, and point-index metadata
  as an explicitly unverified local snapshot. Missing paper/test/GRAB mappings
  remain documented gaps; no random replacement split is generated.
- Added the resource preparation/provenance guide. No checkpoints are included.
- Updated data preparation instructions and marked pretrained HOIGPT checkpoints
  as "to be tuned", with release pending tuning and validation.

## 0.1.0

- Added the canonical `hoigpt` Python package.
- Included the original Stage 1 tokenizer training, Stage 2 language-model
  pretraining, Stage 3 instruction tuning, token export, and evaluation pipeline.
- Migrated the training model, data loaders, losses, metrics, and required geometry
  utilities from the local `mGPT`/`lib` tree into `hoigpt`.
- Included ARCTIC/GRAB stage configurations and the original instruction templates.
- Replaced machine-specific asset paths with configurable paths and CLI overrides.
- Added shared-hand and separate-object EMA codebooks with resumable state.
- Added an injectable, frozen-by-default PointNet object encoder.
- Added strict validation for features, point clouds, and structured tokens.
- Added migration support for historical `vae.` tokenizer state dictionaries.
- Kept the standalone tokenizer API alongside the original training architecture.
- Added clean packaging, tests, CI, and third-party license notices.
