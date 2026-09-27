# Changelog

Notable changes to the HOIGPT code release are documented here.

## Final release

- Added checkpoint-free ARCTIC/GRAB clip-to-feature converters and a deterministic
  ARCTIC annotation-to-manifest command, preserving explicit sample IDs and frames.
- Added original caption processing and grouped normalization, plus object-cache
  reconstruction from separately obtained meshes and recorded point indices.
- Preserved mesh file vertex order for point indices and part labels, and
  documented the normal-cache convention. Training/geometry loaders honor the
  new cache marker and preserve the behavior of historical caches.
- Organized the local ARCTIC split lists, normalization and point-index metadata
  into a checkpoint-matched resource snapshot with explicit manifest inputs
  for other datasets and evaluation splits.
- Added the resource preparation/provenance guide.
- Published the [Final ARCTIC models](releases/arctic-final/README.md): paired
  dual-codebook tokenizer and Stage 3 weights, PointNet initialization, matching
  source, model configuration, normalization, checksums and strict-load verification.

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
