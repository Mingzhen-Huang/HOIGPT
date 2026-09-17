# Changelog

Notable changes to the HOIGPT code release are documented here.

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
