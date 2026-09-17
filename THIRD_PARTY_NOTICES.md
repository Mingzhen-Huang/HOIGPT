# Third-party notices

The HOIGPT tokenizer, training, and evaluation code contains or adapts the
third-party software listed below.
These notices apply to the relevant portions and are not replaced by the
repository-level MIT license.

## T2M-GPT

- Source: <https://github.com/Mael-zys/T2M-GPT>
- License: Apache License 2.0
- Used in: temporal encoder/decoder residual blocks and EMA vector-quantization
  foundations in `hoigpt/resnet.py`, `hoigpt/quantizer.py`, and
  `hoigpt/tokenizer.py`.
- Modifications: structured left-hand/right-hand/object streams, independent
  object codebook, resumable EMA state, device-safe buffers, validation, and
  the release-facing HOIGPT API.

The full license is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## MotionGPT

- Source: <https://github.com/OpenMotionLab/MotionGPT>
- Copyright: Copyright (c) 2023 OpenMotionLab
- License: MIT
- Used in: the tokenizer backbone, language-model architecture, Lightning
  training and evaluation pipeline, data loaders, metrics, configuration,
  callbacks, and utilities from which the HOIGPT implementation evolved.
- Modifications: HOI feature layouts, object conditioning, structured HOI
  tokens, HOI losses and dataset handling, and migration into the `hoigpt`
  Python package.

The full notice is in
[`LICENSES/MIT-OpenMotionLab.txt`](LICENSES/MIT-OpenMotionLab.txt).

## Text2HOI

- Source: <https://github.com/JunukCha/Text2HOI>
- Copyright: Copyright (c) 2024 Junuk Cha
- License: MIT
- Used in: the PointNet object encoder in `hoigpt/pointnet.py` and
  `hoigpt/lib/networks/pointnet.py`; MANO and object wrappers in
  `hoigpt/lib/models/`; and the HOI data, geometry, and rendering utilities in
  `hoigpt/lib/utils/`.
- Modifications: integration with the HOIGPT training and evaluation pipeline,
  local imports, configuration, and checkpoint handling.

The full notice is in [`LICENSES/MIT-Text2HOI.txt`](LICENSES/MIT-Text2HOI.txt).

## pointnet.pytorch

- Source: <https://github.com/fxia22/pointnet.pytorch>
- Copyright: Copyright (c) 2017 Fei Xia
- License: MIT
- Used in: the PointNet architecture incorporated through Text2HOI.

The full notice is in
[`LICENSES/MIT-pointnet.pytorch.txt`](LICENSES/MIT-pointnet.pytorch.txt).

## PyTorch3D

- Source: <https://github.com/facebookresearch/pytorch3d>
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates.
- License: BSD 3-Clause
- Used in: rotation-conversion functions adapted in `hoigpt/lib/utils/rot.py`.
  The renderer, mesh structures, and nearest-neighbor operations are imported
  from the separately installed `pytorch3d` package.

The full notice is in
[`LICENSES/BSD-3-Clause-PyTorch3D.txt`](LICENSES/BSD-3-Clause-PyTorch3D.txt).

## Kornia

- Source: <https://github.com/kornia/kornia>
- Copyright: Copyright 2018 Kornia Team
- License: Apache License 2.0
- Used in: matrix/quaternion/axis-angle conversion functions adapted through
  Text2HOI in `hoigpt/lib/utils/rot.py`.

The attribution and modification notice is in
[`LICENSES/NOTICE-Kornia.txt`](LICENSES/NOTICE-Kornia.txt); the full license is
in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## pyquaternion

- Source: <https://github.com/KieranWynn/pyquaternion>
- Copyright: Copyright (c) 2015 Kieran Wynn
- License: MIT
- Used in: the matrix-to-quaternion algorithm credited by the Kornia-derived
  conversion function in `hoigpt/lib/utils/rot.py`.

The full notice is in
[`LICENSES/MIT-pyquaternion.txt`](LICENSES/MIT-pyquaternion.txt).

## Ceres Solver

- Source: <https://github.com/ceres-solver/ceres-solver>
- Copyright: Copyright 2023 Google Inc.
- License: BSD 3-Clause for the rotation routines
- Used in: the quaternion-to-axis-angle conversion credited by the
  Kornia-derived function in `hoigpt/lib/utils/rot.py`.

The applicable notice is in
[`LICENSES/BSD-3-Clause-Ceres.txt`](LICENSES/BSD-3-Clause-Ceres.txt).

## External dependencies and excluded assets

The `smplx` implementation is an external dependency. The HOIGPT MANO wrapper
imports it; this distribution does not vendor the SMPL-X/MANO implementation
or the MANO model files. Users must obtain the model files and accept their
applicable terms independently.

Datasets, pretrained language models, PointNet and evaluator checkpoints,
trained HOIGPT weights, and experiment outputs are not included. They must be
obtained independently under their respective terms. The published source
also excludes the legacy MPG-proprietary transformation utilities and the
QuaterNet-derived HumanML3D quaternion preprocessing code.
