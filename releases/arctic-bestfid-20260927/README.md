# ARCTIC best-FID model snapshot — 2026-09-27

This release pairs the completed dual-codebook training run's best tokenizer with
the current best validation checkpoint from the ongoing Stage 3 run.

| Model | Epoch (one-based) | Selection metric | File |
| --- | --- | --- | --- |
| Dual-codebook tokenizer | 380 / 2000 | Reconstruction FID 0.4223284243 | [codebook-epoch0380.ckpt](codebook-epoch0380.ckpt) |
| Stage 3, FLAN-T5-base | 10 / planned 100 | Generation validation FID 2.9045820236 | [stage3-epoch0010.ckpt](stage3-epoch0010.ckpt) |

Stage 3 starts from Stage 2's epoch 80 best (FID 3.5523691177).
Stage 3 epoch 20 subsequently scored 3.1227087975, so epoch 10 remains the
best at this snapshot. These are local ARCTIC results. Reconstruction FID and
generation FID measure different tasks. The split contains 111 IDs; the motion
length filter leaves 101 unique validation examples. Eight-rank validation pads
to 104 FID samples; the retrieval metric uses 96 examples in groups of 32.
Codebook reconstruction FID uses 101 examples. The final 20-seed evaluation is pending;
this release does not claim the paper's mixed ARCTIC+GRAB benchmark scores.

## Download

From the repository root:

```bash
git lfs install
git lfs pull --include="releases/arctic-bestfid-20260927/*.ckpt,releases/arctic-bestfid-20260927/pointfeat.pth"
cd releases/arctic-bestfid-20260927
sha256sum -c SHA256SUMS
tar -xzf source.tar.gz
```

The checkpoint files use Git LFS. A small text file beginning with
`version https://git-lfs.github.com/spec/v1` is a pointer; run the pull command
above to obtain the actual weights. Downloaded GitHub source ZIPs may contain
pointers instead of weights.

## Contents and compatibility

- Both checkpoints contain `state_dict` and plain `metadata`. Tensor values and
  dtypes are preserved. Stage 3 includes its tokenizer and evaluator parameters.
- Optimizer, scheduler, callback, loop, RNG and pickled configuration objects are
  omitted. These files support inference and weight initialization, not exact
  interrupted-training resume.
- `pointfeat.pth` is extracted from the tokenizer's own frozen PointNet parameters.
  The VQVae constructor requires this initialization file before loading its state.
- `mean.npy` and `std.npy` are the exact 208-feature normalization arrays.
- `source.tar.gz` contains the matching training architecture and configuration
  snapshot, plus its license notices. Use this source for these weights; the
  repository's historical top-level implementation has different details.
- `source-manifest.json` records every archived source file's SHA256.
  `manifest.json` and `SHA256SUMS` identify the released assets.
- `model-config.json`, `hoi-tokens.json` and `weights-verification.json` record the
  resolved model construction, ordered HOI vocabulary, and actual CPU strict-load
  checks. `environment.json` lists the observed package versions.
- The training YAML files and `codebook-recipe.json` document the recipe.
  Machine-local paths in the original training YAMLs must be overridden for a new
  environment; these YAMLs are not standalone launch scripts.

No ARCTIC/GRAB motions, captions, meshes, canonical object caches, MANO models,
GloVe or base FLAN-T5 files are included.

## Verify loading

Use the training environment dependencies described in the repository README.
Supply a local `google/flan-t5-base` directory with its configuration, tokenizer,
SentencePiece files and either `model.safetensors` or `pytorch_model.bin`.
The constructor initializes from this base model before applying the Stage 3 state.

From this release directory, after extracting the source:

```bash
python verify_weights.py \
  --source ./source \
  --codebook ./codebook-epoch0380.ckpt \
  --stage3 ./stage3-epoch0010.ckpt \
  --pointnet-output ./pointfeat.pth \
  --flan-path /absolute/path/to/flan-t5-base \
  --report /tmp/hoigpt-weights-verification.json
```

The verifier uses CPU, reads weights with `weights_only=True`, constructs VQVae
and MLM from the archived source, strictly loads the tokenizer and language-model components, and checks every
loaded tensor. Evaluator tensors are retained but its modules are not instantiated
by this verifier. It also checks that the separately released tokenizer exactly
equals the tokenizer embedded in Stage 3. It does not load datasets or compute FID.

For component loading, run the following example from the repository root and
use the constructor dictionaries in `model-config.json`:

```python
import json, sys
from pathlib import Path
import torch

release = Path("releases/arctic-bestfid-20260927").resolve()
sys.path.insert(0, str(release / "source"))
from hoigpt.config import instantiate_from_config

cfg = json.loads((release / "model-config.json").read_text())
cfg["vae"]["params"]["pointnet_checkpoint"] = str(release / "pointfeat.pth")
cfg["lm"]["params"]["model_path"] = "/absolute/path/to/flan-t5-base"
book = torch.load(str(release / "codebook-epoch0380.ckpt"),
                  map_location="cpu", weights_only=True)
stage3 = torch.load(str(release / "stage3-epoch0010.ckpt"),
                    map_location="cpu", weights_only=True)
vae = instantiate_from_config(cfg["vae"]).eval()
vae.load_state_dict({k[4:]: v for k, v in book["state_dict"].items()}, strict=True)
lm = instantiate_from_config(cfg["lm"]).eval()
lm.load_state_dict({k[3:]: v for k, v in stage3["state_dict"].items()
                    if k.startswith("lm.")}, strict=True)
```

Use the archived tokenizer logic and the recorded order of the 1030 added HOI
tokens. The shared hand book and separate object book each contain 512 entries.
Do not mix these weights with tokens generated by a different codebook.

Decoding requires the object's corresponding normalized canonical point cloud;
it is not determined by the text alone. Use the original object preparation
convention and matching object assets. Decoded motion has 208 features:
left hand `0:99`, right hand `99:198`, object `198:208`; convert normalized
features with `motion * std + mean`. Temporal downsampling is 4.
MANO is required for the mesh/joint and rendering workflows.

The full `test.py` evaluation pipeline additionally requires the matching
prepared ARCTIC data, GloVe and TM2T evaluator. The Stage 3 state preserves
`metrics.*` so evaluator identity can be checked; strict CPU loading alone is
not an independent evaluation of the published FID.
