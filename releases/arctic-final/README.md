# HOIGPT Final Models

This is the final release of the paired ARCTIC dual-codebook tokenizer and
Stage 3 FLAN-T5-base model.

| Model | Epoch (one-based) | Validation metric | File |
| --- | --- | --- | --- |
| Dual-codebook tokenizer | 380 | Reconstruction FID 0.4223284243 | [codebook-epoch0380.ckpt](codebook-epoch0380.ckpt) |
| Stage 3, FLAN-T5-base | 10 | Generation validation FID 2.9045820236 | [stage3-epoch0010.ckpt](stage3-epoch0010.ckpt) |

Stage 3 starts from Stage 2's best checkpoint at epoch 80 (FID 3.5523691177).
Both models use the same tokenizer parameters. Reconstruction and generation FID
measure different tasks. These scores use the local ARCTIC validation protocol:
111 split IDs, 101 unique examples after the motion-length filter, and 104 FID
samples after eight-rank sampler padding. Retrieval metrics use 96 examples in
groups of 32. Codebook reconstruction FID uses 101 examples.

## Download

From the repository root:

```bash
git lfs install
git lfs pull --include="releases/arctic-final/*.ckpt,releases/arctic-final/pointfeat.pth"
cd releases/arctic-final
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
- `source.tar.gz` contains the matching training architecture and configuration,
  plus its license notices. Use this source for these weights; the
  repository's historical top-level implementation has different details.
- `source-manifest.json` records every archived source file's SHA256.
  `manifest.json` and `SHA256SUMS` identify the released assets.
- `model-config.json`, `hoi-tokens.json` and `weights-verification.json` record the
  resolved model construction, ordered HOI vocabulary, and actual CPU strict-load
  checks. `environment.json` lists the observed package versions.
- The training YAML files and `codebook-recipe.json` document the recipe.
  Machine-local paths in the original training YAMLs must be overridden for a new
  environment; these YAMLs are not standalone launch scripts.

Provide ARCTIC/GRAB motions, captions, meshes, canonical object caches, MANO models,
GloVe and base FLAN-T5 files in the locations used by your configuration.

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

release = Path("releases/arctic-final").resolve()
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
`metrics.*` for evaluator identity checks. The CPU verifier checks loading and
weight identity; FID computation uses the full evaluation pipeline and the
protocol documented above.
