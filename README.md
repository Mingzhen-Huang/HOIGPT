# HOIGPT

Training and evaluation code for **HOIGPT: Learning Long-Sequence Hand-Object
Interaction with Language Models** (CVPR 2025).

[[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_HOIGPT_Learning_Long-Sequence_Hand-Object_Interaction_with_Language_Models_CVPR_2025_paper.html)]
[[arXiv](https://arxiv.org/abs/2503.19157)]

This release includes the original three-stage training pipeline, motion-token
export, and evaluation code, migrated into the `hoigpt` Python package. It is **not
tokenizer-only**. Datasets, model weights, MANO assets, and experiment outputs are
not bundled. Training starts from prepared HOI features. Checkpoint-free ARCTIC/GRAB
feature converters, caption processing, normalization, and object-cache tools are
now included; see the [resource preparation guide](docs/resources.md).
The exact paper raw-to-clip manifest and all official split provenance are not yet
verified. Historical object point indices are included, but rebuilt normals are
not equivalent to the old cache. Training of the external PointNet/evaluation
models is not included.

## Code layout

```text
train.py                       Original Lightning training entry point
test.py                        Original generation/reconstruction evaluation
scripts/get_motion_code.py     Export trained tokenizer outputs
scripts/prepare_data.py        Checkpoint-free data/resource preparation
hoigpt/
  archs/                       Training VQ tokenizer, language model, evaluators
  models/                      HOIGPT model and stage-specific training logic
  data/                        ARCTIC/GRAB data modules and token datasets
  losses/                      Reconstruction, commitment, and geometry losses
  metrics/                     Retrieval, FID, diversity, and caption metrics
  lib/                         Required MANO, object, PointNet, geometry utilities
  tokenizer.py                 Lightweight standalone tokenizer API
  preprocessing/               Raw clip conversion, statistics, object caches
configs/                       Three-stage ARCTIC/GRAB configurations
assets/                        Original pretraining/instruction templates
  resources/                   Labeled local ARCTIC snapshot and manifest examples
docs/resources.md              Preparation commands, provenance, missing resources
```

The training architecture is `hoigpt.archs.hoigpt_vq.VQVae`; the Lightning model is
`hoigpt.models.hoigpt.HOIGPT`. The lightweight `from hoigpt import HOITokenizer`
API is also retained, but does not replace the training architecture. The release
does not depend on a top-level `mGPT`, `hoitokenizer`, or `lib` directory.

## Installation

Run commands from the repository root. Use Python 3.10 for the training environment.
Install PyTorch with CUDA support and a matching PyTorch3D build first; PyTorch3D
is required by the original geometry pipeline and is not automatically installed
by the package extra. Then install the training dependencies:

```bash
python -m pip install -e '.[train]'
```

`python -m pip install -r requirements.txt` installs the same extra. The lightweight
tokenizer alone can be installed with `python -m pip install .` and only depends
on PyTorch. Training commands require the source checkout (or unpacked source
distribution), including its root scripts, configs, and templates; a wheel alone
is not the training workspace.

Motion-to-text caption metrics additionally require:

```bash
python -m pip install -e '.[train,caption-eval]'
python -m spacy download en_core_web_sm
```

The spaCy model is also needed if `DATASET.ARCTIC.STD_TEXT` or
`DATASET.GRAB.STD_TEXT` is enabled. The supplied configs leave this disabled.

## Data and external assets

Start with the [resource preparation guide](docs/resources.md) for raw-data
conversion and existing split/normalization metadata. The included ARCTIC snapshot
has 5,423 training and 111 validation IDs, but no test list; it is explicitly a
local prepared-data snapshot, not a certified paper split. Do not combine its IDs
with a newly numbered raw-data manifest. GRAB's original clip/split manifest still
needs to be recovered. This update provides preparation code and metadata, not
raw datasets or downloadable prepared motions. Obtain the licensed data separately.

Data preparation tools are available through:

```bash
python -m pip install -e '.[preprocess]'
python scripts/prepare_data.py --help
```

See the guide for raw clip conversion, caption processing, normalization,
snapshot restoration, and object-cache generation. For model weights, see
[Checkpoint status](#checkpoint-status).

Set local paths in [`configs/assets.yaml`](configs/assets.yaml), or override any
configuration value with repeated `--set KEY=VALUE` arguments.

| Asset | Configuration/default location | Used by |
| --- | --- | --- |
| Prepared ARCTIC data | `DATASET.ARCTIC.ROOT`: `dataset/hoigen` | ARCTIC stages and evaluation |
| Prepared GRAB data | `DATASET.GRAB.ROOT`: `dataset/hoigrab` | GRAB stages and evaluation |
| Frozen PointNet checkpoint | `POINTNET.CHECKPOINT`: `checkpoints/arctic/pointfeat.pth` | Object-conditioned tokenizer |
| Pretrained FLAN-T5-base and tokenizer | `LANGUAGE_MODEL.PATH`: `deps/flan-t5-base` | Stages 2/3 and LM evaluation |
| GloVe vectorizer files | `DATASET.WORD_VERTILIZER_PATH`: `deps/glove` | Original data modules, including Stage 1 |
| MANO left/right model files | `MANO.ROOT`: `data/mano/mano_v1_2/models` | Geometry losses and rendering |
| HOI text-motion evaluator | `METRIC.TM2T_CHECKPOINT` (explicit file override) | FID, retrieval, diversity, multimodality |

Without an explicit evaluator override, the checkpoint is loaded from
`checkpoints/{arctic,grab}/text_mot_match/model/finest.tar`. It must match the
208-dimensional HOI representation, not a HumanML3D body-motion evaluator.
GloVe requires `our_vab_data.npy`, `our_vab_words.pkl`, and `our_vab_idx.pkl`.
PointNet checkpoints may contain a `model` or `state_dict` mapping, or a raw state
dictionary. Obtain MANO and datasets separately under their respective licenses.

Each prepared dataset root has this layout:

```text
dataset/hoigen/                 # or dataset/hoigrab/
  train.txt
  val.txt
  test.txt                     # required for test evaluation; not bundled
  mean.npy                     # [208]
  std.npy                      # [208], nonzero normalization scales
  new_joints/<name>.npy         # [T, 208] unnormalized HOI features
  texts/<name>.txt              # caption#word/POS word/POS#start_seconds#end_seconds
  arctic.pkl                   # grab.pkl for GRAB
  meshes/<object>/mesh.obj      # ARCTIC: also <object>/parts.json
  TOKENS_PAPER/<name>.npz       # generated after Stage 1
```

GRAB meshes instead use `meshes/<object>.ply`. ARCTIC mesh vertices are converted
from millimeters to meters by the loader. Split files contain sequence names without
extensions; the second underscore-separated name component identifies the object
(for example `sequence_box_take`). A full-sequence caption uses `#0#0` timestamps.

The object cache contains `object_name`, `obj_pcs`, `obj_pc_normals`, `point_sets`,
and `obj_path`; ARCTIC also uses `obj_pc_top`. These must correspond to the meshes
and object names in the split files. Use trusted caches: the original data loader
uses Python pickle to read them.

The feature layout is left hand `0:99`, right hand `99:198`, and object `198:208`.
The tokenizer shares one hand codebook and uses a separate object codebook. Object
motion is decoded and re-encoded to condition the hand decoders, together with
PointNet object features. The supplied configs use 512 entries per codebook and
temporal downsampling by 4. Token export writes aligned `left`, `right`, and `obj`
integer arrays, plus `length` and `format`, to `.npz` files.

Use the original prepared-data frame convention: the motion loaders and token
exporter apply stride 4 to sequences longer than 400 frames. Do not independently
resample only one stage. The language-model context length is configurable through
`lm.default.params.max_length`; structured triplets take six language-model tokens
per VQ timestep plus boundary tokens, so longer data may need a larger value.

## Checkpoint status

**Checkpoints: to be tuned.** Pretrained HOIGPT checkpoints, including the Stage 1
tokenizer and Stage 3 language model, are not included in this update. Their
release is pending tuning and validation; no download links are available yet.
The matching PointNet and text-motion evaluator weights are also pending release.
The commands below describe the training/evaluation workflow, but require the
corresponding external assets; the current repository alone is not a complete
paper-reproduction bundle.

## Training

The commands below use ARCTIC. For GRAB, replace `config_hoi_paper_stageN.yaml`
with `config_grab_paper_stageN.yaml` and supply the corresponding data, object
encoder, and evaluator assets. Replace `path/to/...` placeholders with actual
checkpoints. The configs retain the original training hyperparameters; reduce
`--batch_size` if needed for your GPU, understanding that this changes the run.

### Stage 1: HOI tokenizer

```bash
python train.py --cfg configs/config_hoi_paper_stage1.yaml --device 0 --nodebug
```

This trains the original VQ tokenizer with reconstruction, commitment, and HOI
geometry losses. It does not load the language model. The PointNet encoder is
pretrained and frozen; its checkpoint is required. Use a single GPU for tokenizer
training: the original EMA codebooks do not synchronize statistics across workers.

Checkpoints are written to
`experiments/hoigpt/Paper_ARCTIC_Stage1/checkpoints/`. The latest saved checkpoint
is `last.ckpt`. The default checkpoint/validation interval is 10 epochs.

### Export training tokens

```bash
python scripts/get_motion_code.py \
  --cfg configs/config_hoi_paper_stage1.yaml --device 0 \
  --set TRAIN.PRETRAINED_VAE=path/to/stage1.ckpt \
  --set TRAIN.SPLIT=train
```

This writes `dataset/hoigen/TOKENS_PAPER/*.npz`. The export split is
`TRAIN.SPLIT`, not `TEST.SPLIT`. Repeat with a different split if you intend to
train on that split. Validation/test motions are encoded or decoded online and
do not require pre-exported validation/test tokens.

### Stage 2: language-model pretraining

```bash
python train.py --cfg configs/config_hoi_paper_stage2.yaml --device 0 --nodebug \
  --set TRAIN.PRETRAINED_VAE=path/to/stage1.ckpt
```

This loads the pretrained language model, adds the structured HOI token vocabulary,
and trains with the frozen tokenizer. The exported training tokens and Stage 1
checkpoint must come from the same tokenizer. The original pretraining templates
are included in `assets/template_pretrain.json`.

### Stage 3: instruction tuning

```bash
python train.py --cfg configs/config_hoi_paper_stage3.yaml --device 0 --nodebug \
  --set TRAIN.PRETRAINED=path/to/stage2.ckpt \
  --set TRAIN.PRETRAINED_VAE=path/to/stage1.ckpt \
  --set 'LOGGER.TYPE=[tensorboard]'
```

This starts from Stage 2 and uses the original
`assets/template_instructions.json`. Dataset-local templates take precedence;
`DATASET.TASK_PATH` can explicitly select a template file. Some original configs
enable W&B; use the logger override above for local TensorBoard logging without
a W&B account.

To resume an interrupted run from this release, add
`--set TRAIN.RESUME=path/to/experiment` or `--set TRAIN.RESUME=path/to/last.ckpt`.
For historical Stage 1 checkpoints, use `TRAIN.PRETRAINED_VAE` to warm-start:
removing the unused language model changes optimizer parameter groups, so an old
full optimizer state is not guaranteed to resume unchanged. Old codebook-only
EMA checkpoints are upgraded when loading; new checkpoints persist EMA state.

## Evaluation and generated results

`test.py` is the original model evaluation entry point, not a unit-test suite.
Evaluate a trained Stage 3 checkpoint with:

```bash
python test.py --cfg configs/config_hoi_paper_stage3.yaml --device 0 \
  --set TEST.CHECKPOINTS=path/to/stage3.ckpt \
  --set TEST.SPLIT=test \
  --set TEST.REPLICATION_TIMES=20
```

The full Stage 3 checkpoint includes the tokenizer weights. The initial
PointNet/language-model assets are still needed to construct the architecture.
`TM2TMetrics` uses the HOI evaluator for FID, text-motion matching, retrieval, and
diversity; the script also runs multimodality evaluation and reports replication
means and confidence intervals. Use the full evaluation split: these metrics have
minimum sample requirements and are not meaningful on a tiny debugging subset.

Evaluate tokenizer reconstruction using the Stage 1 configuration:

```bash
python test.py --cfg configs/config_hoi_paper_stage1.yaml --device 0 \
  --set TEST.CHECKPOINTS=path/to/stage1.ckpt \
  --set TEST.SPLIT=test \
  --set 'METRIC.TYPE=[TM2TMetrics]'
```

For motion-to-text evaluation, use the Stage 3 checkpoint with `--task m2t` and
`--set 'METRIC.TYPE=[M2TMetrics]'`, after installing the caption-evaluation extra.

Metrics are saved under `results/hoigpt/<experiment>/`. With
`TEST.SAVE_PREDICTIONS=true`, motion outputs are saved as denormalized `.npy`
features, or captions as JSON for motion-to-text. Optional
`TEST.RENDER_PREDICTIONS=true` also renders motions and requires the MANO/mesh
assets and rendering dependencies.

## Release notes

This is a source-code release, not a bundle of trained weights or preprocessed
datasets. Reproducing paper scores requires the matching data preparation,
checkpoints, and evaluation assets. Packaging and entry-point/configuration checks
do not establish reproduced paper metrics; no full training or benchmark run is
claimed for this release preparation.

The published files are allowlisted in `.gitignore` and `MANIFEST.in`. Old local
research directories, private machine-specific configs, datasets, MANO files,
checkpoints, and experiment logs are excluded. Historical local directories may
still exist in a development workspace; they are neither imported nor released.

Only load trusted training checkpoints: the original Lightning checkpoints include
pickled training metadata and are loaded with `weights_only=False`. For sharing
standalone tokenizer weights, prefer a tensor-only state dictionary and
`HOITokenizer.load_checkpoint_state(...)`.

## Citation

```bibtex
@InProceedings{Huang_2025_CVPR,
    author    = {Huang, Mingzhen and Chu, Fu-Jen and Tekin, Bugra and Liang, Kevin J. and Ma, Haoyu and Wang, Weiyao and Chen, Xingyu and Gleize, Pierre and Xue, Hongfei and Lyu, Siwei and Kitani, Kris and Feiszli, Matt and Tang, Hao},
    title     = {HOIGPT: Learning Long-Sequence Hand-Object Interaction with Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025}
}
```

## License and acknowledgments

HOIGPT project code is released under the MIT license, with third-party portions
retaining their respective licenses. The implementation builds on MotionGPT,
T2M-GPT, Text2HOI, pointnet.pytorch, and the geometry utilities credited in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Full notices are included in
[`LICENSES/`](LICENSES/). Dataset and MANO licenses are separate from the code license.
