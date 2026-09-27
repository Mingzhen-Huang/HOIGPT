"""CPU-only strict loading check for the paired public inference checkpoints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--stage3", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pointnet-output", required=True)
    parser.add_argument("--flan-path", required=True)
    parser.add_argument("--report", help="Optional JSON report; stdout always emits the same JSON.")
    args = parser.parse_args()
    if args.report:
        args.report = str(Path(args.report).resolve())
    source = Path(args.source).resolve(strict=True)
    codebook_path = Path(args.codebook).resolve(strict=True)
    stage3_path = Path(args.stage3).resolve(strict=True)
    flan_path = Path(args.flan_path).resolve(strict=True)
    pointnet_path = Path(args.pointnet_output).resolve()
    assert (source / "hoigpt/archs/hoigpt_vq.py").is_file()
    assert (source / "hoigpt/archs/tools/quantize_cnn.py").is_file()
    assert (flan_path / "config.json").is_file()
    assert pointnet_path not in (codebook_path, stage3_path)
    sys.path.insert(0, str(source))
    os.chdir(source)
    import torch
    from omegaconf import OmegaConf
    from hoigpt.config import get_module_config, instantiate_from_config

    torch.set_num_threads(2)
    torch.manual_seed(1234)

    def read_weights(path):
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        assert isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict)
        forbidden = {"optimizer", "optimizer_states", "scheduler", "lr_schedulers",
                     "callbacks", "loops", "rng", "rank_rng_states", "hyper_parameters"}
        assert not (forbidden & payload.keys()), ("Not an inference-only checkpoint", sorted(forbidden & payload.keys()))
        state = payload["state_dict"]
        assert state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items())
        assert all(value.device.type == "cpu" for value in state.values())
        return payload, state

    codebook_payload, codebook_state = read_weights(codebook_path)
    stage3_payload, stage3_state = read_weights(stage3_path)
    assert all(key.startswith("vae.") for key in codebook_state), "Keep the published codebook's vae. prefix"
    vae_state = {key[4:]: value for key, value in codebook_state.items()}
    stage3_vae = {key[4:]: value for key, value in stage3_state.items() if key.startswith("vae.")}
    assert set(vae_state) == set(stage3_vae), "Stage 3 and standalone codebook tensor names differ"
    for key, value in vae_state.items():
        assert value.dtype == stage3_vae[key].dtype and torch.equal(value, stage3_vae[key]), "Tokenizer mismatch: " + key

    pointnet_state = {key[len("pointnet."):]: value for key, value in vae_state.items()
                      if key.startswith("pointnet.")}
    assert pointnet_state, "No embedded PointNet initialization weights"
    pointnet_path.parent.mkdir(parents=True, exist_ok=True)
    if pointnet_path.exists():
        existing = torch.load(str(pointnet_path), map_location="cpu", weights_only=True)
        existing = existing.get("state_dict", existing.get("model", existing))
        assert set(existing) == set(pointnet_state)
        assert all(torch.equal(existing[k], v) for k, v in pointnet_state.items()), "Existing PointNet output differs"
    else:
        # Clone only this small subset so torch.save cannot retain unrelated storages.
        temporary = pointnet_path.with_name(pointnet_path.name + ".tmp")
        assert not temporary.exists()
        torch.save({"state_dict": {key: value.detach().clone() for key, value in pointnet_state.items()}}, temporary)
        temporary.replace(pointnet_path)

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    configs = source / "configs"
    cfg = OmegaConf.merge(OmegaConf.load(configs / "default.yaml"),
                          OmegaConf.load(configs / "config_hoi_paper_stage3.yaml"))
    cfg = get_module_config(cfg, str(configs))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(configs / "assets.yaml"))
    cfg.LANGUAGE_MODEL.PATH = str(flan_path)
    cfg.POINTNET.CHECKPOINT = str(pointnet_path)
    cfg.DATASET.CODE_FORMAT = "hoi_triplet"
    assert cfg.TRAIN.STAGE == "lm_instruct"
    vae_config = OmegaConf.to_container(cfg.model.params.motion_vae, resolve=True)
    lm_config = OmegaConf.to_container(cfg.model.params.lm, resolve=True)
    assert vae_config["params"]["dual_codebook"] is True
    assert vae_config["params"]["dual_decoder"] is False
    assert vae_config["params"]["pointnet"] is True
    assert vae_config["params"]["code_num"] == 512
    assert vae_config["params"]["nfeats"] == 208
    assert lm_config["params"]["token_format"] == "hoi_triplet"
    assert lm_config["params"]["stage"] == "lm_instruct"

    vae = instantiate_from_config(vae_config).cpu().eval()
    assert set(vae.state_dict()) == set(vae_state), "VQVae source/state tensor names differ"
    vae_result = vae.load_state_dict(dict(vae_state), strict=True)
    assert not vae_result.missing_keys and not vae_result.unexpected_keys
    assert all(torch.equal(value, vae.state_dict()[key]) for key, value in vae_state.items())
    del vae

    lm_state = {key[3:]: value for key, value in stage3_state.items() if key.startswith("lm.")}
    assert lm_state, "Stage 3 lacks language-model weights"
    lm = instantiate_from_config(lm_config).cpu().eval()
    assert set(lm.state_dict()) == set(lm_state), "MLM source/state tensor names differ"
    lm_result = lm.load_state_dict(lm_state, strict=True)
    assert not lm_result.missing_keys and not lm_result.unexpected_keys
    loaded_lm_state = lm.state_dict()
    assert all(value.dtype == loaded_lm_state[key].dtype and torch.equal(value, loaded_lm_state[key])
               for key, value in lm_state.items()), "Loaded language-model tensor mismatch"
    tokenizer_length = len(lm.tokenizer)
    shared_shape = list(lm.language_model.shared.weight.shape)
    assert shared_shape[0] == tokenizer_length
    added_tokens = lm._build_added_tokens()
    added_token_ids = lm.tokenizer.convert_tokens_to_ids(added_tokens)
    assert len(set(added_token_ids)) == len(added_tokens) == 1030

    result = {
        "passed": True, "device": "cpu", "torch": str(torch.__version__),
        "weights_only_deserialization": True, "inference_only_not_exact_training_resume": True,
        "source": str(source),
        "codebook": {"path": str(codebook_path), "sha256": sha256(codebook_path), "state_keys": len(codebook_state)},
        "stage3": {"path": str(stage3_path), "sha256": sha256(stage3_path), "state_keys": len(stage3_state)},
        "vae_strict_load": True, "lm_strict_load": True, "all_loaded_tensors_equal": True,
        "stage3_embedded_vae_equals_codebook": True,
        "vae_tensor_count": len(vae_state), "lm_tensor_count": len(lm_state),
        "preserved_evaluator_tensor_count": sum(key.startswith("metrics.") for key in stage3_state),
        "pointnet": {"path": str(pointnet_path), "sha256": sha256(pointnet_path), "state_keys": len(pointnet_state)},
        "vae_configuration": vae_config, "lm_configuration": lm_config,
        "tokenizer_length": tokenizer_length, "shared_embedding_shape": shared_shape,
        "added_tokens": added_tokens, "added_token_ids": added_token_ids,
        "checks_excluded": ["full FID evaluation", "motion generation", "training", "dataset loading"]
    }
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
