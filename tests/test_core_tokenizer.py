from __future__ import annotations

import unittest

import torch
from torch import nn

import hoigpt
from hoigpt import EMACodebook, HOIQuantizer, HOITokenizer, PointNetEncoder, VQVae


class DummyPointEncoder(nn.Module):
    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.projection(points.mean(dim=1)))


def make_tokenizer(*, point_conditioning: bool = True) -> HOITokenizer:
    return HOITokenizer(
        codebook_size=8,
        latent_dim=8,
        downsample_layers=1,
        width=8,
        residual_depth=1,
        dilation_growth_rate=2,
        point_conditioning=point_conditioning,
        condition_dim=4,
        point_encoder=DummyPointEncoder() if point_conditioning else None,
    )


class PublicApiTests(unittest.TestCase):
    def test_public_api(self) -> None:
        self.assertEqual(hoigpt.__version__, "0.1.0")
        self.assertIs(hoigpt.HOITokenizer, HOITokenizer)

    def test_pointnet_shape_and_validation(self) -> None:
        encoder = PointNetEncoder().eval()
        points = torch.randn(2, 16, 3)
        with torch.no_grad():
            encoded = encoder(points)
        self.assertEqual(encoded.shape, (2, 1024))
        with self.assertRaisesRegex(ValueError, "expects"):
            encoder(torch.randn(2, 3, 16))


class CodebookTests(unittest.TestCase):
    def test_legacy_codebook_state_is_upgraded(self) -> None:
        source = EMACodebook(size=4, dimension=3)
        source.initialize(torch.randn(8, 3))
        legacy_state = {"codebook": source.codebook.detach().clone()}

        restored = EMACodebook(size=4, dimension=3)
        restored.load_state_dict(legacy_state, strict=True)
        self.assertTrue(bool(restored.initialized.item()))
        torch.testing.assert_close(restored.code_sum, restored.codebook)
        torch.testing.assert_close(restored.code_count, torch.ones_like(restored.code_count))

    def test_ema_resume_matches_uninterrupted_training(self) -> None:
        torch.manual_seed(1)
        original = HOIQuantizer(codebook_size=4, code_dim=4, decay=0.9).train()
        first = [torch.randn(2, 4, 3) for _ in range(3)]
        original(*first)
        saved = {key: value.detach().clone() for key, value in original.state_dict().items()}

        restored = HOIQuantizer(codebook_size=4, code_dim=4, decay=0.9).train()
        restored.load_state_dict(saved, strict=True)
        second = [torch.randn(2, 4, 3) for _ in range(3)]
        original_outputs = original(*second)
        restored_outputs = restored(*second)

        for left, right in zip(original_outputs, restored_outputs):
            torch.testing.assert_close(left, right)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])

    def test_shared_hand_update_is_order_invariant(self) -> None:
        torch.manual_seed(2)
        baseline = HOIQuantizer(codebook_size=4, code_dim=4, decay=0.9).train()
        baseline(*[torch.randn(2, 4, 3) for _ in range(3)])
        state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}

        normal = HOIQuantizer(codebook_size=4, code_dim=4, decay=0.9).train()
        swapped = HOIQuantizer(codebook_size=4, code_dim=4, decay=0.9).train()
        normal.load_state_dict(state)
        swapped.load_state_dict(state)
        left, right, obj = [torch.randn(2, 4, 3) for _ in range(3)]
        normal(left, right, obj)
        swapped(right, left, obj)
        torch.testing.assert_close(
            normal.quantizer_hands.codebook,
            swapped.quantizer_hands.codebook,
        )


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.features = torch.randn(2, 8, 208)
        self.points = torch.randn(2, 16, 3)

    def test_round_trip_shapes_and_frozen_point_encoder(self) -> None:
        model = make_tokenizer().train()
        self.assertFalse(model.pointnet.training)

        reconstructed, commitment_loss, perplexity = model(self.features, self.points)
        self.assertEqual(reconstructed.shape, self.features.shape)
        self.assertEqual(commitment_loss.ndim, 0)
        self.assertEqual(perplexity.ndim, 0)

        model.eval()
        tokens = model.encode(self.features, self.points)
        self.assertEqual(set(tokens), {"left", "right", "obj"})
        self.assertEqual(tokens["left"].shape, (2, 4))
        decoded = model.decode(tokens, self.points)
        self.assertEqual(decoded.shape, self.features.shape)

    def test_cpu_autocast_accepts_point_encoder_output_dtype(self) -> None:
        model = make_tokenizer().train()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            reconstructed, commitment_loss, perplexity = model(
                self.features,
                self.points,
            )
        self.assertEqual(reconstructed.shape, self.features.shape)
        self.assertEqual(commitment_loss.ndim, 0)
        self.assertEqual(perplexity.ndim, 0)

    def test_without_point_conditioning(self) -> None:
        model = make_tokenizer(point_conditioning=False).train()
        reconstructed, _, _ = model(self.features)
        self.assertEqual(reconstructed.shape, self.features.shape)
        model.eval()
        tokens = model.encode(self.features)
        self.assertEqual(model.decode(tokens).shape, self.features.shape)
        with self.assertRaisesRegex(ValueError, "without point conditioning"):
            model(self.features, self.points)

    def test_paper_downsampling_default_accepts_196_frames(self) -> None:
        model = HOITokenizer(
            codebook_size=8,
            latent_dim=8,
            width=8,
            residual_depth=0,
            point_conditioning=False,
        ).train()
        features = torch.randn(1, 196, 208)
        reconstructed, _, _ = model(features)
        self.assertEqual(model.downsample_factor, 4)
        self.assertEqual(reconstructed.shape, features.shape)

        compatibility_model = VQVae(
            code_num=8,
            code_dim=8,
            output_emb_width=8,
            width=8,
            depth=0,
            pointnet=False,
        )
        self.assertEqual(compatibility_model.downsample_factor, 4)

    def test_input_validation(self) -> None:
        model = make_tokenizer().train()
        with self.assertRaisesRegex(ValueError, "208"):
            model(torch.randn(2, 8, 207), self.points)
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(torch.randn(2, 9, 208), self.points)
        with self.assertRaisesRegex(ValueError, "required"):
            model(self.features)
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            model(self.features, torch.randn(1, 16, 3))
        with self.assertRaisesRegex(ValueError, "default PointNet"):
            HOITokenizer(condition_dim=4)
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            HOITokenizer(point_encoder=object())

    def test_token_validation(self) -> None:
        model = make_tokenizer().train()
        model(self.features, self.points)
        model.eval()
        tokens = model.encode(self.features, self.points)

        with self.assertRaisesRegex(ValueError, "exactly"):
            model.decode({"left": tokens["left"], "right": tokens["right"]}, self.points)
        bad_shape = dict(tokens)
        bad_shape["obj"] = tokens["obj"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            model.decode(bad_shape, self.points)
        bad_dtype = dict(tokens)
        bad_dtype["left"] = tokens["left"].float()
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            model.decode(bad_dtype, self.points)
        out_of_range = dict(tokens)
        out_of_range["obj"] = torch.full_like(tokens["obj"], 8)
        with self.assertRaisesRegex(ValueError, "must be in"):
            model.decode(out_of_range, self.points)

    def test_checkpoint_prefix_loading(self) -> None:
        source = make_tokenizer().train()
        source(self.features, self.points)
        prefixed = {
            "state_dict": {
                f"vae.{key}": value.detach().clone()
                for key, value in source.state_dict().items()
            }
        }
        restored = make_tokenizer()
        incompatible = restored.load_checkpoint_state(prefixed, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])

    def test_checkpoint_strictness_preserves_prefixed_unknown_keys(self) -> None:
        source = make_tokenizer().train()
        source(self.features, self.points)
        prefixed = {
            f"vae.{key}": value.detach().clone()
            for key, value in source.state_dict().items()
        }
        prefixed["vae.typo_parameter"] = torch.tensor(1.0)

        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            make_tokenizer().load_checkpoint_state(
                {"state_dict": prefixed},
                strict=True,
            )
        incompatible = make_tokenizer().load_checkpoint_state(
            {"state_dict": prefixed},
            strict=False,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, ["typo_parameter"])

    def test_historical_checkpoint_without_ema_accumulators(self) -> None:
        source = make_tokenizer().train()
        source(self.features, self.points)
        historical_state = {
            f"vae.{key}": value.detach().clone()
            for key, value in source.state_dict().items()
            if not key.endswith(("code_sum", "code_count", "initialized"))
        }

        restored = make_tokenizer()
        incompatible = restored.load_checkpoint_state(
            {"state_dict": historical_state},
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(
            restored.quantizer.quantizer_hands.codebook,
            source.quantizer.quantizer_hands.codebook,
        )
        torch.testing.assert_close(
            restored.quantizer.quantizer_obj.codebook,
            source.quantizer.quantizer_obj.codebook,
        )
        self.assertTrue(bool(restored.quantizer.quantizer_hands.initialized.item()))
        self.assertTrue(bool(restored.quantizer.quantizer_obj.initialized.item()))

    def test_vqvae_compatibility_adapter(self) -> None:
        model = VQVae(
            nfeats=208,
            code_num=8,
            code_dim=8,
            output_emb_width=8,
            down_t=1,
            width=8,
            depth=1,
            norm="None",
            pointnet=True,
            cond_dim=4,
            point_encoder=DummyPointEncoder(),
            ablation={"name": "paper"},
            mae_mask_ratio=0.5,
        ).train()
        model(self.features, pc=self.points)
        tokens, distribution = model.encode(self.features, self.points)
        self.assertIsNone(distribution)
        self.assertEqual(model.decode(tokens, self.points).shape, self.features.shape)

        single_tokens = {key: value[0] for key, value in tokens.items()}
        decoded_single = model.decode(single_tokens, pc=self.points[:1])
        self.assertEqual(decoded_single.shape, self.features[:1].shape)

        token_tuple = (
            single_tokens["left"],
            single_tokens["right"],
            single_tokens["obj"],
        )
        decoded_tuple = model.decode(token_tuple, pc=self.points[:1])
        torch.testing.assert_close(decoded_tuple, decoded_single)


if __name__ == "__main__":
    unittest.main()
