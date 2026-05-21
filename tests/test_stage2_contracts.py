import os
import tempfile
import unittest
from unittest import mock

import torch

from src.inference.generator import FaceDiffGenerator
from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.voxel_mamba import VoxelMamba
from src.train_imf import EMA, SlatDataset, load_checkpoint, save_checkpoint


class DummyOVoxelConverter:
    def __init__(self, *args, **kwargs):
        pass

    def process_mesh(self, obj_path: str):
        feats = torch.zeros((4, 10), dtype=torch.float32)
        coords = torch.zeros((4, 3), dtype=torch.int32)
        return {"shape_mat_features": feats, "coords": coords}


class Stage2ContractTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_v_head(self):
        torch.manual_seed(0)
        model = VoxelMamba(
            input_dim=4,
            hidden_dim=8,
            num_layers=1,
            slat_length=8,
            context_dim=3,
            backend="gru",
            num_context_tokens=0,
            num_time_tokens=0,
            num_r_tokens=0,
            num_interval_tokens=0,
            num_guidance_tokens=0,
            use_hilbert_ordering=False,
        )
        v_head = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 4),
        )
        ctx_classifier = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 3),
        )
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(v_head.parameters()) + list(ctx_classifier.parameters()),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = EMA(model, decay=0.95)
        model_cfg = {"arch": "voxel_mamba", "input_dim": 4, "hidden_dim": 8}

        original = v_head[-1].weight.detach().clone()
        original_ctx = ctx_classifier[-1].weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "stage2.pt")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                epoch=3,
                loss=0.25,
                path=ckpt_path,
                v_head=v_head,
                ctx_classifier=ctx_classifier,
                stage2_model_config=model_cfg,
                global_step=17,
                best_loss=0.2,
            )

            with torch.no_grad():
                v_head[-1].weight.fill_(123.0)
                ctx_classifier[-1].weight.fill_(456.0)

            info = load_checkpoint(
                ckpt_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                v_head=v_head,
                ctx_classifier=ctx_classifier,
            )

        self.assertTrue(info["resumed_full"])
        self.assertEqual(info["epoch"], 3)
        self.assertEqual(info["global_step"], 17)
        self.assertAlmostEqual(info["best_loss"], 0.2)
        self.assertTrue(torch.allclose(v_head[-1].weight.detach(), original))
        self.assertTrue(torch.allclose(ctx_classifier[-1].weight.detach(), original_ctx))

    def test_imf_mixed_batch_main_loss_backprops_without_aux_heads(self):
        torch.manual_seed(2)
        model = VoxelMamba(
            input_dim=4,
            hidden_dim=16,
            num_layers=1,
            slat_length=8,
            context_dim=3,
            backend="gru",
            num_context_tokens=0,
            num_time_tokens=0,
            num_r_tokens=0,
            num_interval_tokens=0,
            num_guidance_tokens=0,
            use_hilbert_ordering=False,
        )
        imf = ImprovedMeanFlow(ratio_r_neq_t=0.5, t_sampler="uniform")

        def fixed_t_r(batch_size, device):
            t = torch.tensor([0.25, 0.50, 0.75, 0.90], device=device)
            r = torch.tensor([0.25, 0.10, 0.75, 0.20], device=device)
            return t[:batch_size], r[:batch_size]

        imf._sample_t_r = fixed_t_r
        x = torch.randn(4, 8, 4)
        context = torch.randn(4, 3)
        loss_out = imf.compute_loss(model, x, context, return_components=True)
        loss = loss_out["loss"]

        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(model.output_proj.weight.grad)
        self.assertGreater(float(model.output_proj.weight.grad.norm()), 0.0)

    def test_old_checkpoint_auto_downgrades_to_model_only(self):
        torch.manual_seed(1)
        model = VoxelMamba(
            input_dim=4,
            hidden_dim=8,
            num_layers=1,
            slat_length=8,
            context_dim=3,
            backend="gru",
            num_context_tokens=0,
            num_time_tokens=0,
            num_r_tokens=0,
            num_interval_tokens=0,
            num_guidance_tokens=0,
            use_hilbert_ordering=False,
        )
        v_head = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 4),
        )
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(v_head.parameters()), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = EMA(model, decay=0.95)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "old_stage2.pt")
            torch.save(
                {
                    "epoch": 5,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "loss": 0.5,
                },
                ckpt_path,
            )
            info = load_checkpoint(
                ckpt_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                v_head=v_head,
            )

        self.assertFalse(info["resumed_full"])
        self.assertEqual(info["epoch"], 5)


    def test_generator_infers_missing_optional_tokenizers_as_zero(self):
        model = VoxelMamba(
            input_dim=4,
            hidden_dim=8,
            num_layers=1,
            slat_length=8,
            context_dim=3,
            backend="gru",
            num_context_tokens=0,
            num_time_tokens=0,
            num_r_tokens=0,
            num_interval_tokens=0,
            num_guidance_tokens=0,
            use_hilbert_ordering=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "old_vm.pt")
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            generator = FaceDiffGenerator.__new__(FaceDiffGenerator)
            generator.slat_length = 8
            cfg = generator._infer_stage2_model_config(ckpt_path, default_input_dim=4, default_context_dim=3)
            rebuilt = generator._build_stage2_model(cfg)
            generator._load_imf_checkpoint(rebuilt, ckpt_path)

        self.assertEqual(cfg["num_r_tokens"], 0)
        self.assertEqual(cfg["num_interval_tokens"], 0)
        self.assertEqual(cfg["num_guidance_tokens"], 0)
        self.assertIsInstance(rebuilt, VoxelMamba)

    def test_generator_infers_two_layer_context_tokenizer_width(self):
        model = VoxelMamba(
            input_dim=4,
            hidden_dim=8,
            num_layers=1,
            slat_length=8,
            context_dim=3,
            backend="gru",
            num_context_tokens=3,
            num_time_tokens=1,
            num_r_tokens=1,
            num_interval_tokens=1,
            num_guidance_tokens=1,
            use_hilbert_ordering=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "new_vm_no_config.pt")
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            generator = FaceDiffGenerator.__new__(FaceDiffGenerator)
            generator.slat_length = 8
            cfg = generator._infer_stage2_model_config(ckpt_path, default_input_dim=4, default_context_dim=3)
            rebuilt = generator._build_stage2_model(cfg)
            generator._load_imf_checkpoint(rebuilt, ckpt_path)

        self.assertEqual(cfg["num_context_tokens"], 3)
        self.assertEqual(cfg["num_time_tokens"], 1)
        self.assertEqual(cfg["num_r_tokens"], 1)
        self.assertEqual(cfg["num_interval_tokens"], 1)
        self.assertEqual(cfg["num_guidance_tokens"], 1)

    def test_slat_dataset_is_fail_fast_without_debug_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "sample.obj")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("# dummy obj\n")

            with mock.patch("src.train_imf.OVoxelConverter", side_effect=RuntimeError("converter boom")):
                with self.assertRaises(RuntimeError):
                    SlatDataset(
                        data_root=tmpdir,
                        sc_vae=object(),
                        dataset_name="faceverse",
                        allow_mesh_proxy_fallback=False,
                    )

            with mock.patch("src.train_imf.OVoxelConverter", DummyOVoxelConverter):
                dataset = SlatDataset(
                    data_root=tmpdir,
                    sc_vae=mock.Mock(in_channels=10),
                    dataset_name="faceverse",
                    cache_dir=os.path.join(tmpdir, "cache"),
                    allow_random_context_fallback=False,
                    allow_mesh_proxy_fallback=False,
                )
                dataset._encode_latents = mock.Mock(return_value=torch.zeros((4, 4), dtype=torch.float32))
                with self.assertRaises(RuntimeError):
                    dataset[0]


if __name__ == "__main__":
    unittest.main()
