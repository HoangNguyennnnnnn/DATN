"""FaceDiff Gradio Web UI — Image → 3D Face Mesh.

Run:
    python app.py

Mở http://localhost:7860 trong browser.

Pipeline:
    1. ImagePreprocessor → context [946]
    2. VoxelMamba + iMF sample (n-step) → slat tokens
    3. Reverse normalize → slat_raw
    4. SC-VAE decode + Dual Contouring → mesh
    5. Save PLY → render trong Model3D viewer
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig
from src.data.image_preprocessor import ImagePreprocessor
from src.models.sc_vae import SC_VAE
from src.models.voxel_mamba import VoxelMamba
from scripts.test_e2e_inference import sample_n_step, save_ply, slat_to_mesh


# ============================================================
# Inference Engine — load models một lần, reuse cho mọi request
# ============================================================
class InferenceEngine:
    def __init__(
        self,
        imf_ckpt: str,
        scvae_ckpt: str,
        slat_stats_path: str,
        device: str = "cuda:0",
        decode_device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.decode_device = torch.device(decode_device)

        print(f"[Engine] device={self.device} decode_device={self.decode_device}")
        print("[Engine] Loading ImagePreprocessor...")
        self.preprocessor = ImagePreprocessor(device=str(self.device))

        print(f"[Engine] Loading iMF checkpoint: {imf_ckpt}")
        self.model, self.mcfg = self._load_imf(imf_ckpt)

        print(f"[Engine] Loading SC-VAE checkpoint: {scvae_ckpt}")
        self.sc_vae = self._load_scvae(scvae_ckpt)

        stats_path = self.mcfg.get("slat_stats_path") or slat_stats_path
        print(f"[Engine] Loading slat stats: {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        self.slat_mean = stats["mean"].to(self.device).view(1, 1, -1)
        self.slat_std = stats["std"].to(self.device).view(1, 1, -1)

        print("[Engine] Ready.")

    # --------------------------------------------------------
    def _load_imf(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mcfg = ckpt["stage2_model_config"]
        print(f"  [iMF] epoch={ckpt['epoch']} loss={ckpt['loss']:.4f}")
        model = VoxelMamba(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            slat_length=mcfg["slat_length"],
            context_dim=mcfg["context_dim"],
            backend=mcfg.get("backend", "auto"),
            num_context_tokens=mcfg.get("num_context_tokens", 8),
            num_time_tokens=mcfg.get("num_time_tokens", 4),
            num_r_tokens=mcfg.get("num_r_tokens", 4),
            num_interval_tokens=mcfg.get("num_interval_tokens", 4),
            num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
            d_state=mcfg.get("d_state", 16),
            d_conv=mcfg.get("d_conv", 4),
            expand=mcfg.get("expand", 2),
        ).to(self.device)
        state = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, mcfg

    def _load_scvae(self, ckpt_path: str) -> SC_VAE:
        cfg = TrainConfig()
        sc_vae = SC_VAE(
            in_channels=int(cfg.sc_vae.in_channels),
            latent_dim=int(cfg.sc_vae.latent_dim),
            num_res_blocks=int(cfg.sc_vae.num_res_blocks),
            encoder_dims=list(cfg.sc_vae.encoder_dims),
        ).to(self.decode_device)
        sc_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sc_state = sc_ckpt.get("model_state_dict", sc_ckpt)
        sc_state = {
            k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sc_state.items()
        }
        sc_vae.load_state_dict(sc_state, strict=False)
        sc_vae.eval()
        return sc_vae

    # --------------------------------------------------------
    @torch.no_grad()
    def process(
        self,
        front_path: str,
        back_path: str | None,
        n_steps: int,
        omega: float,
        progress=None,
    ) -> tuple[str, str, float]:
        """Returns (ply_path, status_msg, elapsed_seconds)."""
        t0 = time.time()

        if progress is not None:
            progress(0.05, desc="Extracting context (ArcFace + FLAME + DINOv2)...")
        back_arg = back_path if back_path and os.path.exists(back_path) else None
        ctx = self.preprocessor.process(front_path, back_image_path=back_arg)
        ctx = ctx.unsqueeze(0).to(self.device)

        if progress is not None:
            progress(0.4, desc=f"Sampling iMF ({n_steps}-step, ω={omega:.1f})...")
        slat_shape = (1, self.mcfg["slat_length"], self.mcfg["input_dim"])
        torch.manual_seed(42)  # Deterministic noise cho reproducibility
        with torch.autocast("cuda", dtype=torch.bfloat16):
            slat_norm = sample_n_step(
                self.model, ctx,
                shape=slat_shape,
                num_steps=n_steps,
                omega=omega,
            )
        slat_raw = slat_norm.float() * self.slat_std + self.slat_mean

        if progress is not None:
            progress(0.65, desc="Decoding mesh (SC-VAE + Dual Contouring)...")
        torch.cuda.empty_cache()
        verts, faces, colors, n_voxels = slat_to_mesh(slat_raw, self.sc_vae, self.decode_device)

        if progress is not None:
            progress(0.95, desc="Saving PLY...")
        tmp_dir = tempfile.mkdtemp(prefix="facediff_")
        ply_path = os.path.join(tmp_dir, "face_mesh.ply")
        ok = save_ply(verts, faces, colors, ply_path)
        if not ok:
            raise RuntimeError("save_ply returned False (empty mesh).")

        dt = time.time() - t0
        n_v = len(verts) if verts is not None else 0
        n_f = len(faces) if faces is not None else 0
        msg = (
            f"✓ Generated in {dt:.1f}s  |  {n_v:,} vertices  |  {n_f:,} faces  |  "
            f"{n_voxels:,} voxels  |  {n_steps}-step  ω={omega:.1f}"
        )
        return ply_path, msg, dt


# ============================================================
# Gradio UI
# ============================================================
def build_ui(engine: InferenceEngine) -> gr.Blocks:
    with gr.Blocks(title="FaceDiff: Image → 3D Face", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # FaceDiff — Image to 3D Face Generation
            Upload **1 ảnh frontal** (bắt buộc) + **1 ảnh back-of-head** (optional, để khôi phục hình dạng đầu phía sau)
            → nhấn **Generate** → xem mesh 3D + tải file `.ply`.

            Pipeline: ArcFace + FLAME blendshapes + DINOv2 → 946-dim context → VoxelMamba iMF (1-step generation)
            → SC-VAE decode → Dual Contouring → mesh với vertex colors.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                front_img = gr.Image(
                    type="filepath",
                    label="Frontal face (required)",
                    image_mode="RGB",
                    height=320,
                )
                back_img = gr.Image(
                    type="filepath",
                    label="Back of head (optional)",
                    image_mode="RGB",
                    height=320,
                )
                with gr.Accordion("Advanced settings", open=False):
                    steps_input = gr.Radio(
                        choices=[1, 5, 20],
                        value=1,
                        label="Sampling steps",
                        info="1 = paper iMF 1-NFE (~0.5s, fastest); "
                             "5 = balanced; 20 = highest quality multi-step Euler",
                    )
                    omega_input = gr.Slider(
                        minimum=1.0,
                        maximum=8.0,
                        value=4.0,
                        step=0.5,
                        label="CFG guidance scale (ω)",
                        info="1 = no guidance, 4 = balanced (default), 8 = strong identity",
                    )
                generate_btn = gr.Button("Generate 3D mesh", variant="primary", size="lg")
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Ready. Upload an image and click Generate.",
                    lines=2,
                )

            with gr.Column(scale=2):
                mesh_view = gr.Model3D(
                    label="3D Face Mesh",
                    clear_color=(0.1, 0.1, 0.1, 1.0),
                    height=600,
                )
                download_btn = gr.File(label="Download .ply", interactive=False)

        # Handler
        def on_generate(front, back, steps, omega, progress=gr.Progress()):
            if not front:
                return None, None, "Please upload a frontal face image first."
            try:
                ply_path, msg, _ = engine.process(
                    front_path=front,
                    back_path=back,
                    n_steps=int(steps),
                    omega=float(omega),
                    progress=progress,
                )
                return ply_path, ply_path, msg
            except ValueError as e:
                return None, None, f"{e}"
            except Exception as e:
                traceback.print_exc()
                return None, None, f"Error: {e}"

        generate_btn.click(
            on_generate,
            inputs=[front_img, back_img, steps_input, omega_input],
            outputs=[mesh_view, download_btn, status_box],
        )

    return demo


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf-ckpt", default="checkpoints/imf_unet/best.pt",
                    help="Path to iMF VoxelMamba checkpoint")
    ap.add_argument("--scvae-ckpt", default="checkpoints/sc_vae_shape/epoch_500.pt",
                    help="Path to SC-VAE checkpoint")
    ap.add_argument("--slat-stats", default="data/slat_stats.pt",
                    help="Path to slat normalization stats")
    ap.add_argument("--device", default="cuda:0",
                    help="Device for ImagePreprocessor + iMF (cuda:0 or cpu)")
    ap.add_argument("--decode-device", default="cpu",
                    help="Device for SC-VAE decode (default: cpu to avoid training OOM)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--server-name", default="127.0.0.1",
                    help="Bind address (127.0.0.1 = local only, 0.0.0.0 = LAN)")
    args = ap.parse_args()

    # Verify checkpoints exist
    for label, path in [("iMF", args.imf_ckpt), ("SC-VAE", args.scvae_ckpt),
                        ("slat-stats", args.slat_stats)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    engine = InferenceEngine(
        imf_ckpt=args.imf_ckpt,
        scvae_ckpt=args.scvae_ckpt,
        slat_stats_path=args.slat_stats,
        device=args.device,
        decode_device=args.decode_device,
    )

    demo = build_ui(engine)
    print(f"\n>>> FaceDiff UI ready: http://{args.server_name}:{args.port}\n")
    demo.queue(max_size=4).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=False,
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
