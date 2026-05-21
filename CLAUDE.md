# FaceDiff — Single-GPU 3D Face Generation

## Project Overview

FaceDiff generates high-quality 3D face meshes (200K+ vertices, 10-channel O-Voxel) on a single RTX 4090 using a 3-stage pipeline based on TRELLIS.2 architecture adapted for faces.

## Architecture (3 Stages)

1. **SC-VAE** (35M params) — Sparse Convolution VAE. Compresses O-Voxel (256^3 grid, 10ch) to Slat tokens [4096, 32]. Uses spconv for sparse 3D convolutions.
2. **VoxelMamba + iMF** (~94M backbone + ~17M v-head + optional contrastive) — 12× BiMamba+FFN blocks, dual AdaLN conditioning (context + time), Hilbert ordering. Generates Slat via Improved Mean Flow (1-step). **No prefix tokens** (conditioning via AdaLN broadcast to all 4096 positions).
3. **Decode** — SC-VAE decoder + Dual Contouring → polygon mesh with vertex colors.

## Key Data Formats

- **O-Voxel 10ch:** `dv(0:3) + delta(3:6) + gamma(6:7) + rgb(7:10)` — sparse voxel features
- **Hybrid Context 946-dim:** `ArcFace(512) + FLAME(50) + DINOv2(384)` — identity + expression + back geometry
- **Slat tokens:** `[4096, 32]` float32 — compressed latent representation per mesh

## Directory Structure

```
src/
  config.py              # TrainConfig dataclass (all hyperparameters)
  train_sc_vae.py        # Stage 1 training loop
  train_imf.py           # Stage 2 training loop + SlatDataset
  train_structure.py     # Stage 3 (structure generator, legacy)
  hilbert.py             # Hilbert curve utilities
  utils.py               # Shared utilities
  models/
    sc_vae.py            # SC-VAE encoder/decoder (SparseResMLPBlock)
    sc_vae_loss.py       # Loss functions: MSE(dv), BCE(delta), smooth_l1(gamma), L1(rgb)
    voxel_mamba.py       # VoxelMamba backbone (BiMamba + Hilbert ordering)
    imf_diffusion.py     # iMF loss, sampling, JVP correction
  data/
    ovoxel_converter.py  # Mesh → O-Voxel conversion (wraps third_party/TRELLIS.2/o-voxel)
    arcface_extractor.py # ArcFace identity extraction
    flame_adapter.py     # FLAME expression parameters
    feature_extractor.py # DINOv2 back-of-head features
    mesh_renderer.py     # nvdiffrast/PyTorch3D mesh rendering
  scvae_train/
    data.py              # VoxelDataset for SC-VAE training
    runtime.py           # Training runtime utilities
    render.py            # Point cloud splatting for stage2 render loss
    metrics.py           # TRELLIS.2-aligned metrics
scripts/
  precompute_slat_cache.py    # Precompute slat+context .pt files
  pack_slat_lmdb.py           # Convert .pt → merged LMDB
  pack_lmdb_fast.py           # Pack O-Voxel cache → LMDB
  build_context_lmdb.py       # Build hybrid context LMDB
  train_imf.sh                # Launch Stage 2 (nohup + LMDB auto-detect)
  test_pure_t1.py             # Memorization gate test (1 sample, t≈1)
  test_imf_identity_t0.py     # Multi-ID identity diagnostic at t=0
  test_imf_memorization.py    # Memorization / conditioning diagnostics
  eval_scvae_checkpoints.py   # Evaluate SC-VAE checkpoints
  test_sc_vae_recon_v2.py     # Reconstruction testing + mesh extraction
  generate_ovoxel_cache.py    # Generate O-Voxel cache
third_party/
  TRELLIS.2/                  # Reference implementation (o-voxel library, configs)
data/                         # (gitignored) LMDB caches, slat caches
checkpoints/                  # (gitignored) model checkpoints
```

## Environment

- **Python:** 3.11 (conda env `facediff`)
- **GPU:** RTX 4090 24GB, CUDA 12.x
- **Key deps:** PyTorch 2.x, spconv-cu12x, mamba-ssm 2.3.1, nvdiffrast, lpips, lmdb

Activate: `source miniconda3/etc/profile.d/conda.sh && conda activate facediff`

## Common Commands

```bash
# SC-VAE training
python src/train_sc_vae.py --resume checkpoints/sc_vae_shape/epoch_500.pt --epochs 700

# Precompute slat cache (offline iMF data)
python scripts/precompute_slat_cache.py \
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
  --dataset both --context-lmdb data/hybrid_context.lmdb --skip-existing

# Pack slat cache → LMDB
python scripts/pack_slat_lmdb.py

# iMF training (recommended: script + LMDB)
bash scripts/train_imf.sh
# Or manually:
python src/train_imf.py --offline-data \
  --slat-lmdb data/slat_context.lmdb \
  --context-lmdb data/hybrid_context.lmdb \
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
  --batch-size 4 --gradient-accumulation-steps 16 \
  --disable-cfg-conditioning --disable-id-filters

# Evaluate SC-VAE reconstruction
python scripts/test_sc_vae_recon_v2.py --checkpoint checkpoints/sc_vae_shape/epoch_500.pt
```

## Stage 2 Architecture (VoxelMamba v4, May 2026)

**Sequence:** `[4096 slat tokens]` only (Hilbert-ordered). **No** 24 prefix tokens (`mamba_num_*_tokens=0`).

**Per `BidirectionalMambaBlock` (×12):**
1. **Time AdaLN** → modulate pre-Mamba: `x' = norm(x)·(1+scale_t)+shift_t`
2. **BiMamba** — forward + backward Mamba scan, sum
3. **Context AdaLN** — modulate Mamba output: `out' = out·(1+scale_c)+shift_c`
4. **Gated residual:** `x = x + gate_t·out'` (`gate` bias init **1.0**, not 0)
5. **FFN** (expand×4, GELU) + **Time AdaLN** + gated residual

**Conditioning paths (separate, not concat-fusion):**
- `ctx_cond = context_cond_mlp(context)` → AdaLN_ctx (scale+shift)
- `time_cond = time_guidance_mlp(t, r, t−r, ω, tmin, tmax)` → AdaLN_time + AdaLN_ffn

**Init fixes (2026-05-21):** `output_proj` uses **Xavier gain=0.02** (NOT zero — zero starved backbone gradients). FFN last layer zero-init for safe residual.

**Training loss stack:** iMF velocity (boundary + JVP, 50/50) + v-head (weight 0.5, depth 8) + contrastive InfoNCE (weight 0.3, batch≥2). CFG **off** by default (`--disable-cfg-conditioning`). Slat **per-channel normalize** via `data/slat_stats.pt`.

**Checkpoint compatibility:** Old runs with `adaLN_modulation` / `cond_fusion` / 24-prefix are **incompatible** — train scratch or use matching `stage2_model_config` in checkpoint.

## Important Implementation Details

- **SC-VAE outputs raw logits** (`apply_output_activations=False`). Activations are applied in the loss function (`sc_vae_loss.py`) and during mesh extraction (`is_logits=True`).
- **dv activation:** `(1 + 2*VOXEL_MARGIN) * sigmoid(x) - VOXEL_MARGIN` where `VOXEL_MARGIN=0.5`
- **delta threshold:** `logits > 0` (equivalent to `sigmoid > 0.5`)
- **gamma activation:** `softplus(x)` — split weight for Dual Contouring
- **KL loss** uses `kl_weight=1e-6`, so KL values ~14.6 are normal (not a bug)
- **LMDB env sharing:** `SlatDataset._lmdb_env_cache` (class-level dict) prevents "already open" errors when multiple datasets use the same LMDB
- **Dual Contouring mesh holes:** By design in TRELLIS.2 — `.all(dim=1)` drops quads with missing neighbors. `_prefill_boundary_voxels()` in `test_sc_vae_recon_v2.py` adds synthetic voxels at boundaries to mitigate.

## TRELLIS.2 Differences (Intentional)

| Aspect | FaceDiff | TRELLIS.2 |
|--------|----------|-----------|
| Model scale | 35M params (4 levels) | ~800M params (5 levels) |
| Encoder input | 10ch (dv+delta+gamma+rgb), no centering | 6ch (dv+delta only), centered by -0.5 |
| Render loss | Point splatting, 64px, optional | Mesh rasterization, 1024px, always on, depth=10.0 |
| RGB | Joint geometry+color in SC-VAE | Separate PBR VAE |
| Stage 2 conditioning | Dual AdaLN (ctx + time), no prefix | Prefix tokens + single AdaLN (TRELLIS-style) |
| Stage 2 FFN | Per-block FFN (DiT-style) | Often in Transformer blocks only |

## Code Style

- Vietnamese comments in training code (historical)
- Config via dataclasses in `src/config.py` — all hyperparameters centralized
- LMDB for all large data: O-Voxel cache, hybrid context, slat+context
- `.pt` files as intermediate cache format before LMDB packing
