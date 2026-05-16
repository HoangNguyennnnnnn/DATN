# FaceDiff — Single-GPU 3D Face Generation

## Project Overview

FaceDiff generates high-quality 3D face meshes (200K+ vertices, 10-channel O-Voxel) on a single RTX 4090 using a 3-stage pipeline based on TRELLIS.2 architecture adapted for faces.

## Architecture (3 Stages)

1. **SC-VAE** (35M params) — Sparse Convolution VAE. Compresses O-Voxel (256^3 grid, 10ch) to Slat tokens [4096, 32]. Uses spconv for sparse 3D convolutions.
2. **VoxelMamba + iMF** (~21M params) — Bidirectional Mamba SSM backbone with Hilbert curve ordering. Generates Slat tokens conditioned on 946-dim hybrid context via Improved Mean Flow (1-step generation).
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

# iMF training (offline mode, fastest)
python src/train_imf.py --offline-data --slat-lmdb data/slat_context.lmdb --batch-size 64

# Evaluate SC-VAE reconstruction
python scripts/test_sc_vae_recon_v2.py --checkpoint checkpoints/sc_vae_shape/epoch_500.pt
```

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

## Code Style

- Vietnamese comments in training code (historical)
- Config via dataclasses in `src/config.py` — all hyperparameters centralized
- LMDB for all large data: O-Voxel cache, hybrid context, slat+context
- `.pt` files as intermediate cache format before LMDB packing
