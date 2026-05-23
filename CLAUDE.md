# FaceDiff — Single-GPU 3D Face Generation

## Project Overview

FaceDiff generates high-quality 3D face meshes (200K+ vertices, 10-channel O-Voxel) on a single RTX 4090 using a 3-stage pipeline based on TRELLIS.2 architecture adapted for faces.

## Architecture (3 Stages)

1. **SC-VAE** (35M params) — Sparse Convolution VAE. Compresses O-Voxel (256^3 grid, 10ch) to Slat tokens [4096, 32]. Uses spconv for sparse 3D convolutions.
2. **VoxelMamba + iMF v7** (105.4M backbone + 16.8M v-head + 0.5M contrastive = 122.7M total) — 12× BiMamba+FFN blocks, dual AdaLN conditioning (context + time), Hilbert ordering, **24 prefix tokens at END of sequence**, **per-layer context projections**. Generates Slat via Improved Mean Flow (1-step).
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
  train_imf.sh                # Launch Stage 2 (nohup + LMDB auto-detect)
  train_imf_v8_lite.sh              # Stage 2 v8 lite Phase A (8L, cross-attn, ctx dropout 0.1)
  train_imf_v8_phaseB_cfg.sh        # Stage 2 v8 lite Phase B CFG (400 ep)
  train_imf_v8_lite_pipeline.sh     # Auto A→B pipeline
  train_imf_v7_phaseB.sh            # v7 legacy Phase B+C
  train_imf_balanced.sh       # Stage 2 with balanced context LMDB
  auto_fix_flame_and_retrain.sh  # FLAME refresh + retrain automation
  eval_checkpoints.sh         # Checkpoint evaluation harness
  eval_scvae_checkpoints.py   # Evaluate SC-VAE checkpoints
  data/                       # Data pipeline (precompute, pack, build LMDBs)
    precompute_slat_cache.py    # Precompute slat+context .pt files
    pack_slat_lmdb.py           # Convert .pt → merged LMDB
    pack_lmdb_fast.py           # Pack O-Voxel cache → LMDB
    build_context_lmdb.py       # Build hybrid context LMDB
    generate_ovoxel_cache.py    # Generate O-Voxel cache
    compute_slat_stats.py       # Slat normalization stats
    + 5 more (manifest, split, rebalance, remix, audit)
  test/                       # Diagnostic + memorization tests
    test_imf_identity_t0.py     # Multi-ID identity diagnostic at t=0
    test_imf_memorization.py    # Memorization / conditioning diagnostics
    test_sc_vae_recon_v2.py     # Reconstruction testing + mesh extraction
    test_pure_t1.py             # Memorization gate test (1 sample, t≈1)
    + 7 more (e2e_inference, sample, training_t, etc.)
  inference/                  # Image → mesh inference pipeline
    inference_from_image.py
    preprocess_image.py
  viz/                        # Visualization helpers
  setup/                      # Install scripts (mamba, o_voxel)
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
python scripts/data/precompute_slat_cache.py \
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
  --dataset both --context-lmdb data/hybrid_context.lmdb --skip-existing

# Pack slat cache → LMDB
python scripts/data/pack_slat_lmdb.py

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
python scripts/test/test_sc_vae_recon_v2.py --checkpoint checkpoints/sc_vae_shape/epoch_500.pt
```

## Stage 2 Architecture (VoxelMamba v7, May 2026)

**Sequence:** `[4096 slat tokens (Hilbert-ordered) + 24 prefix tokens at END]` = 4120 tokens. Prefix tokens: `ctx=8, time=4, r=4, interval=4, guidance=4`.

**Per `BidirectionalMambaBlock` (×12):**
1. **Time AdaLN** → modulate pre-Mamba: `x' = norm(x)·(1+scale_t)+shift_t`
2. **BiMamba** — forward + backward Mamba scan, sum
3. **Context AdaLN** — modulate Mamba output: `out' = out·(1+scale_c)+shift_c`
4. **Gated residual:** `x = x + gate_t·out'` (`gate` bias init **1.0**, not 0)
5. **FFN** (expand×4, GELU) + **Time AdaLN** + gated residual

**Conditioning paths:**
- `ctx_cond = context_cond_mlp(context)` → **per-layer projection** `ctx_layer_projs[i]` (12× Linear(512→512)) → AdaLN_ctx (scale+shift)
- `time_cond = time_guidance_mlp(t, r, t−r, ω, tmin, tmax)` → AdaLN_time + AdaLN_ffn

**Context segment scaling (`_scale_context_segments`):** weights `(1.5, 1.0, 0.5)` for `(ArcFace, FLAME, DINOv2)` — moderate identity bias for **balanced LMDB** (already L2-normalized per segment).

**Init fixes (2026-05-22 v7):**
- `output_proj` uses **`normal_(std=sqrt(0.1/fan_in))` ≈ 0.014** per iMF Appendix A (was xavier(0.02) ≈ 0.0009 — 16x too small, starved backbone gradients)
- `adaLN_ctx` scale init **N(0, 0.1)** (was 0.02 — 5x stronger context modulation)
- FFN last layer zero-init for safe residual

**Checkpoint compatibility:** v4 checkpoints (no prefix, no per-layer ctx) are **incompatible** — train v7 from scratch.

## Stage 2 Training Protocol (Phase A/B/C — 2026-05-22/23 audit)

**v8 lite (current — 23/05/2026):** `bash scripts/train_imf_v8_lite_pipeline.sh` — 8L×FFN2, ArcFace cross-attn, JVP 0.5, `cfg_context_dropout=0.1` Phase A, auto Phase B 400ep. See `docs/STAGE2_GUIDE.md` + `docs/AUDIT_FINDINGS.md` Revision 19.

Train in 3 sequential phases (v7 legacy):

**Phase A — Pure Boundary (ep0-20):**
- Config: `ratio_r_neq_t=0`, `use_auxiliary_v_head=False`, `contrastive_loss_weight=0`
- batch=4, grad_accum=8 (effective 32), ~16min/epoch, **VRAM 10.3GB**
- Goal: Learn time conditioning + velocity field shape
- Expected: loss → ~1.4 by ep20, time cos@t0vs1 < 0.1, context dormant (BY DESIGN)
- Script: `scripts/train_imf_v7.sh`

**Phase B — JVP + v-head (ep20-26):**
- Config: `ratio_r_neq_t=0.5`, `use_auxiliary_v_head=True` (depth=8, weight 0.5)
- batch=2, grad_accum=16 (effective 32), ~40min/epoch, **VRAM 19.3GB**
- Goal: Enable JVP correction for mean velocity
- **Warning:** batch=4 will OOM (Phase A 10GB + JVP doubles → 24GB)

**Phase C — + Contrastive (ep26+):**
- Add: `contrastive_loss_weight=0.2`, `contrastive_mode="arcface"`, temp=0.1
- batch=2, ~43min/epoch, **VRAM 19.5GB**
- Goal: Force context dependency (Mamba absorbs context without this!)
- Script: `scripts/train_imf_v7_phaseB.sh` (resume from Phase A checkpoint)

## Critical Findings (2026-05-22/23 audit)

- **iMF paper-aligned hyperparams (REQUIRED):** `lr=1e-4 constant` (NOT cosine), `lr_warmup_steps=5000`, `t_sampler="logit_normal"` (mean=-0.4, scale=1.0), `ratio_r_neq_t=0.5` for Phase B
- **Mamba absorbs context modulation:** AdaLN ±10-22% per element < Mamba SSM dynamics (mean activation 0.32). Without contrastive loss, hidden state cos(ctx_a, ctx_b) stays ~0.999. **Contrastive InfoNCE (weight 0.2) is REQUIRED to activate context.**
- **`context_velocity_sep_loss` DISABLED:** 2 extra forward passes through 122M model → OOM risk. Use contrastive instead.
- **VRAM budget (RTX 4090 24GB):** Phase A 10GB batch=4, Phase B/C 19.5GB batch=2. With ctx_sep: 23+GB risk OOM.
- **`output_proj` init was 16x too small** (xavier σ≈0.0009). Fixed to `normal_(std=0.014)` per iMF Appendix A.
- **v8 lite:** cross-attn + ctx dropout Phase A (fix CFG null path). Identity test ep 10–15 trước khi đổi kiến trúc.
- Full details: `docs/AUDIT_FINDINGS.md`, `Bao_cao_FaceDiff_ChiTiet.md` Section 8.

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
