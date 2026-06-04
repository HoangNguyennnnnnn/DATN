# FaceDiff — Single-GPU 3D Face Generation

## Project Overview

FaceDiff generates high-quality 3D face meshes (200K+ vertices, 10-channel O-Voxel) on a single RTX 4090 using a 3-stage pipeline based on TRELLIS.2 architecture adapted for faces.

## Architecture (3 Stages)

1. **SC-VAE** (35M params) — Sparse Convolution VAE. Compresses O-Voxel (256^3 grid, 10ch) to Slat tokens [4096, 32]. Uses spconv for sparse 3D convolutions.
2. **VoxelUNet3D + iMF** — 3D conv UNet trên Slat-as-grid (4096 tokens = **16³**). Time qua **FiLM** (global), context qua **cross-attention** (ở 8³ + bottleneck 4³ + decoder 8³) trên hybrid context đã PCA-whiten, cộng **occupancy head** (fix floaters → IoU). Sinh Slat qua Improved Mean Flow (flow-matching v-pred + JVP mean-flow). *(VoxelMamba: **DEPRECATED** — data-hungry, không sinh được ở scale; UNet3D data-efficient thay thế.)*
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
    unet3d.py            # VoxelUNet3D backbone (CURRENT) — 3D conv 16³, FiLM time + cross-attn context, occupancy head, context-whiten buffer
    voxel_mamba.py       # VoxelMamba backbone (DEPRECATED — thay bằng unet3d.py)
    imf_diffusion.py     # iMF loss, sampling, JVP mean-flow correction
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
  auto_regen_and_train_both20k.sh  # Full pipeline: precompute→pack→stats→variance→train (both 20K) — orchestrator
  train_imf_both20k.sh        # Train-only iMF UNet3D (resume; BATCH/ACCUM/OCCUPANCY_LOSS_WEIGHT/TARGET env)
  train_imf_unet.sh           # Stage 2 UNet3D launcher (CFG, env-driven)
  train_imf_v4_pipeline.sh    # Reference 2-phase A(boundary ratio=0)→B(JVP ratio=0.5)
  eval_trend_both20k.sh       # Eval gen-self định kỳ → logs/eval_trend_both20k.log (gate metric)
  eval_scvae_checkpoints.py   # Evaluate SC-VAE checkpoints
  data/                       # Data pipeline (precompute, pack, build LMDBs)
    precompute_slat_cache.py    # Precompute slat+context .pt (đọc ovoxel-lmdb + context-lmdb; --shard-id/--num-shards song song)
    pack_slat_lmdb.py           # Convert .pt → merged slat+context LMDB
    compute_slat_stats.py       # Slat per-channel mean/std (2-pass)
    compute_context_whiten.py   # PCA-whiten context 946→~632 (loại chiều variance≈0)
    compute_voxel_variance.py   # Per-voxel variance → variance-weighted loss
    pack_lmdb_fast.py / build_context_lmdb.py / generate_ovoxel_cache.py / generate_mesh_manifest.py / audit_hybrid_context_segments.py
  test/                       # Diagnostic + inference tests
    gen_scale_indep_noise.py    # Gate metric: gen-self diag-off ở scale (--stride 25 lấy ID đa dạng, --steps multi-step)
    test_e2e_inference_unet.py  # Decode generation → mesh (occ head)
    test_real_image_inference.py# Ảnh thật → mesh (pipeline cuối)
    test_sc_vae_recon_v2.py     # SC-VAE recon + mesh extraction (Dual Contouring)
    decode_both_compare.py / diag_roundtrip_holes.py / decode_gt_slat_mesh.py  # chẩn đoán mesh/lỗ
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

# iMF UNet3D — full auto pipeline (regen slat → pack → stats → variance → train).
# Đọc ovoxel từ data/ovoxel_cache_lmdb + context từ data/hybrid_context.lmdb (không render lại).
NUM_SHARDS=4 BATCH=16 ACCUM=16 bash scripts/auto_regen_and_train_both20k.sh

# Train-only (data đã sẵn): resume checkpoint mới nhất → TARGET epoch
BATCH=32 ACCUM=8 TARGET=8000 bash scripts/train_imf_both20k.sh

# (Thủ công) precompute slat song song nhiều shard:
python scripts/data/precompute_slat_cache.py --sc-vae-ckpt checkpoints/sc_vae_both/latest_step.pt \
  --dataset both --ovoxel-lmdb data/ovoxel_cache_lmdb --context-lmdb data/hybrid_context.lmdb \
  --manifest data/mesh_manifest.json --disable-id-filters --skip-existing \
  --shard-id 0 --num-shards 4

# Gate metric (gen-self diag-off, --stride 25 để lấy ID đa dạng):
python scripts/test/gen_scale_indep_noise.py --ckpt checkpoints/imf_both20k/latest_step.pt \
  --lmdb data/slat_context_both20k.lmdb --stats data/slat_stats_both20k.pt \
  --n 8 --stride 25 --steps 8 --omega 2
# Theo dõi trend tự động (mỗi 30 phút → logs/eval_trend_both20k.log):
bash scripts/eval_trend_both20k.sh

# Evaluate SC-VAE reconstruction
python scripts/test/test_sc_vae_recon_v2.py --checkpoint checkpoints/sc_vae_both/latest_step.pt
```

## Stage 2 Architecture (VoxelUNet3D — 29/05/2026)

**Backbone** (`src/models/unet3d.py`, `VoxelUNet3D`): Slat `[B,4096,32]` reshape thành **grid 16³×32** (raster D,H,W) → 3D conv UNet (`unet_base=128`, mults [1,2,4] → 16³/8³/4³). Build từ config lưu trong checkpoint bằng `voxel_unet3d_from_stage2_config(mcfg)`.

**Conditioning:**
- **Time** qua **FiLM** (scale+shift global mỗi `_ResBlock3D`, `cond_dim=512`).
- **Context** qua **cross-attention** (`_CrossAttn3D`): voxel (query) attend tới context tokens (K/V) ở **8³ encoder + 4³ bottleneck + 8³ decoder**. Cross-attn per-position fix "mean-face trap" của FiLM-only ở scale; `proj` init nhỏ NON-zero (std 0.02) để gate mở kịp.

**Context whitening (BUFFER trong model):** `register_buffer _whiten_mean`, `_whiten_W [out_dim=632, 946]` — lưu trong checkpoint. `_prep_context()` áp PCA-whiten **946→632 NỘI BỘ** ⇒ eval/inference truyền context **RAW 946-d**, model tự whiten. Loại chiều variance≈0 (FLAME gần hằng) → off-diag cos context giảm mạnh.

**Occupancy head:** `occ_conv = Conv3d(c1, 1, 3)` → `[B, 4096]` logit khi `forward(..., return_occupancy=True)`. Dùng cho BCE occupancy loss (học voxel occupied/empty → gate output, fix floaters, tăng IoU).

**Interface khớp `ImprovedMeanFlow.compute_loss`:** `forward(z_t, t, context, r=, omega=, cfg_tmin=, cfg_tmax=)` — r/omega/cfg nhận-và-bỏ-qua (FM v-pred chuẩn), giữ để tương thích sampler.

## Stage 2 Training (UNet3D — iMF JVP mean-flow)

**2-phase** (xem `train_imf_v4_pipeline.sh`): **Phase A** boundary warmup (`--ratio-r-neq-t 0.0`, no JVP) → **Phase B** JVP mean-flow (`--ratio-r-neq-t 0.5` + v-head + CFG, resume). JVP-from-scratch nổ loss (du/dt khổng lồ khi u random) → BẮT BUỘC warmup velocity field trước.

**Env BẮT BUỘC Phase B (chống nổ JVP):** `IMEFLOW_ADAPTIVE=paper IMEFLOW_ADAPTIVE_ON=1` (loss/(loss+ε)^p). Thiếu → loss diverge ~1e9.
**Env khác:** `OCCUPANCY_LOSS_WEIGHT=1.0` (bật occ head), `PREDICTION_TYPE=velocity`, `VOXEL_VARIANCE_PATH=<pt>` + `VOXEL_VARIANCE_MULT=4.0`.
**Flags chính:** `--backbone unet3d --unet-base 128 --context-use-all --context-whiten <pt> --cfg-context-dropout 0.1 --enable-cfg-conditioning --cfg-omega-max 8 --v-loss-weight 1.0 --t-sampler logit_normal --lr 1e-4`.

**VRAM (4090):** train footprint batch16 ~5 GB / batch32 ~9 GB (chưa kể job khác trên cùng GPU). **Bottleneck = JVP single-thread launch-bound** (GPU ~66%, 1 core CPU launch kernel) — tăng batch/core/RAM/torch.compile đều KHÔNG nhanh hơn (torch.compile bất khả vì JVP = double-backward).

**Gate:** `gen_scale_indep_noise.py --stride 25` đo `gen-self diag-off` (ID đa dạng); v4 đạt ~0.38; target ~0.35. Eval định kỳ tự động bằng `eval_trend_both20k.sh`.

## Training Strategy (Pretrain both → Finetune FaceVerse)

**Lý do domain:**
- **FaceScape** (~18.3K mẫu, 89% data) cho **đa dạng identity** NHƯNG **thiếu mắt + vai** (geometry không đầy đủ).
- **FaceVerse** (~2.3K mẫu, 11%; ID 1 held-out test → 109 ID × 21 expr = 2289) có **hình học đầy đủ (mắt + vai)** nhưng ít ID.

**Kế hoạch 2 pha:**
1. **Pretrain trên `both` (20K):** dựng identity prior rộng. Vì FaceScape áp đảo, eval cho thấy FaceScape `diag-off` cao hơn FaceVerse (faceverse bị "loãng") — đây là **kỳ vọng, không phải bug**; finetune sẽ sửa. Data dùng `slat_context_both20k.lmdb` + `slat_stats_both20k.pt` + `voxel_variance_both20k.pt`; whiten dùng tạm `context_whiten_v4.pt` (faceverse-frame, 632-d).
2. **Finetune trên `faceverse`:** khôi phục hình học đầy đủ (mắt+vai) + sharpen identity faceverse. Recompute `slat_stats_faceverse` / `context_whiten_faceverse` / `voxel_variance_faceverse`; LR nhỏ (~5e-5); occupancy giữ bật. **Verify bằng decode mesh** (`test_e2e_inference_unet.py` / `test_real_image_inference.py`) — kiểm mắt+vai hiện + ArcFace(mesh, GT), KHÔNG chỉ nhìn cos.

**Lưu ý:** pretrain quá lâu trên FaceScape (thiếu mắt/vai) càng làm lệch geometry prior → finetune phải sửa nhiều hơn. Khi FaceScape `diag-off` bão hòa (~0.32-0.35) là thời điểm hợp lý chuyển finetune.

## Critical Findings

- **iMF paper-aligned hyperparams (REQUIRED):** `lr=1e-4 constant` (NOT cosine), `lr_warmup_steps=5000`, `t_sampler="logit_normal"` (-0.4, 1.0), `ratio_r_neq_t=0.5` cho Phase B.
- **Slat phải `.dense()`:** encode `spconv.SparseConvTensor(mu, idx, shape, 1).dense().view().transpose()` để GIỮ vị trí 3D voxel. `model.encode()` thuần làm mất vị trí (occupied dồn về z=0-3) → Stage 2 sinh rác — **đây là root cause mọi lần Stage 2 fail trước**.
- **adaptive=paper bắt buộc cho JVP:** `IMEFLOW_ADAPTIVE=paper IMEFLOW_ADAPTIVE_ON=1`, nếu không JVP loss nổ ~1e9. (Đánh đổi: làm phẳng gradient sample-khó → có thể góp phần "mặt trung bình".)
- **Context whiten là buffer model (632-d):** phải khớp `ctx_tokenizer.0` (512×632). Recompute whiten đổi out_dim/frame → resume hỏng conditioning; reuse whiten cũ giữ tương thích.
- **Resume scheduler mismatch:** ckpt cũ thiếu key `_schedulers` của `SequentialLR` → `KeyError`. Đã fix trong `load_checkpoint` (bắt `KeyError` → auto-downgrade: giữ model+epoch, scheduler→`ConstantLR` 1e-4, EMA re-init từ weights).
- **Occupancy head (`OCCUPANCY_LOSS_WEIGHT`):** BCE per-voxel occupied/empty → gate output lúc decode → bớt floater/lỗ, tăng IoU.
- **config.py default LỆCH với pipeline:** `backbone="voxel_mamba"` + `prediction_type="x0"` vẫn là default → **mọi lệnh train PHẢI override** `--backbone unet3d` và env `PREDICTION_TYPE=velocity`.
- **torch.compile bất khả với iMF:** JVP = double-backward, torch.compile không hỗ trợ → code tắt compile cho unet3d. Bottleneck tốc độ là JVP single-thread launch (1 core CPU), không vượt được bằng batch/core/RAM.
- Full details: `docs/AUDIT_FINDINGS.md`, `Bao_cao_FaceDiff_ChiTiet.md`.

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
| Stage 2 backbone | VoxelUNet3D (3D conv trên grid 16³) | DiT-style transformer |
| Stage 2 conditioning | FiLM time + cross-attn full context (whiten 632-d) + occupancy head | Prefix + cross-attn |

## Code Style

- Vietnamese comments in training code (historical)
- Config via dataclasses in `src/config.py` — all hyperparameters centralized
- LMDB for all large data: O-Voxel cache, hybrid context, slat+context
- `.pt` files as intermediate cache format before LMDB packing
