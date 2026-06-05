# Scripts Directory Index

Organized into subdirectories by purpose. Core training scripts remain at root.

## Root (Training + Evaluation)

| Script | Purpose |
|--------|---------|
| `train_imf_both20k.sh` / `auto_regen_and_train_both20k.sh` | iMF UNet3D training (xem CLAUDE.md — bảng này có thể cũ; CLAUDE.md là nguồn chuẩn) |
| `train_imf_v7.sh` | **Stage 2 v7 Phase A** (boundary-only, batch=4) |
| `train_imf_v7_phaseB.sh` | **Stage 2 v7 Phase B+C** (+ JVP + v-head + contrastive, batch=2) |
| `train_imf_balanced.sh` | Training with balanced context LMDB (baseline reference) |
| `auto_fix_flame_and_retrain.sh` | Full automation: rebuild FLAME context + retrain |
| `eval_checkpoints.sh` | Checkpoint evaluation harness |
| `eval_scvae_checkpoints.py` | SC-VAE checkpoint evaluation |

## `data/` — Data Pipeline

Mandatory scripts for preparing training data. See `CLAUDE.md` for full pipeline.

| Script | Purpose |
|--------|---------|
| `precompute_slat_cache.py` | Encode mesh → slat tokens via SC-VAE (offline) |
| `pack_slat_lmdb.py` | Pack `.pt` slat caches → LMDB |
| `pack_lmdb_fast.py` | Pack O-Voxel cache → LMDB |
| `build_context_lmdb.py` | Build hybrid context (Arc+FLAME+DINO) LMDB |
| `generate_ovoxel_cache.py` | Convert mesh → O-Voxel via TRELLIS.2 |
| `generate_mesh_manifest.py` | Build dataset manifest JSON |
| `compute_slat_stats.py` | 2-pass compute slat mean/std → `data/slat_stats.pt` |
| `split_train_test_ids.py` | Train/val/test split by identity |
| `rebalance_slat_context_lmdb.py` | L2-normalize each context segment in LMDB |
| `remix_slat_lmdb_with_new_context.py` | Merge new context with existing slat (no re-encode) |
| `audit_hybrid_context_segments.py` | Audit context segment magnitudes |

## `test/` — Diagnostic Tests

Used during training to verify model behavior. Re-usable.

| Script | Purpose |
|--------|---------|
| `test_imf_identity_t0.py` | **Multi-ID identity diagnostic** (time + context cos_sim) |
| `test_imf_sample.py` | Sample mesh from model (1-step + N-step) |
| `test_imf_at_training_t.py` | Loss at specific t values |
| `test_imf_memorization.py` | Memorization gate (1 sample, all t) |
| `test_pure_t1.py` | Memorization at t≈1 boundary |
| `test_pure_t1_no_ffn.py` | FFN ablation test |
| `test_skip_connection.py` | Skip connection ablation |
| `test_sc_vae_recon_v2.py` | SC-VAE reconstruction + mesh extraction |
| `test_e2e_inference.py` | E2E pipeline test |
| `test_mediapipe_flame.py` | FLAME via MediaPipe smoke test |
| `overfit_one_sample_audit.py` | Single-sample overfit debugging |

## `inference/` — Inference Pipeline

| Script | Purpose |
|--------|---------|
| `preprocess_image.py` | Image → hybrid context vector (real-image→mesh unet3d sẽ build khi finetune; bản mamba `inference_from_image.py`/`generator.py` đã xoá) |

## `viz/` — Visualization

| Script | Purpose |
|--------|---------|
| `visualize_full_gt.py` | Ground truth mesh visualization |
| `visualize_test_context.py` | Context vector inspection |

## `setup/` — Install Scripts

| Script | Purpose |
|--------|---------|
| `install_mamba_optional.sh` | Install mamba-ssm CUDA kernels |
| `install_o_voxel.sh` | Install TRELLIS.2 o-voxel library |

## Notes

- **Active training (2026-05-23):** PID 1364952 running Phase B+C, log `logs/train_imf_v7_phaseB_20260523_015443.log`
- **Phase A active checkpoint:** `checkpoints/imf_v7/epoch_20.pt` (loss=1.4155)
- See [docs/STAGE2_GUIDE.md](../docs/STAGE2_GUIDE.md) for Stage 2 training workflow
- See [docs/AUDIT_FINDINGS.md](../docs/AUDIT_FINDINGS.md) for recent audit findings
- See [Bao_cao_FaceDiff_ChiTiet.md](../Bao_cao_FaceDiff_ChiTiet.md) Section 11 for full Revision 18 details
