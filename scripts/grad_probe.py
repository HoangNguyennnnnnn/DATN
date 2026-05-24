"""
Gradient Probe — FaceDiff iMF  (standalone)
============================================
Đọc trực tiếp từ LMDB (format: torch.load, keys: slat+context),
forward+backward, in gradient norm + context sensitivity.

Usage:
    /path/to/conda/envs/facediff/bin/python scripts/grad_probe.py \
        --checkpoint  checkpoints/imf_v8_lite/best.pt \
        --slat-lmdb   data/slat_context_balanced.lmdb \
        --num-samples 8
"""

import argparse, sys, os, io
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   default="checkpoints/imf_v8_lite/best.pt")
parser.add_argument("--slat-lmdb",   default="data/slat_context_balanced.lmdb")
parser.add_argument("--num-samples", type=int, default=8)
parser.add_argument("--device",      default="cuda:0")
args = parser.parse_args()

device = torch.device(args.device)

# ─── 1. Load samples từ LMDB ─────────────────────────────────────────────────
print("\n[1/4] Đọc samples từ LMDB …")
import lmdb

slats_list, contexts_list = [], []
env = lmdb.open(args.slat_lmdb, readonly=True, lock=False, max_readers=1)
with env.begin() as txn:
    cursor = txn.cursor()
    collected = 0
    for k, v in cursor.iternext(keys=True, values=True):
        if collected >= args.num_samples:
            break
        k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        if k_str == "__meta__":
            continue
        try:
            data = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
            slats_list.append(data["slat"].float())       # [4096, 32]
            contexts_list.append(data["context"].float()) # [946]
            collected += 1
        except Exception as e:
            print(f"  skip {k_str[:40]}: {e}")
env.close()

print(f"   Loaded {len(slats_list)} samples")
print(f"   slat shape:    {slats_list[0].shape}")
print(f"   context shape: {contexts_list[0].shape}")

slats    = torch.stack(slats_list).to(device)      # [B, 4096, 32]
contexts = torch.stack(contexts_list).to(device)   # [B, 946]
B = slats.shape[0]

# ─── 2. Normalize slats ───────────────────────────────────────────────────────
stats_path = "data/slat_stats.pt"
if os.path.exists(stats_path):
    st = torch.load(stats_path, map_location="cpu", weights_only=False)
    slat_mean = st["mean"].to(device)
    slat_std  = st["std"].to(device)
    slats = (slats - slat_mean) / slat_std.clamp(min=1e-6)
    print(f"   Normalized → [{slats.min():.3f}, {slats.max():.3f}]")

# ─── 3. Load model ────────────────────────────────────────────────────────────
print("\n[2/4] Load model …")
from src.models.voxel_mamba import voxel_mamba_from_stage2_config

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
mcfg = ckpt.get("model_config", {
    "input_dim": 32, "hidden_dim": 512, "num_layers": 8,
    "slat_length": 4096, "context_dim": 946,
    "context_cond_mode": "cross_attn",
    "context_use_arcface_only": True,
    "num_context_kv_tokens": 8,
})
print(f"   context_cond_mode: {mcfg.get('context_cond_mode')}")
print(f"   num_context_kv_tokens: {mcfg.get('num_context_kv_tokens')}")
print(f"   context_use_arcface_only: {mcfg.get('context_use_arcface_only')}")

model = voxel_mamba_from_stage2_config(mcfg)
state = (ckpt.get("ema_state_dict")
         or ckpt.get("model_state_dict")
         or ckpt.get("state_dict")
         or ckpt)
missing, _ = model.load_state_dict(state, strict=False)
if missing:
    print(f"   WARN missing: {len(missing)} keys, e.g. {missing[:2]}")

model = model.to(device).train()
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"   VRAM:   {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

# ─── 4. Forward + Backward ───────────────────────────────────────────────────
print("\n[3/4] Forward + Backward …")
t = torch.rand(B, device=device)
r = torch.zeros_like(t)

model.zero_grad()
with torch.enable_grad():
    pred = model(slats, t, contexts, r=r)
    loss = F.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()

print(f"   loss={loss.item():.6f}")
print(f"   pred |mean|={pred.detach().abs().mean():.5f}  std={pred.detach().std():.5f}")
print(f"   VRAM: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

# ─── 5. Gradient norm by group ───────────────────────────────────────────────
group_norms = defaultdict(list)
dead_params = []

for name, param in model.named_parameters():
    if param.grad is None:
        dead_params.append(name)
        continue
    gn = param.grad.detach().norm().item()

    if any(x in name for x in ["time_mlp","r_mlp","interval_mlp","time_guidance"]):
        g = "⏱  time_embed"
    elif any(x in name for x in ["context_cond_mlp","arcface_tokenizer"]):
        g = "🎭 ctx_embed"
    elif "cross_attn" in name:
        g = "🔗 cross_attn"
    elif "adaLN_ctx" in name:
        g = "📐 adaLN_ctx"
    elif any(x in name for x in ["adaLN_time","adaLN_ffn"]):
        g = "📐 adaLN_time"
    elif any(x in name for x in ["mamba","mixer","conv1d"]):
        g = "🌀 mamba"
    elif any(x in name for x in ["output_proj","output_norm"]):
        g = "📤 output"
    elif "input_embed" in name:
        g = "📥 input_embed"
    else:
        g = "❓ other"
    group_norms[g].append((name, gn))

CONTEXT_GROUPS = {"🎭 ctx_embed", "🔗 cross_attn", "📐 adaLN_ctx"}
ORDER = ["📥 input_embed","⏱  time_embed","🎭 ctx_embed",
         "🔗 cross_attn","📐 adaLN_ctx","📐 adaLN_time",
         "🌀 mamba","📤 output","❓ other"]

print("\n" + "=" * 72)
print("  GRADIENT NORM BY GROUP")
print("=" * 72)
print(f"  {'Group':<28} {'#':>4}  {'mean':>10}  {'max':>10}  {'min':>10}")
print("-" * 72)

ctx_grad_dead = False
for grp in ORDER:
    items = group_norms.get(grp, [])
    if not items:
        continue
    norms = [n for _, n in items]
    mn, mx, mi = np.mean(norms), np.max(norms), np.min(norms)
    flag = ""
    if grp in CONTEXT_GROUPS:
        if mn < 1e-7:
            flag = "  🚨 DEAD"
            ctx_grad_dead = True
        elif mn < 1e-5:
            flag = "  ⚠️  vanishing"
    print(f"  {grp:<28} {len(items):>4}  {mn:>10.3e}  {mx:>10.3e}  {mi:>10.3e}{flag}")

print("-" * 72)
if dead_params:
    print(f"  ☠️  grad=None: {len(dead_params)} tensors (first 5):")
    for p in dead_params[:5]:
        print(f"     {p}")

# ─── 6. Context sensitivity ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  CONTEXT SENSITIVITY  (eval mode, no grad)")
print("=" * 72)
model.eval()
with torch.no_grad():
    n = min(4, B)
    v_real  = model(slats[:n], t[:n], contexts[:n], r=r[:n])
    v_null  = model(slats[:n], t[:n], torch.zeros_like(contexts[:n]), r=r[:n])
    idx     = torch.randperm(B)[:n]
    v_wrong = model(slats[:n], t[:n], contexts[idx], r=r[:n])

def cos_sim(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean().item()
def mae(a, b):
    return (a - b).abs().mean().item()

cos_rn = cos_sim(v_real, v_null)
cos_rw = cos_sim(v_real, v_wrong)
print(f"  cos(real_ctx, null_ctx):   {cos_rn:+.5f}   MAE={mae(v_real,v_null):.6f}")
print(f"  cos(real_ctx, wrong_ctx):  {cos_rw:+.5f}   MAE={mae(v_real,v_wrong):.6f}")

# Thêm: so sánh variance của velocity theo context dim
v_stack = torch.stack([v_real, v_null, v_wrong], dim=0)  # [3, n, L, D]
v_var_ctx = v_stack.var(dim=0).mean().item()  # variance across ctx conditions
v_var_base = v_real.var().item()              # baseline variance
print(f"  velocity var(across ctx conditions): {v_var_ctx:.6f}")
print(f"  velocity var(baseline):              {v_var_base:.6f}")
print(f"  ratio (ctx_var / base_var):          {v_var_ctx/max(v_var_base,1e-8):.4f}")

if cos_rn > 0.999:
    verdict_ctx = "🚨 CRITICAL — context hoàn toàn bị IGNORE"
elif cos_rn > 0.99:
    verdict_ctx = "⚠️  context ảnh hưởng rất yếu (<1% signal)"
elif cos_rn > 0.95:
    verdict_ctx = "⚠️  context yếu — cần kiểm tra scale"
else:
    verdict_ctx = "✅ context sensitivity OK"
print(f"\n  → {verdict_ctx}")

# ─── 7. Activation hooks: cross_attn / adaLN ─────────────────────────────────
print("\n" + "=" * 72)
print("  ACTIVATION MAGNITUDE (per layer)")
print("=" * 72)
act_data = {}
hooks = []

def make_hook(tag):
    def h(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if isinstance(o, torch.Tensor):
            act_data[tag] = {"|mean|": o.detach().abs().mean().item(),
                             "std":   o.detach().std().item(),
                             "|max|": o.detach().abs().max().item()}
    return h

for i, layer in enumerate(model.layers):
    for attr, tag in [("cross_attn", f"L{i:02d}.cross_attn"),
                      ("adaLN_ctx",  f"L{i:02d}.adaLN_ctx"),
                      ("adaLN_time", f"L{i:02d}.adaLN_time"),
                      ("adaLN_ffn",  f"L{i:02d}.adaLN_ffn")]:
        m = getattr(layer, attr, None)
        if m is not None:
            hooks.append(m.register_forward_hook(make_hook(tag)))

model.eval()
with torch.no_grad():
    _ = model(slats[:1], t[:1], contexts[:1], r=r[:1])
for h in hooks:
    h.remove()

if act_data:
    print(f"  {'Layer':<30} {'|mean|':>8}  {'std':>8}  {'|max|':>8}")
    print("-" * 60)
    for tag in sorted(act_data):
        d = act_data[tag]
        flag = "  ⚠️ near-zero!" if d["|mean|"] < 1e-4 else ""
        print(f"  {tag:<30} {d['|mean|']:>8.5f}  {d['std']:>8.5f}  {d['|max|']:>8.5f}{flag}")
else:
    print("  (không tìm thấy cross_attn / adaLN layers)")

# ─── 8. VERDICT ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)
if ctx_grad_dead:
    print("  ❌ Context gradient DEAD → model không học identity context!")
    print("     Nguyên nhân thường gặp:")
    print("     1. arcface_tokenizer bị detach()")
    print("     2. cross_attn bị skip trong forward (ctx_tokens=None)")
    print("     3. scale/gate init khiến output≈0 → gradient bị cancel")
elif cos_rn > 0.99:
    print("  ❌ Context bị BYPASS trong inference dù gradient flow OK")
    print("     → Kiểm tra gate init: nếu gate≈0 thì cross_attn output bị triệt tiêu")
elif cos_rn > 0.95:
    print("  ⚠️  Context yếu — cần tăng learning rate cho ctx layers, hoặc init lại gate")
else:
    print("  ✅ Gradient flow và context sensitivity đều bình thường")

print()
print("  Lý giải loss≈2.0:")
print("  ┌─ v_target = noise − x_data  (cả hai đã normalized)")
print("  ├─ ‖v_target‖² ≈ ‖noise‖² + ‖x‖² ≈ 1 + 1 = 2.0  (mặc định với normalized data)")
print("  ├─ Nếu v_theta ≈ 0 (model chưa học gì) → MSE(v_theta, v_target) ≈ ‖v_target‖² ≈ 2.0")
print("  └─ adaptive_loss_mode='paper': loss = MSE / (MSE+ε)^p ≈ MSE^(1-p) ~ const khi MSE≈2")
print()
print("  → loss bắt đầu ở 2.0 là BÌNH THƯỜNG với model khởi tạo random!")
print("  → loss từ 2.0 → 1.98 sau 1 epoch (epoch 35 cũ) = model đang học rất chậm")
print("  → Cần xem loss ở cuối epoch mới để đánh giá thật sự")
