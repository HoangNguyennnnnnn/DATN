"""Test: Pure t=1 denoising with gradient accumulation.

Previous test diverged because: random noise per iter → noisy gradients → LR too high.
Fix: accumulate gradients over 16 noise samples before each optimizer step.
Effective batch = 16 → gradient 4x cleaner.
"""
import torch, sys, io, lmdb, numpy as np, torch.nn.functional as F
sys.path.insert(0, '.')
from src.models.voxel_mamba import VoxelMamba

device = 'cuda'
env = lmdb.open('data/slat_context.lmdb', readonly=True, lock=False, readahead=False)
with env.begin() as txn:
    keys = [k for k, _ in txn.cursor() if k != b'__meta__'][:100]
    rng = np.random.default_rng(42)
    blob = torch.load(io.BytesIO(txn.get(keys[rng.choice(len(keys))])), map_location='cpu', weights_only=False)
env.close()
ctx = torch.from_numpy(blob['context']).float() if isinstance(blob['context'], np.ndarray) else blob['context'].float()
slt = torch.from_numpy(blob['slat']).float() if isinstance(blob['slat'], np.ndarray) else blob['slat'].float()

stats = torch.load('data/slat_stats.pt', map_location='cpu', weights_only=False)
slat_mean = stats['mean'].to(device).view(1,1,-1)
slat_std = stats['std'].to(device).view(1,1,-1)
x = ((slt.unsqueeze(0).to(device)) - slat_mean) / slat_std
ctx_dev = ctx.unsqueeze(0).to(device)

model = VoxelMamba(
    input_dim=32, hidden_dim=512, num_layers=12, slat_length=4096, context_dim=946,
    backend='auto', strict=False,
    num_context_tokens=8, num_time_tokens=4, num_r_tokens=4,
    num_interval_tokens=4, num_guidance_tokens=4,
    d_state=16, d_conv=4, expand=2, dropout=0.0,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Params: {params/1e6:.1f}M')

ACCUM = 16  # Accumulate over 16 noise samples
STEPS_PER_EPOCH = 50
LR = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
sigma = 1e-4

# Test 1: Check gradient flow from context to output
print('\n=== GRADIENT FLOW CHECK ===')
model.train()
e = torch.randn_like(x)
t = torch.full((1,), 1.0 - sigma, device=device)
z_t = sigma * x + (1.0 - sigma) * e

with torch.autocast('cuda', dtype=torch.bfloat16):
    v_pred = model(z_t, t, ctx_dev, r=t)
    loss = F.mse_loss(v_pred, (e - x))
loss.backward()

# Check if context-related params got gradients
ctx_grad_norm = 0
total_ctx_params = 0
for name, p in model.named_parameters():
    if 'ctx_' in name and p.grad is not None:
        ctx_grad_norm += p.grad.norm().item() ** 2
        total_ctx_params += p.numel()
ctx_grad_norm = ctx_grad_norm ** 0.5
print(f'  Context params: {total_ctx_params:,}')
print(f'  Context grad norm: {ctx_grad_norm:.6f}')
print(f'  Context grad norm > 0: {"YES ✅" if ctx_grad_norm > 1e-8 else "NO ❌ - BROKEN!"}')

# Check output_proj gradient
out_grad = model.output_proj.weight.grad.norm().item()
print(f'  output_proj grad norm: {out_grad:.6f}')

optimizer.zero_grad(set_to_none=True)

# Test 2: Pure t=1 training with gradient accumulation
print(f'\n=== PURE t=1 TRAINING (accum={ACCUM}, LR={LR}) ===')
print(f'Each step averages gradient over {ACCUM} noise samples')
print(f'{STEPS_PER_EPOCH} steps/epoch = {STEPS_PER_EPOCH * ACCUM} forward passes/epoch')
print()

for ep in range(1, 301):
    model.train()
    ep_loss = 0
    for step in range(STEPS_PER_EPOCH):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0
        for _ in range(ACCUM):
            e = torch.randn_like(x)
            t = torch.full((1,), 1.0 - sigma, device=device)
            z_t = sigma * x + (1.0 - sigma) * e
            
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v_pred = model(z_t, t, ctx_dev, r=t)
                target = e - x
                loss = F.mse_loss(v_pred, target) / ACCUM
            loss.backward()
            accum_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += accum_loss
    
    if ep % 2 == 0 or ep == 1:
        model.eval()
        cos_sum = 0
        n_eval = 10
        with torch.no_grad():
            for _ in range(n_eval):
                z_1 = torch.randn_like(x)
                t_1 = torch.ones(1, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    v_pred = model(z_1, t_1, ctx_dev, r=t_1)
                x_pred = (z_1 - v_pred).float()
                cos = F.cosine_similarity(x_pred.flatten(), x.float().flatten(), dim=0).item()
                cos_sum += cos
        avg_cos = cos_sum / n_eval
        print(f'  ep {ep:4d}: loss={ep_loss/STEPS_PER_EPOCH:.5f}, cos={avg_cos:.4f} (avg {n_eval})')
        if avg_cos > 0.90:
            print(f'  >>> PASS at epoch {ep}!')
            sys.exit(0)

print(f'\nFinal avg_cos={avg_cos:.4f}')
