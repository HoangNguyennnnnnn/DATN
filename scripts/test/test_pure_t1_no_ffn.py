"""Test: Pure t=1 denoising WITHOUT FFN to check if FFN is really needed."""
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
    use_ffn=False  # DISABLE FFN HERE
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Params (No FFN): {params/1e6:.1f}M')

ACCUM = 16
STEPS_PER_EPOCH = 50
LR = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
sigma = 1e-4

print(f'\n=== PURE t=1 TRAINING (NO FFN, accum={ACCUM}, LR={LR}) ===')

for ep in range(1, 101):
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
