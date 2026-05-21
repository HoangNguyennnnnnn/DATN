"""Quick test: skip connection effect on iMF memorization."""
import torch, sys, io, lmdb, numpy as np, torch.nn.functional as F
sys.path.insert(0, '.')
from src.models.voxel_mamba import VoxelMamba
from src.models.imf_diffusion import ImprovedMeanFlow

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

print(f'skip_scale init: {model.skip_scale.item():.4f}')
params = sum(p.numel() for p in model.parameters())
print(f'Params: {params/1e6:.1f}M')

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
imf = ImprovedMeanFlow(sigma_min=1e-4, ratio_r_neq_t=0.5, t_sampler='uniform', adaptive_loss_weighting=False)

print()
print('SKIP CONNECTION TEST: velocity = output_proj(h) + skip_scale * x_t')
print('Training with iMF compute_loss, eval with sample_1_step')
print()

for ep in range(1, 301):
    model.train()
    ep_loss = 0
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss_out = imf.compute_loss(model, x, ctx_dev, v_head=None, return_components=True)
            loss = loss_out['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()
    
    if ep % 25 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                x_pred = imf.sample_1_step(model, ctx_dev, x.shape, omega=torch.ones(1, device=device))
            cos = F.cosine_similarity(x_pred.float().flatten(), x.float().flatten(), dim=0).item()
        print(f'  ep {ep:4d}: loss={ep_loss/50:.5f}, cos_sim={cos:.4f}, skip={model.skip_scale.item():.4f}')
        if cos > 0.95:
            print(f'  >>> PASS at epoch {ep}!')
            sys.exit(0)

print(f'\nFinal: cos_sim={cos:.4f}')
if cos > 0.5:
    print('PARTIAL: Model learning, need more epochs')
else:
    print('FAIL: Still not enough')
