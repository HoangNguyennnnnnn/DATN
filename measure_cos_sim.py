import torch
import torch.nn.functional as F
from src.models.voxel_mamba import voxel_mamba_from_stage2_config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Generating random batch data...")
    b = 8
    x_data = torch.randn(b, 4096, 32, device=device)
    context = torch.randn(b, 946, device=device)
    
    print("2. Loading checkpoint...")
    checkpoint = torch.load("checkpoints/imf_v8_lite/best.pt", map_location='cpu')
    
    print("3. Building model...")
    backbone = voxel_mamba_from_stage2_config(checkpoint['stage2_model_config']).to(device)
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
    backbone.eval()
    
    # Sample random t
    t = torch.rand((b,), device=device) * 0.999
    
    # Generate noise
    e = torch.randn_like(x_data)
    
    # Interpolate
    z_t = (1 - t.view(-1, 1, 1)) * x_data + t.view(-1, 1, 1) * e
    
    # Ground truth target is e - x_data
    v_target = e - x_data
    
    print("4. Computing Forward Pass...")
    with torch.no_grad():
        omega_effective = torch.ones((b,), device=device) * 1.5
        v_theta = backbone(z_t, t, context, r=torch.zeros_like(t), omega=omega_effective)
        
        # Calculate Cosine Similarity
        cos_sim = F.cosine_similarity(v_theta.flatten(1), v_target.flatten(1), dim=-1).mean().item()
        
        print("\n" + "="*50)
        print(f"📊 KẾT QUẢ ĐO LƯỜNG TỪ CHECKPOINT 'best.pt'")
        print("="*50)
        print(f"MSE Loss thực tế: {F.mse_loss(v_theta, v_target).item():.4f}")
        print(f"Cosine Similarity (v_theta vs Target): {cos_sim:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
