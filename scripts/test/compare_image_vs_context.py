import argparse
import os
import sys
import torch
import lmdb
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.image_preprocessor import ImagePreprocessor
from src.inference.generator import FaceDiffGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/064_03/064_03.jpg", help="Path to raw image")
    parser.add_argument("--id", type=str, default="faceverse_064_03", help="ID in hybrid_context.lmdb")
    parser.add_argument("--lmdb", type=str, default="data/hybrid_context.lmdb", help="Path to context LMDB")
    parser.add_argument("--omega", type=float, default=4.0, help="CFG scale")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = "outputs_pipeline_test"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- 1. Getting Context from LMDB for ID: {args.id} ---")
    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    lmdb_context = None
    import io
    with env.begin() as txn:
        val = txn.get(args.id.encode())
        if val is not None:
            lmdb_context = torch.load(io.BytesIO(val), weights_only=False).to(device)
            print(f"Loaded LMDB context shape: {lmdb_context.shape}")
        else:
            print(f"ERROR: ID {args.id} not found in LMDB!")
            return
            
    print(f"--- 2. Getting Context from Raw Image: {args.image} ---")
    preprocessor = ImagePreprocessor(device=device)
    image_context = preprocessor.process(args.image)
    image_context = image_context.to(device)
    print(f"Loaded Image context shape: {image_context.shape}")
    
    # Compare contexts
    mse = torch.nn.functional.mse_loss(image_context, lmdb_context)
    print(f"--- [Analysis] MSE between LMDB context and Image context: {mse.item():.6f} ---")
    
    print("--- 3. Initializing FaceDiff Generator ---")
    generator = FaceDiffGenerator(
        device=device,
        imf_ckpt="checkpoints/imf_v8_lite/epoch_100.pt",
        sc_vae_shape_ckpt="checkpoints/sc_vae_shape/epoch_500.pt",
        cfg_scale=args.omega,
    )
    
    print(f"--- 4. Generating Mesh from LMDB Context ---")
    safe_id = args.id.replace('/', '_')
    out_lmdb = os.path.join(out_dir, f"{safe_id}_from_lmdb.ply")
    generator.generate(
        context=lmdb_context.unsqueeze(0),
        output_path=out_lmdb,
        omega=args.omega
    )
    print(f"Saved: {out_lmdb}")
    
    print(f"--- 5. Generating Mesh from Raw Image Context ---")
    out_img = os.path.join(out_dir, f"{safe_id}_from_image.ply")
    generator.generate(
        context=image_context.unsqueeze(0),
        output_path=out_img,
        omega=args.omega
    )
    print(f"Saved: {out_img}")
    
    print("DONE! You can now compare the two meshes.")

if __name__ == "__main__":
    main()
