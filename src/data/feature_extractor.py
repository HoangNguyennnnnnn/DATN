import torch
from typing import Tuple

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    pass

try:
    # Sử dụng torchao cho việc lượng tử hóa (quantization) INT4 theo như mô tả trong RULES.md để bảo toàn VRAM cho RTX 4090
    import torchao
except Exception as e:
    print(f"[DinoV3Extractor] torchao skipped: {e}")
    torchao = None

class DinoV3Extractor:
    """
    Trích xuất đặc trưng hình khối (shape/hair) từ ảnh Back (mặt sau)
    sử dụng DINOv2-Small. Đầu ra (Output): vector 384 chiều (384-dim).
    Tối ưu hóa VRAM qua INT4 (torchao) hoặc bfloat16.
    """
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-small",
        device: str = "cuda:0",
        quantize_int4: bool = False
    ):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        print(f"[DinoV2Extractor] Loading {model_name} in bfloat16...")
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        if quantize_int4 and torchao is not None:
            try:
                print(f"[DinoV2Extractor] Applying torchao INT4 weight-only quantization...")
                torchao.quantization.quantize_(
                    self.model, 
                    torchao.quantization.int4_weight_only()
                )
            except Exception as e:
                print(f"[DinoV2Extractor] Warning: torchao fail. Fallback to bfloat16. Error: {e}")
                
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(self, back_img: torch.Tensor) -> torch.Tensor:
        """
        Nhận tensor ảnh mặt sau (Back image) (B, 3, H, W).
        Trả về giá trị nhúng (embeddings) của token [CLS] [B, 384].
        """
        mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)

        # Xử lý (Process) ảnh mặt sau
        back_inputs = self.processor(images=back_img, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        back_outputs = self.model(**back_inputs)

        # Token CLS [B, 384]
        back_cls = back_outputs.last_hidden_state[:, 0, :]
        
        mem_after = torch.cuda.memory_allocated(self.device) / (1024**2)
        print(f"[DinoV2Extractor] Memory Delta: {mem_after - mem_before:.2f} MB")
        
        return back_cls

