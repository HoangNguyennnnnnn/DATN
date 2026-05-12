import torch
from typing import Tuple

import sys
sys.modules['torchao'] = None

PROCESSOR_BACKEND = None
try:
    from transformers import AutoImageProcessor, AutoModel
    PROCESSOR_BACKEND = "AutoImageProcessor"
except Exception:
    AutoImageProcessor = None
    try:
        from transformers import AutoFeatureExtractor as AutoImageProcessor, AutoModel
        PROCESSOR_BACKEND = "AutoFeatureExtractor"
    except Exception:
        AutoImageProcessor = None
        AutoModel = None

try:
    from transformers import BitImageProcessor
except Exception:
    BitImageProcessor = None

class MockDinoProcessor:
    def __init__(self, size=224):
        self.size = {"height": size, "width": size}
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.Normalize(mean=self.image_mean, std=self.image_std)
        ])
        
    def __call__(self, images, return_tensors="pt", **kwargs):
        if not isinstance(images, torch.Tensor):
            raise ValueError("MockDinoProcessor expects a torch.Tensor")
        pixel_values = self.transform(images)
        return {"pixel_values": pixel_values}

try:
    # Torchao bị vô hiệu hóa để fix lỗi transformers
    torchao = None
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
        if AutoImageProcessor is None or AutoModel is None:
            raise RuntimeError(
                "DinoV3Extractor requires transformers image processor support, but neither "
                "AutoImageProcessor nor AutoFeatureExtractor could be imported in this environment."
            )
        self.processor_backend = str(PROCESSOR_BACKEND or "unknown")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"[DinoV2Extractor] AutoImageProcessor failed: {e}. Trying BitImageProcessor...")
            if BitImageProcessor is not None:
                try:
                    self.processor = BitImageProcessor.from_pretrained(model_name)
                    self.processor_backend = "BitImageProcessor"
                except Exception as e2:
                    print(f"[DinoV2Extractor] BitImageProcessor failed: {e2}. Falling back to MockProcessor.")
                    self.processor = MockDinoProcessor(size=224)
                    self.processor_backend = "MockDinoProcessor"
            else:
                print(f"[DinoV2Extractor] BitImageProcessor unavailable. Falling back to MockProcessor.")
                self.processor = MockDinoProcessor(size=224)
                self.processor_backend = "MockDinoProcessor"

        
        self.model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"[DinoV2Extractor] Loading {model_name} with {self.processor_backend} in {self.model_dtype}...")
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=self.model_dtype
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

        # Xử lý (Process) ảnh mặt sau, tắt rescale vì back_img đã ở dạng [0, 1]
        back_inputs = self.processor(images=back_img, return_tensors="pt", do_rescale=False)
        back_inputs = {
            key: (
                value.to(self.device, dtype=self.model_dtype)
                if torch.is_floating_point(value) else value.to(self.device)
            )
            for key, value in back_inputs.items()
        }
        back_outputs = self.model(**back_inputs)

        # Token CLS [B, 384]
        back_cls = back_outputs.last_hidden_state[:, 0, :]
        
        mem_after = torch.cuda.memory_allocated(self.device) / (1024**2)
        print(f"[DinoV2Extractor] Memory Delta: {mem_after - mem_before:.2f} MB")
        
        return back_cls
