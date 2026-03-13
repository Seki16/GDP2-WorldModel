import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants for ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class DinoV2Encoder(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the DINOv2 Small model (ViT-S/14) from torch hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.eval().to(self.device)
        
        # Register buffers so they move to GPU automatically with the model
        self.register_buffer("mean", IMAGENET_MEAN.to(self.device))
        self.register_buffer("std", IMAGENET_STD.to(self.device))

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Takes a raw observation tensor (H, W, 3) or (1, 3, H, W).
        Returns a fixed embedding vector (384,).
        """
        # 1. Handle Input Shape: Convert (64, 64, 3) -> (1, 3, 64, 64)
        if img.ndim == 3 and img.shape[-1] == 3:
            x = img.permute(2, 0, 1).unsqueeze(0)
        else:
            x = img.unsqueeze(0)

        # 2. Scaling and Normalization
        # Ensure float and scale 0-255 -> 0-1 if needed
        x = x.to(self.device).float() / 255.0 if x.dtype == torch.uint8 else x.to(self.device).float()
        x = (x - self.mean) / self.std

        # 3. Resize to 224x224 (Required by DINOv2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # 4. Forward Pass -> Return (384,)
        return self.backbone(x).squeeze(0)