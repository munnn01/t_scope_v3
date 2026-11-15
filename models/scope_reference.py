"""Paper-derived SCOPE image pre-editor adapted to the MOT17/x265 interface.

The H/4 U-Net, FiLM condition, direct attention blend and residual bounds follow
SCOPE (Dani et al., arXiv:2610.39778, Sections 3.1--3.2). This is an independent
adaptation using this repository's conditioning/convolution primitives, not the
authors' implementation, original weights or an official COCO reproduction.
Codec family IDs retain the repository's convention (0 is x265).
"""
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from models.temporal_scope import FiLMConvBlock, SeverityConditioner, gaussian_blur_5x5


class ScopePaperAdapted(nn.Module):
    """Image-only conditional pre-editor with no recurrence or QG allocation.

    Defaults use channels 48/96/192, an 8-dimensional family embedding and a
    64-dimensional condition. A clip is simply a batch of independent images.
    ``eff_attn_map`` aliases the direct sigmoid attention for loss compatibility.
    ``qg_size`` determines only the shape of the all-zero compatibility output.
    """

    def __init__(
        self,
        base_ch: int = 48,
        num_families: int = 4,
        cond_dim: int = 64,
        max_obj_res: float = 0.09,
        max_bg_res: float = 0.03,
        qg_size: int = 16,
        max_delta_qp: float = 0.0,
    ) -> None:
        super().__init__()
        if type(qg_size) is not int or qg_size < 1:
            raise ValueError("qg_size must be a positive integer")
        if max_delta_qp != 0.0:
            raise ValueError("The paper-adapted reference has no QG allocation")
        self.base_ch = base_ch
        self.max_obj_res = max_obj_res
        self.max_bg_res = max_bg_res
        self.qg_size = qg_size
        self.max_delta_qp = 0.0
        self.conditioner = SeverityConditioner(
            num_families=num_families, family_emb_dim=8, cond_dim=cond_dim
        )

        self.enc1 = FiLMConvBlock(3, base_ch, cond_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.enc2 = FiLMConvBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.bottleneck = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.dec2 = FiLMConvBlock(base_ch * 4, base_ch * 2, cond_dim)
        self.up1 = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.dec1 = FiLMConvBlock(base_ch * 2, base_ch, cond_dim)
        self.attn_head = nn.Conv2d(base_ch, 1, 3, padding=1)
        self.obj_res_head = nn.Conv2d(base_ch, 3, 3, padding=1)
        self.bg_res_head = nn.Conv2d(base_ch, 3, 3, padding=1)

    def _edit_images(self, image: torch.Tensor, condition: torch.Tensor):
        """Apply the image-only U-Net to one batch of frames."""
        enc1 = self.enc1(image, condition)
        enc2 = self.enc2(self.down1(enc1), condition)
        bottleneck = self.bottleneck(self.down2(enc2), condition)
        up2 = self.up2(F.interpolate(
            bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False
        ))
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1), condition)
        up1 = self.up1(F.interpolate(
            dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False
        ))
        features = self.dec1(torch.cat((up1, enc1), dim=1), condition)

        attention = torch.sigmoid(self.attn_head(features))
        delta_obj = self.max_obj_res * torch.tanh(self.obj_res_head(features))
        delta_bg = self.max_bg_res * torch.tanh(self.bg_res_head(features))
        object_image = (image + delta_obj).clamp(0.0, 1.0)
        background_image = (gaussian_blur_5x5(image) + delta_bg).clamp(0.0, 1.0)
        edited = attention * object_image + (1.0 - attention) * background_image
        return edited, attention, delta_obj, delta_bg

    def forward(
        self,
        video: torch.Tensor,
        family_id: torch.Tensor,
        severity: torch.Tensor,
        reuse_iframe_qp: bool = False,
        disable_temporal: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Edit BCHW or BTCHW RGB input; temporal compatibility flags are inert.

        Clip training checkpoints each independent frame to avoid retaining a
        full-resolution U-Net's activations for every frame simultaneously.
        """
        if video.dim() not in (4, 5):
            raise ValueError("Expected BCHW or BTCHW RGB input")
        single_image = video.dim() == 4
        clip = video.unsqueeze(1) if single_image else video
        b, t, channels, height, width = clip.shape
        if b < 1 or t < 1 or channels != 3 or min(height, width) < 4:
            raise ValueError("Expected nonempty RGB images with height and width at least 4")
        condition = self.conditioner(family_id, severity)
        checkpoint_frames = self.training and torch.is_grad_enabled() and t > 1
        frames = []
        for step in range(t):
            if checkpoint_frames:
                frame = grad_checkpoint(self._edit_images, clip[:, step], condition, use_reentrant=False)
            else:
                frame = self._edit_images(clip[:, step], condition)
            frames.append(frame)

        def restore(index: int) -> torch.Tensor:
            restored = torch.stack([frame[index] for frame in frames], dim=1)
            return restored.squeeze(1) if single_image else restored

        attn_map = restore(1)
        offsets = clip.new_zeros(b, t, max(1, height // self.qg_size), max(1, width // self.qg_size))
        return {
            "i_edit": restore(0),
            "attn_map": attn_map,
            "eff_attn_map": attn_map,
            "delta_obj": restore(2),
            "delta_bg": restore(3),
            "ctu_qp_offsets": offsets.squeeze(1) if single_image else offsets,
            "aligned_prev_edit": None,
            "aligned_prev_orig": None,
            "warp_valid_mask": None,
        }
