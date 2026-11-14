from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_scope import gaussian_blur_5x5
from video_codecs.hybrid_x265_ste import estimate_motion_compensated_pred


class VCMTemporalPreEditLoss(nn.Module):
    """
    Complete Training Objective for T-SCOPE on Video Coding for Machines:
      L_total = L_task + L_attn + L_bg_spatial + L_temp + L_reg
    Combines:
      - SCOPE (arXiv:2610.39778) Eq. (14)-(19): Masked FPN teacher-student consistency,
        foreground ROI pixel/gradient preservation, high-recall Tversky + weighted BCE
        + foreground saturation attention supervision, spatial background simplification,
        and residual TV / CTU Delta-QP regularization.
      - Explicit I-frame + P-frame rate optimization and occlusion-weighted
        motion-compensated temporal inter-frame background consistency (L_temp).
    """

    def __init__(
        self,
        lambda_det: float = 5.0,
        lambda_attn: float = 0.45,
        lambda_attn_sparse: float = 1.4,
        lambda_bg_flat: float = 1.2,
        lambda_bg_smooth: float = 0.8,
        lambda_rate: float = 2.5,
        lambda_temp_bg: float = 1.5,
        lambda_temp_obj: float = 0.5,
        lambda_res_l1: float = 0.1,
        lambda_res_tv: float = 0.1,
        lambda_rate_i: float = 1.0,
        lambda_rate_p: float = 4.0,
        lambda_rate_p_log: float = 0.45,
        lambda_fg_pix: float = 4.0,
        lambda_fg_grad: float = 2.0,
        lambda_fg_recon: float = 1.5,
        lambda_fg_sat: float = 2.0,
        lambda_dqp_fg: float = 0.5,
        lambda_dqp_mono: float = 0.4,
        lambda_dqp_smooth: float = 0.15,
    ) -> None:
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_attn = lambda_attn
        self.lambda_attn_sparse = lambda_attn_sparse
        self.lambda_bg_flat = lambda_bg_flat
        self.lambda_bg_smooth = lambda_bg_smooth
        self.lambda_rate = lambda_rate
        self.lambda_temp_bg = lambda_temp_bg
        self.lambda_temp_obj = lambda_temp_obj
        self.lambda_res_l1 = lambda_res_l1
        self.lambda_res_tv = lambda_res_tv
        self.lambda_rate_i = lambda_rate_i
        self.lambda_rate_p = lambda_rate_p
        self.lambda_rate_p_log = lambda_rate_p_log
        self.lambda_fg_pix = lambda_fg_pix
        self.lambda_fg_grad = lambda_fg_grad
        self.lambda_fg_recon = lambda_fg_recon
        self.lambda_fg_sat = lambda_fg_sat
        self.lambda_dqp_fg = lambda_dqp_fg
        self.lambda_dqp_mono = lambda_dqp_mono
        self.lambda_dqp_smooth = lambda_dqp_smooth

    @staticmethod
    def total_variation(x: torch.Tensor) -> torch.Tensor:
        tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return tv_h + tv_w

    @staticmethod
    def spatial_gradient_l1(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
        mx = mask[..., :, 1:]
        my = mask[..., 1:, :]
        grad_x = (mx * dx).sum() / (mx.sum() * x.shape[2] + 1e-6)
        grad_y = (my * dy).sum() / (my.sum() * x.shape[2] + 1e-6)
        return grad_x + grad_y

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        target_sum = target.sum(dim=(-3, -2, -1))
        valid = (target_sum > 0.5).to(dtype=pred.dtype)
        inter = (pred * target).sum(dim=(-3, -2, -1))
        union = pred.sum(dim=(-3, -2, -1)) + target_sum
        dice = (2.0 * inter + smooth) / (union + smooth)
        return ((1.0 - dice) * valid).sum() / valid.sum().clamp_min(1.0)

    @staticmethod
    def tversky_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        beta: float = 0.75,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        target_sum = target.sum(dim=(-3, -2, -1))
        valid = (target_sum > 0.5).to(dtype=pred.dtype)
        tp = (pred * target).sum(dim=(-3, -2, -1))
        fp = (pred * (1.0 - target)).sum(dim=(-3, -2, -1))
        fn = ((1.0 - pred) * target).sum(dim=(-3, -2, -1))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return ((1.0 - tversky) * valid).sum() / valid.sum().clamp_min(1.0)

    @staticmethod
    def masked_grad_diff_l1(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
        dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
        mx = mask[..., :, 1:] * mask[..., :, :-1]
        my = mask[..., 1:, :] * mask[..., :-1, :]
        c = pred.shape[2]
        gx = (mx * dx.abs()).sum() / (mx.sum() * c).clamp_min(1.0)
        gy = (my * dy.abs()).sum() / (my.sum() * c).clamp_min(1.0)
        return gx + gy

    def forward(
        self,
        i_edit: torch.Tensor,
        i_recon: torch.Tensor,
        attn_map: torch.Tensor,
        delta_obj: torch.Tensor,
        delta_bg: torch.Tensor,
        rate_proxy_bpp: torch.Tensor,
        m_attn: torch.Tensor,
        m_det: torch.Tensor,
        teacher_fpn_feats: Dict[str, torch.Tensor],
        student_fpn_feats: Dict[str, torch.Tensor],
        i_frame_bpp_proxy: Optional[torch.Tensor] = None,
        p_frame_bpp_proxy: Optional[torch.Tensor] = None,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        i_orig: Optional[torch.Tensor] = None,
        aligned_prev_edit: Optional[torch.Tensor] = None,
        aligned_prev_orig: Optional[torch.Tensor] = None,
        warp_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            i_edit: Pre-edited video [B, T, 3, H, W].
            i_recon: Reconstructed video after Codec/Proxy [B, T, 3, H, W].
            attn_map: Predicted attention A_t [B, T, 1, H, W].
            delta_obj, delta_bg: Predicted residuals [B, T, 3, H, W].
            rate_proxy_bpp: Estimated bitrate [B].
            m_attn: Tight GT object mask [B, T, 1, H, W].
            m_det: Expanded detector context mask [B, T, 1, H, W].
            teacher_fpn_feats: Dict of clean FPN feature tensors {level: [B*T, C_l, H_l, W_l]}.
            student_fpn_feats: Dict of compressed FPN feature tensors {level: [B*T, C_l, H_l, W_l]}.
            i_frame_bpp_proxy, p_frame_bpp_proxy: Optional separated I/P BPP tensors [B].
            ctu_qp_offsets: Optional predicted CTU QP offsets [B, T, H_qg, W_qg].
            i_orig: Optional clean input clip [B, T, 3, H, W] for foreground ROI preservation.
            aligned_prev_edit, aligned_prev_orig, warp_valid_mask: Optional motion-warped [B, T-1, ...] tensors.
        """
        b, t, c, h, w = i_edit.shape
        calc_dtype = (
            torch.float32
            if i_edit.dtype in (torch.float16, torch.bfloat16)
            else i_edit.dtype
        )

        def _cast(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return x.to(dtype=calc_dtype) if x.dtype != calc_dtype else x

        # Run all loss reductions, masks, and CTU dQP penalties in fp32 outside autocast
        # so 512x512 pixel sums and T4 fp16 GradScaler(init_scale=65536.0) never overflow.
        with torch.amp.autocast(device_type=i_edit.device.type, enabled=False):
            i_edit = _cast(i_edit)
            i_recon = _cast(i_recon)
            attn_map = _cast(attn_map)
            delta_obj = _cast(delta_obj)
            delta_bg = _cast(delta_bg)
            rate_proxy_bpp = _cast(rate_proxy_bpp)
            m_attn = _cast(m_attn)
            m_det = _cast(m_det)
            teacher_fpn_feats = {k: v.to(dtype=calc_dtype) for k, v in teacher_fpn_feats.items()}
            student_fpn_feats = {k: v.to(dtype=calc_dtype) for k, v in student_fpn_feats.items()}
            i_frame_bpp_proxy = _cast(i_frame_bpp_proxy)
            p_frame_bpp_proxy = _cast(p_frame_bpp_proxy)
            ctu_qp_offsets = _cast(ctu_qp_offsets)
            i_orig = _cast(i_orig)
            aligned_prev_edit = _cast(aligned_prev_edit)
            aligned_prev_orig = _cast(aligned_prev_orig)
            warp_valid_mask = _cast(warp_valid_mask)

            m_det_clamped = torch.clamp(m_det, 0.0, 1.0)
            m_bg = 1.0 - m_det  # [B, T, 1, H, W]

            # 1. Detector Feature Consistency Loss + Direct Foreground ROI Preservation (when i_orig is supplied)
            loss_det = torch.tensor(0.0, device=i_edit.device, dtype=calc_dtype)
            m_det_flat = m_det.view(b * t, 1, h, w)
            num_levels = max(1, len(teacher_fpn_feats))
            for level_name, f_t in teacher_fpn_feats.items():
                f_s = student_fpn_feats[level_name]
                f_t_det = f_t.detach()
                scale = f_t_det.pow(2).mean().sqrt().clamp_min(1e-3)
                m_l = F.interpolate(m_det_flat, size=f_t.shape[-2:], mode="nearest")
                diff_l = F.smooth_l1_loss(f_s / scale, f_t_det / scale, reduction="none", beta=0.1)
                masked_diff = (m_l * diff_l).sum() / (m_l.sum() * f_t.shape[1] + 1e-6)
                loss_det = loss_det + masked_diff
            loss_task = self.lambda_det * (loss_det / float(num_levels))

            fg_sum = m_det_clamped.sum()
            has_fg = (fg_sum > 0.5).to(dtype=calc_dtype)
            if i_orig is not None:
                fg_denom = (fg_sum * c).clamp_min(1.0)
                edit_diff = i_edit - i_orig
                fg_l1 = (m_det_clamped * edit_diff.abs()).sum() / fg_denom
                fg_mse = (m_det_clamped * edit_diff.pow(2)).sum() / fg_denom
                loss_fg_pix = has_fg * self.lambda_fg_pix * (fg_l1 + 10.0 * fg_mse)
                loss_fg_grad = has_fg * self.lambda_fg_grad * self.masked_grad_diff_l1(
                    i_edit, i_orig, m_det_clamped
                )
                recon_fg_mse = (m_det_clamped * (i_recon - i_orig).pow(2)).sum() / fg_denom
                loss_fg_recon = has_fg * self.lambda_fg_recon * torch.sqrt(recon_fg_mse + 1e-8)
                loss_task = loss_task + loss_fg_pix + loss_fg_grad + loss_fg_recon

            # 2. High-Recall Attention Supervision + Foreground Saturation Hinge (pushes fg_attn -> 1.0)
            halo4px = F.max_pool2d(
                m_attn.view(b * t, 1, h, w), kernel_size=9, stride=1, padding=4
            ).view(b, t, 1, h, w)
            attn_fp32 = torch.clamp(attn_map.float(), 1e-6, 1.0 - 1e-6)
            halo_fp32 = halo4px.float()
            det_fp32 = m_det_clamped.float()
            pixel_w = 1.0 + 4.0 * det_fp32 + 1.5 * (halo_fp32 - det_fp32).clamp_min(0.0)
            loss_bce = F.binary_cross_entropy(attn_fp32, halo_fp32, weight=pixel_w).to(dtype=calc_dtype)
            loss_tversky = self.tversky_loss(attn_map, halo4px, alpha=0.25, beta=0.75)

            det_valid = (m_det_clamped > 0.5).to(dtype=calc_dtype)
            attn_valid = (m_attn > 0.5).to(dtype=calc_dtype)
            det_cnt = det_valid.sum()
            attn_cnt = attn_valid.sum()
            loss_sat_det = (det_cnt > 0.5).to(dtype=calc_dtype) * (
                det_valid * F.relu(0.95 - attn_map)
            ).sum() / det_cnt.clamp_min(1.0)
            loss_sat_attn = (attn_cnt > 0.5).to(dtype=calc_dtype) * (
                attn_valid * F.relu(0.90 - attn_map)
            ).sum() / attn_cnt.clamp_min(1.0)
            loss_fg_sat = self.lambda_fg_sat * (loss_sat_det + 0.5 * loss_sat_attn)

            loss_sparse = ((1.0 - halo4px) * attn_map).mean()
            loss_att_group = (
                self.lambda_attn * (loss_bce + loss_tversky)
                + loss_fg_sat
                + self.lambda_attn_sparse * loss_sparse
            )

            # 3. Pre-Codec Spatial Background Simplification + Explicit I/P Rate Loss
            i_edit_flat = i_edit.view(b * t, c, h, w)
            blur_edit = gaussian_blur_5x5(i_edit_flat).view(b, t, c, h, w)
            loss_bg_flat = (m_bg * (i_edit - blur_edit).abs()).sum() / (
                m_bg.sum() * c + 1e-6
            )
            loss_bg_smooth = self.spatial_gradient_l1(i_edit, m_bg)
            if i_frame_bpp_proxy is not None and p_frame_bpp_proxy is not None:
                p_clean = p_frame_bpp_proxy.clamp_min(0.0)
                loss_rate = (
                    self.lambda_rate_i * i_frame_bpp_proxy.mean()
                    + self.lambda_rate_p * p_clean.mean()
                    + self.lambda_rate_p_log * torch.log1p(p_clean / 0.05).mean()
                )
            else:
                loss_rate = self.lambda_rate * rate_proxy_bpp.mean()
            loss_bg_group = (
                self.lambda_bg_flat * loss_bg_flat
                + self.lambda_bg_smooth * loss_bg_smooth
                + loss_rate
            )

            # 4. Occlusion-Weighted Motion-Aligned Temporal Consistency + P-frame dQP Monotonicity/Stability
            if t > 1:
                if aligned_prev_edit is not None:
                    prev_edit_ref = aligned_prev_edit
                    if aligned_prev_orig is not None and i_orig is not None:
                        warp_err = (i_orig[:, 1:] - aligned_prev_orig).abs().mean(dim=2, keepdim=True)
                        occ_w = torch.exp(-5.0 * warp_err)
                    else:
                        occ_w = 1.0
                    valid_m = warp_valid_mask if warp_valid_mask is not None else 1.0
                    joint_bg_mask = m_bg[:, 1:] * occ_w * valid_m
                    inter_bg_diff = (i_edit[:, 1:] - prev_edit_ref).abs()
                elif i_orig is not None and self.lambda_temp_bg > 0.0:
                    warped_edits = []
                    warped_origs = []
                    for step_idx in range(t - 1):
                        warped_edits.append(
                            estimate_motion_compensated_pred(
                                i_orig[:, step_idx + 1], i_edit[:, step_idx], block_size=16
                            )
                        )
                        warped_origs.append(
                            estimate_motion_compensated_pred(
                                i_orig[:, step_idx + 1], i_orig[:, step_idx], block_size=16
                            )
                        )
                    prev_edit_ref = torch.stack(warped_edits, dim=1)
                    prev_orig_ref = torch.stack(warped_origs, dim=1)
                    warp_err = (i_orig[:, 1:] - prev_orig_ref).abs().mean(dim=2, keepdim=True)
                    occ_w = torch.exp(-5.0 * warp_err)
                    joint_bg_mask = m_bg[:, 1:] * occ_w
                    inter_bg_diff = (i_edit[:, 1:] - prev_edit_ref).abs()
                else:
                    joint_bg_mask = m_bg[:, 1:] * m_bg[:, :-1]  # [B, T-1, 1, H, W]
                    inter_bg_diff = (i_edit[:, 1:] - i_edit[:, :-1]).abs()

                loss_temp_bg = 20.0 * (joint_bg_mask * inter_bg_diff).sum() / (
                    joint_bg_mask.sum() * c + 1e-6
                )
                loss_temp_obj = 5.0 * (delta_obj[:, 1:] - delta_obj[:, :-1]).abs().mean()
                loss_temp_group = (
                    self.lambda_temp_bg * loss_temp_bg + self.lambda_temp_obj * loss_temp_obj
                )
            else:
                loss_temp_group = torch.tensor(0.0, device=i_edit.device, dtype=calc_dtype)

            # 5. Residual Regularization + CTU Delta-QP Foreground Protection & P-frame Stability
            loss_res_l1 = delta_obj.abs().mean() + delta_bg.abs().mean()
            loss_res_tv = self.total_variation(delta_obj) + self.total_variation(delta_bg)
            loss_reg_group = (
                self.lambda_res_l1 * loss_res_l1 + self.lambda_res_tv * loss_res_tv
            )

            if ctu_qp_offsets is not None:
                hq, wq = ctu_qp_offsets.shape[-2], ctu_qp_offsets.shape[-1]
                halo_qg = F.adaptive_max_pool2d(
                    halo4px.view(b * t, 1, h, w), (hq, wq)
                ).view(b, t, hq, wq)
                fg_qg = (halo_qg > 0.1).to(dtype=calc_dtype)
                bg_qg = 1.0 - fg_qg
                loss_dqp_fg = self.lambda_dqp_fg * (fg_qg * F.relu(ctu_qp_offsets)).mean()
                loss_reg_group = loss_reg_group + loss_dqp_fg

                if t > 1:
                    bg_sum = bg_qg.sum(dim=(-2, -1))
                    has_bg = (bg_sum > 0.5).to(dtype=calc_dtype)
                    dqp_bg = (ctu_qp_offsets * bg_qg).sum(dim=(-2, -1)) / bg_sum.clamp_min(1.0)
                    pair_bg = has_bg[:, :-1] * has_bg[:, 1:]
                    anchor_bg = has_bg[:, 0:1] * has_bg[:, 1:]
                    loss_dqp_mono = self.lambda_dqp_mono * (
                        (anchor_bg * F.relu(0.95 * dqp_bg[:, 0:1].detach() - dqp_bg[:, 1:])).mean()
                        + (pair_bg * F.relu(dqp_bg[:, :-1] - dqp_bg[:, 1:])).mean()
                    )
                    loss_dqp_smooth = self.lambda_dqp_smooth * (
                        pair_bg * (dqp_bg[:, 1:] - dqp_bg[:, :-1]).abs()
                    ).mean()
                    loss_temp_group = loss_temp_group + loss_dqp_mono + loss_dqp_smooth

            total_loss = (
                loss_task
                + loss_att_group
                + loss_bg_group
                + loss_temp_group
                + loss_reg_group
            )

        return {
            "loss_total": total_loss,
            "loss_task": loss_task,
            "loss_att": loss_att_group,
            "loss_bg": loss_bg_group,
            "loss_temp": loss_temp_group,
            "loss_reg": loss_reg_group,
        }
