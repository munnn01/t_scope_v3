import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional, Tuple


_GAUSSIAN_5X5_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


class SeverityConditioner(nn.Module):
    """
    Encodes (codec_family_id, normalized_severity) into a compact conditioning vector.
    Extends SCOPE (arXiv:2610.39778) Eq. (1)-(3) to video codec families:
      0: x265 (HEVC)
      1: VVenC (VVC)
      2: x264 (AVC)
      3: Intra/BPG proxy
    """

    def __init__(
        self,
        num_families: int = 4,
        family_emb_dim: int = 8,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.family_emb = nn.Embedding(num_families, family_emb_dim)
        # Basis: [s, s^2, s^3, h_0.50(s), h_0.75(s), h_0.90(s)] -> 6 dims
        in_dim = family_emb_dim + 6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

    @staticmethod
    def severity_basis(s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Tensor of shape [B] or [B, 1] in [0, 1].
        Returns:
            Basis tensor of shape [B, 6].
        """
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        s = torch.clamp(s, 0.0, 1.0)
        h50 = torch.clamp((s - 0.50) / 0.50, 0.0, 1.0)
        h75 = torch.clamp((s - 0.75) / 0.25, 0.0, 1.0)
        h90 = torch.clamp((s - 0.90) / 0.10, 0.0, 1.0)
        return torch.cat([s, s ** 2, s ** 3, h50, h75, h90], dim=-1)

    def forward(self, family_id: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        ef = self.family_emb(family_id)  # [B, 8]
        phi_s = self.severity_basis(severity)  # [B, 6]
        return self.mlp(torch.cat([ef, phi_s], dim=-1))  # [B, cond_dim]


class FiLMConvBlock(nn.Module):
    """
    Two 3x3 Conv + optional GroupNorm + GELU layers modulated by Feature-wise Linear Modulation (FiLM):
        FiLM(x, c) = x * (1 + gamma(c)) + beta(c)
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int = 64, use_norm: bool = True
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if use_norm:
            num_groups = min(8, out_ch)
            while num_groups > 1 and (out_ch % num_groups) != 0:
                num_groups -= 1
            self.norm1: nn.Module = nn.GroupNorm(num_groups, out_ch)
            self.norm2: nn.Module = nn.GroupNorm(num_groups, out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        gamma_beta = self.film_proj(c).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        h = h * (1.0 + gamma) + beta
        return self.act2(h)


class TemporalConvGRUCell(nn.Module):
    """
    Lightweight Bottleneck ConvGRU at H/4 x W/4 resolution.
    Ensures temporal consistency of object attention A_t and residuals across video frames
    so x265 Motion Estimation does not suffer from frame-to-frame pre-editing flicker.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv_gates = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.conv_can = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(
        self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        candidate = torch.tanh(self.conv_can(torch.cat([x_t, reset_gate * h_prev], dim=1)))
        h_curr = (1.0 - update_gate) * h_prev + update_gate * candidate
        return h_curr


def gaussian_blur_5x5(x: torch.Tensor) -> torch.Tensor:
    """Fast separable 5x5 Gaussian blur with per-(channel, device, dtype) cached weights."""
    c = x.shape[1]
    key = (c, str(x.device), x.dtype)
    weight = _GAUSSIAN_5X5_CACHE.get(key)
    if weight is None:
        kernel_1d = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype
        ) / 16.0
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        weight = kernel_2d.repeat(c, 1, 1, 1).contiguous()
        _GAUSSIAN_5X5_CACHE[key] = weight
    x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)


@torch.no_grad()
def _estimate_coarse_translation(
    g_prev: torch.Tensor,
    g_curr: torch.Tensor,
    h: int,
    w: int,
    scale_div: int = 8,
    search_r: int = 6,
) -> torch.Tensor:
    """Coarse H/8 integer + subpixel translation search in float32 normalized affine coords."""
    b = g_curr.shape[0]
    device = g_curr.device
    hc = max(8, h // scale_div)
    wc = max(8, w // scale_div)
    r = min(search_r, max(1, min(hc, wc) // 4))

    gc = F.interpolate(g_curr, size=(hc, wc), mode="bilinear", align_corners=False)
    gp = F.interpolate(g_prev, size=(hc, wc), mode="bilinear", align_corners=False)

    core = gc[:, :, r : hc - r, r : wc - r]
    costs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            patch = gp[:, :, r - dy : hc - r - dy, r - dx : wc - r - dx]
            # Microscopic center tie-breaker keeps uniform/zero frames cleanly at (0, 0).
            tie = 1e-6 * float(dx * dx + dy * dy)
            diff = (core - patch).abs().mean(dim=(1, 2, 3)) + tie
            costs.append(diff)

    width = 2 * r + 1
    cost_vol = torch.stack(costs, dim=1).view(b, width, width)
    flat_idx = cost_vol.view(b, -1).argmin(dim=1)
    iy = torch.div(flat_idx, width, rounding_mode="floor")
    ix = flat_idx % width

    dx_f = (ix - r).float()
    dy_f = (iy - r).float()
    for bi in range(b):
        yi, xi = int(iy[bi].item()), int(ix[bi].item())
        if 0 < xi < 2 * r:
            c_m, c_0, c_p = cost_vol[bi, yi, xi - 1], cost_vol[bi, yi, xi], cost_vol[bi, yi, xi + 1]
            denom = 2.0 * (c_m - 2.0 * c_0 + c_p)
            if denom > 1e-8:
                dx_f[bi] += torch.clamp((c_m - c_p) / denom, -0.5, 0.5)
        if 0 < yi < 2 * r:
            c_m, c_0, c_p = cost_vol[bi, yi - 1, xi], cost_vol[bi, yi, xi], cost_vol[bi, yi + 1, xi]
            denom = 2.0 * (c_m - 2.0 * c_0 + c_p)
            if denom > 1e-8:
                dy_f[bi] += torch.clamp((c_m - c_p) / denom, -0.5, 0.5)

    sx = float(w) / float(wc)
    sy = float(h) / float(hc)
    theta = torch.zeros((b, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * (dx_f * sx) / float(w)
    theta[:, 1, 2] = -2.0 * (dy_f * sy) / float(h)
    return theta


@torch.no_grad()
def estimate_global_affine_irls(
    i_prev: torch.Tensor,
    i_curr: torch.Tensor,
    iters_per_level: Tuple[int, int] = (4, 3),
    corner_snap_px: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    6-DOF Cauchy-IRLS global affine motion estimator (H/8 -> H/4) with corner-displacement static snap.
    Runs strictly in float32 (AMP-safe for linalg.solve) and returns (theta [N,2,3], is_static [N]).
    """
    n, _, h, w = i_curr.shape
    device = i_curr.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        id_theta = torch.zeros((n, 2, 3), device=device, dtype=torch.float32)
        id_theta[:, 0, 0] = 1.0
        id_theta[:, 1, 1] = 1.0
        if min(h, w) < 16 or n == 0:
            return id_theta, torch.ones((n,), dtype=torch.bool, device=device)

        ip_f = i_prev.float()
        ic_f = i_curr.float()
        luma_w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        g_prev_full = (ip_f * luma_w).sum(dim=1, keepdim=True)
        g_curr_full = (ic_f * luma_w).sum(dim=1, keepdim=True)

        theta = _estimate_coarse_translation(g_prev_full, g_curr_full, h=h, w=w, scale_div=8, search_r=6)
        eye6 = torch.eye(6, device=device, dtype=torch.float32).unsqueeze(0)

        for div, n_iter in zip((8, 4), iters_per_level):
            hs = max(8, h // div)
            ws = max(8, w // div)
            gc = F.interpolate(g_curr_full, size=(hs, ws), mode="bilinear", align_corners=False)
            gp = F.interpolate(g_prev_full, size=(hs, ws), mode="bilinear", align_corners=False)

            ys = (torch.arange(hs, device=device, dtype=torch.float32) + 0.5) * (2.0 / hs) - 1.0
            xs = (torch.arange(ws, device=device, dtype=torch.float32) + 0.5) * (2.0 / ws) - 1.0
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            gx_flat = gx.reshape(1, -1)
            gy_flat = gy.reshape(1, -1)

            gx_pad = F.pad(gc, (1, 1, 1, 1), mode="replicate")
            ix_flat = ((gx_pad[:, :, 1:-1, 2:] - gx_pad[:, :, 1:-1, :-2]) * (ws / 4.0)).reshape(n, -1)
            iy_flat = ((gx_pad[:, :, 2:, 1:-1] - gx_pad[:, :, :-2, 1:-1]) * (hs / 4.0)).reshape(n, -1)

            for _ in range(n_iter):
                grid = F.affine_grid(theta, size=(n, 1, hs, ws), align_corners=False)
                gp_w = F.grid_sample(gp, grid, mode="bilinear", padding_mode="border", align_corners=False)
                valid = ((grid[..., 0].abs() <= 0.96) & (grid[..., 1].abs() <= 0.96)).float().reshape(n, -1)

                gp_pad = F.pad(gp_w, (1, 1, 1, 1), mode="replicate")
                ix_w = ((gp_pad[:, :, 1:-1, 2:] - gp_pad[:, :, 1:-1, :-2]) * (ws / 4.0)).reshape(n, -1)
                iy_w = ((gp_pad[:, :, 2:, 1:-1] - gp_pad[:, :, :-2, 1:-1]) * (hs / 4.0)).reshape(n, -1)
                ix_f = 0.5 * (ix_w + ix_flat)
                iy_f = 0.5 * (iy_w + iy_flat)

                err = (gc - gp_w).reshape(n, -1)
                delta = 0.04
                w_irls = valid * (delta * delta) / (delta * delta + err * err)

                j_mat = torch.stack(
                    [ix_f * gx_flat, ix_f * gy_flat, ix_f, iy_f * gx_flat, iy_f * gy_flat, iy_f],
                    dim=-1,
                )
                jw = j_mat * w_irls.unsqueeze(-1)
                h_mat = torch.bmm(jw.transpose(1, 2), j_mat)
                rhs = torch.bmm(jw.transpose(1, 2), err.unsqueeze(-1))
                dp = torch.linalg.solve(h_mat + 1e-2 * eye6, rhs).squeeze(-1)
                dp = torch.clamp(dp, -0.25, 0.25)
                theta = theta + dp.view(n, 2, 3)

        # Static identity snap when corner shift < 0.75 px (prevents bilinear edge blur on static cameras)
        corners = torch.tensor(
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).t()
        disp = torch.matmul(theta, corners) - corners[:2].unsqueeze(0)
        disp_px = torch.stack([disp[:, 0].abs() * (0.5 * w), disp[:, 1].abs() * (0.5 * h)], dim=1)
        max_corner_px = disp_px.amax(dim=(1, 2))
        flat_frame = (g_curr_full - g_prev_full).abs().amax(dim=(1, 2, 3)) < 1e-5
        is_static = (max_corner_px < corner_snap_px) | flat_frame
        theta = torch.where(is_static.view(n, 1, 1), id_theta, theta)
        return theta, is_static


class TemporalSCOPE(nn.Module):
    """
    T-SCOPE: Temporally-Consistent Semantic Pre-Editor & CTU Delta-QP Predictor.
    - Operates on a video clip I of shape [B, T, 3, H, W] (or single frame [B, 3, H, W]).
    - Outputs:
        1. I_edit: Pre-edited video [B, T, 3, H, W] ready for standard x265/VVenC encoding.
        2. attn_map: Object attention A_t [B, T, 1, H, W] in [0, 1].
        3. delta_obj, delta_bg: Bounded residual edits [B, T, 3, H, W].
        4. ctu_qp_offsets: Block-level Delta-QP map [B, T, H/qg_size, W/qg_size] for x265 quantOffsets.
    """

    def __init__(
        self,
        base_ch: int = 48,
        num_families: int = 4,
        cond_dim: int = 64,
        max_obj_res: float = 0.025,
        max_bg_res: float = 0.03,
        qg_size: int = 16,
        max_delta_qp: float = 4.0,
    ) -> None:
        super().__init__()
        self.base_ch = base_ch
        self.max_obj_res = max_obj_res
        self.max_bg_res = max_bg_res
        self.qg_size = qg_size
        self.max_delta_qp = max_delta_qp

        self.conditioner = SeverityConditioner(
            num_families=num_families, family_emb_dim=8, cond_dim=cond_dim
        )

        # Encoder: HxW -> H/2xW/2 -> H/4xW/4 (+ H/8xW/8 global context branch)
        # Input channels: 3 (current frame I_t) + 3 (temporal frame diff I_t - I_{t-1}) = 6
        # Skip GroupNorm at full HxW resolution (enc1/dec1) to avoid FP32 bandwidth bottlenecks;
        # retain GroupNorm at H/2..H/8 where feature normalization matters most.
        self.enc1 = FiLMConvBlock(6, base_ch, cond_dim, use_norm=False)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMConvBlock(base_ch * 2, base_ch * 2, cond_dim, use_norm=True)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.ctx_fuse = FiLMConvBlock(base_ch * 8, base_ch * 4, cond_dim, use_norm=True)

        # Recurrent Temporal Alignment at H/4 x W/4
        self.temporal_gru = TemporalConvGRUCell(base_ch * 4)

        # Decoder: H/4xW/4 -> H/2xW/2 -> HxW
        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1)
        self.dec2 = FiLMConvBlock(base_ch * 4, base_ch * 2, cond_dim, use_norm=True)
        self.up1 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1)
        self.dec1 = FiLMConvBlock(base_ch * 2, base_ch, cond_dim, use_norm=False)

        # Output prediction heads
        self.attn_head = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        self.attn_cond_bias = nn.Linear(cond_dim, 1)
        self.obj_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_blur_strength = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Temporal EMA gate head for static background locking across P-frames
        self.bg_temporal_gate = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.bg_temporal_gate.bias, 1.0)

    def compute_ctu_qp_offsets(
        self,
        attn_map: torch.Tensor,
        reuse_iframe_across_gop: bool = False,
        disable_temporal: bool = False,
    ) -> torch.Tensor:
        """
        Converts pixel-level attention A_t [B, T, 1, H, W] into x265 QG/CTU-level
        Delta-QP offsets [B, T, H//qg_size, W//qg_size] using an absolute threshold pivot
        so sparse foreground preserves neutral QP while background maintains non-decreasing +QP
        across P-frames (unless `disable_temporal=True` or `reuse_iframe_across_gop=True`).
        """
        squeezed = attn_map.dim() == 4
        if squeezed:
            attn_map = attn_map.unsqueeze(1)
        b, t, _, h, w = attn_map.shape
        attn_flat = attn_map.view(b * t, 1, h, w)
        avg_qg = F.avg_pool2d(
            attn_flat, kernel_size=self.qg_size, stride=self.qg_size
        )
        max_qg = F.max_pool2d(
            attn_flat, kernel_size=self.qg_size, stride=self.qg_size
        )
        # Foreground-sensitive pool mix prevents small pedestrians in 16x16 QG cells from being coarsened.
        qg_attn = 0.62 * avg_qg + 0.38 * max_qg  # [B*T, 1, H_qg, W_qg]
        h_qg, w_qg = qg_attn.shape[-2], qg_attn.shape[-1]

        tau = 0.38
        bg_part = F.relu(tau - qg_attn) / tau
        dqp_frame = self.max_delta_qp * (1.75 * bg_part)
        dqp_frame = torch.clamp(dqp_frame, min=0.0, max=0.14 * 51.0)
        dqp_frame = dqp_frame.view(b, t, h_qg, w_qg)

        if reuse_iframe_across_gop and t > 1:
            out_qp = dqp_frame[:, 0:1].expand_as(dqp_frame)
            return out_qp.squeeze(1) if squeezed else out_qp

        if disable_temporal or t <= 1 or self.max_delta_qp <= 0.0:
            return dqp_frame.squeeze(1) if squeezed else dqp_frame

        qg_attn_t = qg_attn.view(b, t, h_qg, w_qg)
        max_qg_t = max_qg.view(b, t, h_qg, w_qg)
        out = [dqp_frame[:, 0]]
        prev_bg = dqp_frame[:, 0]
        for s in range(1, t):
            # Foreground guard drops carry-forward smoothly to 0 before any FG threshold.
            fg_score = torch.maximum(qg_attn_t[:, s], 0.90 * max_qg_t[:, s])
            fg_safe = 1.0 - torch.clamp((fg_score - 0.24) / 0.12, 0.0, 1.0)
            prev_smooth = 0.5 * prev_bg + 0.5 * F.avg_pool2d(
                prev_bg.unsqueeze(1), kernel_size=3, stride=1, padding=1, count_include_pad=False
            ).squeeze(1)
            curr = torch.clamp(
                torch.maximum(dqp_frame[:, s], fg_safe * prev_smooth),
                min=0.0,
                max=0.14 * 51.0,
            )
            out.append(curr)
            prev_bg = torch.maximum(prev_bg, fg_safe * curr)

        out_qp = torch.stack(out, dim=1)
        return out_qp.squeeze(1) if squeezed else out_qp

    def forward_single_step(
        self,
        i_t: torch.Tensor,
        i_prev_warped: torch.Tensor,
        c: torch.Tensor,
        h_prev_warped: Optional[torch.Tensor] = None,
        bg_prev_warped: Optional[torch.Tensor] = None,
        valid_full: Optional[torch.Tensor] = None,
        valid_h4: Optional[torch.Tensor] = None,
        sev_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Single frame step t with motion-compensated recurrent bottleneck state and background EMA."""
        diff_raw = i_t - i_prev_warped
        diff_mc_t = diff_raw * valid_full if valid_full is not None else diff_raw
        x_in = torch.cat([i_t, diff_mc_t], dim=1)  # [B, 6, H, W]

        f1 = self.enc1(x_in, c)  # [B, C, H, W]
        f2 = self.enc2(self.down1(f1), c)  # [B, 2C, H/2, W/2]
        f3 = self.bottleneck(self.down2(f2), c)  # [B, 4C, H/4, W/4]
        f4 = self.enc3(self.down3(f3), c)  # [B, 4C, H/8, W/8]
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_ctx = self.ctx_fuse(torch.cat([f3, f4_up], dim=1), c)  # [B, 4C, H/4, W/4]

        h_in = (
            h_prev_warped * valid_h4
            if (h_prev_warped is not None and valid_h4 is not None)
            else h_prev_warped
        )
        h_curr = self.temporal_gru(f3_ctx, h_in)  # [B, 4C, H/4, W/4]

        u2 = F.interpolate(h_curr, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([self.up2(u2), f2], dim=1), c)

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([self.up1(u1), f1], dim=1), c)

        attn_bias = self.attn_cond_bias(c).view(-1, 1, 1, 1)
        attn_logits = self.attn_head(u1) + attn_bias
        attn_t = torch.sigmoid(attn_logits)  # [B, 1, H, W]

        # Smoothstep STE saturation removes FG background-blur leak while preserving gradients
        u_sat = torch.clamp((attn_t - 0.12) / 0.46, 0.0, 1.0)
        eff_hard = u_sat * u_sat * (3.0 - 2.0 * u_sat)
        eff_proxy = torch.sigmoid(2.0 * (attn_logits + 0.65))
        eff_attn_t = eff_proxy + (eff_hard - eff_proxy).detach()

        s_gate = 1.0 if sev_gate is None else sev_gate
        delta_obj_t = (self.max_obj_res * s_gate) * eff_attn_t * torch.tanh(self.obj_res_head(u1))
        delta_bg_t = self.max_bg_res * torch.tanh(self.bg_res_head(u1))

        # Object-preserving branch
        i_obj_t = torch.clamp(i_t + delta_obj_t, 0.0, 1.0)

        # Severity-conditioned low-pass background branch (5x5 -> multi-scale ~11x11 Gaussian blend)
        blur_mild = gaussian_blur_5x5(i_t)
        blur_strong = F.interpolate(
            gaussian_blur_5x5(F.avg_pool2d(blur_mild, kernel_size=2, stride=2)),
            size=i_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        blur_mix = self.bg_blur_strength(c).view(-1, 1, 1, 1)
        blur_i_t = (1.0 - blur_mix) * blur_mild + blur_mix * blur_strong
        i_bg_spatial = torch.clamp(blur_i_t + delta_bg_t, 0.0, 1.0)
        if bg_prev_warped is not None:
            # Tight occlusion/parallax/disocclusion gate on motion-compensated frames
            diff_mag = diff_mc_t.abs().mean(dim=1, keepdim=True)
            motion_mag = 0.6 * diff_mag + 0.4 * F.avg_pool2d(
                diff_mag, kernel_size=5, stride=1, padding=2
            )
            res_gate = torch.clamp(1.0 - motion_mag / 0.05, 0.0, 1.0)
            v_mask = valid_full if valid_full is not None else 1.0
            conf = v_mask * torch.exp(-60.0 * motion_mag) * res_gate
            temporal_weight = 0.85 * torch.sigmoid(self.bg_temporal_gate(u1)) * conf
            i_bg_t = temporal_weight * bg_prev_warped + (1.0 - temporal_weight) * i_bg_spatial
        else:
            i_bg_t = i_bg_spatial

        i_edit_t = eff_attn_t * i_obj_t + (1.0 - eff_attn_t) * i_bg_t
        return i_edit_t, attn_t, eff_attn_t, delta_obj_t, delta_bg_t, h_curr, i_bg_t

    def forward(
        self,
        video: torch.Tensor,
        family_id: torch.Tensor,
        severity: torch.Tensor,
        reuse_iframe_qp: bool = False,
        disable_temporal: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            video: [B, T, 3, H, W] or [B, 3, H, W] in [0, 1].
            family_id: [B] integer codec family index (0=x265, 1=VVenC, 2=x264, 3=Intra).
            severity: [B] normalized compression severity in [0, 1].
            reuse_iframe_qp: If True, reuses I-frame CTU Delta-QP across the GOP (IWA-QP mode).
            disable_temporal: If True, processes every frame independently (h_prev=None, bg_prev=None,
                i_prev=i_t), providing a strict, noise-free framewise Spatial-SCOPE baseline.
        """
        single_frame_input = video.dim() == 4
        if single_frame_input:
            video = video.unsqueeze(1)  # [B, 1, 3, H, W]

        b, t, _, h, w = video.shape
        c = self.conditioner(family_id, severity)  # [B, cond_dim]
        sev_t = severity.to(device=video.device, dtype=video.dtype).reshape(-1)
        if sev_t.numel() == 1 and b > 1:
            sev_t = sev_t.expand(b)
        sev_gate = 0.30 + 0.70 * torch.clamp(sev_t.view(b, 1, 1, 1), 0.0, 1.0)

        use_motion_comp = (t > 1) and (not disable_temporal)
        aligned_prev_orig: Optional[torch.Tensor] = None
        warp_valid_mask: Optional[torch.Tensor] = None
        aligned_prev_edit: Optional[torch.Tensor] = None
        grid_full_pairs: Optional[torch.Tensor] = None
        grid_h4_pairs: Optional[torch.Tensor] = None
        valid_h4_pairs: Optional[torch.Tensor] = None
        is_static_pairs: Optional[torch.Tensor] = None
        st_mask_flat: Optional[torch.Tensor] = None

        if use_motion_comp:
            bt_pairs = b * (t - 1)
            prev_flat = video[:, :-1].reshape(bt_pairs, 3, h, w)
            curr_flat = video[:, 1:].reshape(bt_pairs, 3, h, w)
            with torch.no_grad():
                thetas_flat, is_static_flat = estimate_global_affine_irls(prev_flat, curr_flat)
                h4, w4 = max(1, h // 4), max(1, w // 4)
                grid_full_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype), size=(bt_pairs, 3, h, w), align_corners=False
                )
                grid_h4_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype), size=(bt_pairs, self.base_ch * 4, h4, w4), align_corners=False
                )
                warped_orig = F.grid_sample(
                    prev_flat, grid_full_flat, mode="bilinear", padding_mode="border", align_corners=False
                )
                valid_raw = (
                    (grid_full_flat[..., 0].abs() <= 0.985) & (grid_full_flat[..., 1].abs() <= 0.985)
                ).to(dtype=video.dtype).unsqueeze(1)
                valid_h4_raw = (
                    (grid_h4_flat[..., 0].abs() <= 0.96) & (grid_h4_flat[..., 1].abs() <= 0.96)
                ).to(dtype=video.dtype).unsqueeze(1)
                st_mask_flat = is_static_flat.view(bt_pairs, 1, 1, 1)
                aligned_prev_orig = torch.where(st_mask_flat, prev_flat, warped_orig).view(b, t - 1, 3, h, w)
                warp_valid_mask = torch.where(
                    st_mask_flat, torch.ones_like(valid_raw), valid_raw
                ).view(b, t - 1, 1, h, w)
                valid_h4_pairs = torch.where(
                    st_mask_flat, torch.ones_like(valid_h4_raw), valid_h4_raw
                ).view(b, t - 1, 1, h4, w4)
                grid_full_pairs = grid_full_flat.view(b, t - 1, h, w, 2)
                grid_h4_pairs = grid_h4_flat.view(b, t - 1, h4, w4, 2)
                is_static_pairs = is_static_flat.view(b, t - 1)

        edits, attns, eff_attns, d_objs, d_bgs = [], [], [], [], []
        h_prev: Optional[torch.Tensor] = None
        bg_prev: Optional[torch.Tensor] = None

        use_ckpt = self.training and torch.is_grad_enabled() and t > 1
        for step in range(t):
            i_t = video[:, step]
            if disable_temporal or step == 0:
                i_prev_w = i_t
                valid_full_s: Optional[torch.Tensor] = None
                valid_h4_s: Optional[torch.Tensor] = None
                if disable_temporal:
                    h_prev_w = None
                    bg_prev_w = None
                else:
                    h_prev_w = h_prev
                    bg_prev_w = bg_prev
            else:
                p_idx = step - 1
                assert aligned_prev_orig is not None and warp_valid_mask is not None
                assert (
                    is_static_pairs is not None
                    and grid_full_pairs is not None
                    and grid_h4_pairs is not None
                    and valid_h4_pairs is not None
                )
                i_prev_w = aligned_prev_orig[:, p_idx]
                valid_full_s = warp_valid_mask[:, p_idx]
                valid_h4_s = valid_h4_pairs[:, p_idx]
                if bool(is_static_pairs[:, p_idx].all()):
                    h_prev_w = h_prev
                    bg_prev_w = bg_prev
                else:
                    st_step = is_static_pairs[:, p_idx].view(b, 1, 1, 1)
                    h_prev_w = (
                        None
                        if h_prev is None
                        else torch.where(
                            st_step,
                            h_prev,
                            F.grid_sample(
                                h_prev,
                                grid_h4_pairs[:, p_idx].to(dtype=h_prev.dtype),
                                mode="bilinear",
                                padding_mode="zeros",
                                align_corners=False,
                            ),
                        )
                    )
                    bg_prev_w = (
                        None
                        if bg_prev is None
                        else torch.where(
                            st_step,
                            bg_prev,
                            F.grid_sample(
                                bg_prev,
                                grid_full_pairs[:, p_idx].to(dtype=bg_prev.dtype),
                                mode="bilinear",
                                padding_mode="border",
                                align_corners=False,
                            ),
                        )
                    )

            if use_ckpt:
                i_edit_t, attn_t, eff_attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = grad_checkpoint(
                    self.forward_single_step,
                    i_t,
                    i_prev_w,
                    c,
                    h_prev_w,
                    bg_prev_w,
                    valid_full_s,
                    valid_h4_s,
                    sev_gate,
                    use_reentrant=False,
                )
            else:
                i_edit_t, attn_t, eff_attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = self.forward_single_step(
                    i_t,
                    i_prev_w,
                    c,
                    h_prev_w,
                    bg_prev_w,
                    valid_full_s,
                    valid_h4_s,
                    sev_gate,
                )
            edits.append(i_edit_t)
            attns.append(attn_t)
            eff_attns.append(eff_attn_t)
            d_objs.append(d_obj_t)
            d_bgs.append(d_bg_t)

        i_edit = torch.stack(edits, dim=1)  # [B, T, 3, H, W]
        attn_map = torch.stack(attns, dim=1)  # [B, T, 1, H, W]
        eff_attn_map = torch.stack(eff_attns, dim=1)  # [B, T, 1, H, W]
        delta_obj = torch.stack(d_objs, dim=1)
        delta_bg = torch.stack(d_bgs, dim=1)
        ctu_qp_offsets = self.compute_ctu_qp_offsets(
            eff_attn_map,
            reuse_iframe_across_gop=reuse_iframe_qp,
            disable_temporal=disable_temporal,
        )

        if use_motion_comp:
            assert grid_full_pairs is not None and st_mask_flat is not None
            bt_pairs = b * (t - 1)
            prev_edit_flat = i_edit[:, :-1].reshape(bt_pairs, 3, h, w)
            warped_edit = F.grid_sample(
                prev_edit_flat,
                grid_full_pairs.reshape(bt_pairs, h, w, 2).to(dtype=i_edit.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            aligned_prev_edit = torch.where(st_mask_flat, prev_edit_flat, warped_edit).view(
                b, t - 1, 3, h, w
            )

        if single_frame_input:
            return {
                "i_edit": i_edit.squeeze(1),
                "attn_map": attn_map.squeeze(1),
                "eff_attn_map": eff_attn_map.squeeze(1),
                "delta_obj": delta_obj.squeeze(1),
                "delta_bg": delta_bg.squeeze(1),
                "ctu_qp_offsets": ctu_qp_offsets.squeeze(1),
                "aligned_prev_edit": None,
                "aligned_prev_orig": None,
                "warp_valid_mask": None,
            }

        return {
            "i_edit": i_edit,
            "attn_map": attn_map,
            "eff_attn_map": eff_attn_map,
            "delta_obj": delta_obj,
            "delta_bg": delta_bg,
            "ctu_qp_offsets": ctu_qp_offsets,
            "aligned_prev_edit": aligned_prev_edit,
            "aligned_prev_orig": aligned_prev_orig,
            "warp_valid_mask": warp_valid_mask,
        }
