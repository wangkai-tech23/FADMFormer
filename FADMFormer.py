"""
Fixed FADMFormer / SREB-CRRB-SDCA implementation.

Main fixes over the uploaded version:
1) DynamicSubbandDepthModulator now feeds 8C router input consistently:
   4C self-energy + 4C cross-subband contrast.
2) window_reverse avoids float division when recovering batch size.
3) blc_to_bchw supports one-side known H/W and checks token reshaping robustly.
4) Cross-only parameters are instantiated only for cross-attention layers, and
   self-only SDCA parameters are instantiated only for self-attention layers.
5) FAA/SREB blocks no longer instantiate unused DMA/CRRB projection parameters.
6) In DMA/CRRB, the guidance branch and each KV source branch are prepared
   by independent CGDP/CDPG modules before cross-source attention.

The public model interface remains:
    model = FADMFormer(input_channels, nb_filter, depth, num_heads, win_size=8, img_size=256)
    out = model(x)
"""

"""
FADMFormer refactored for a TGRS-style revision.

Main changes over the old FAA/DMA implementation:
1) FAA is refactored as a Spectral Residual Encoding Block (SREB):
   Detail/Context decomposition -> subband-depth competitive attention residual -> adaptive residual fusion -> ConvFFN.
2) DMA is refactored as a Cross-source Residual Refinement Block (CRRB):
   multi-source residual attention uses token concatenation rather than source-head binding.
3) WFDA is replaced by DynamicSubbandDepthModulator:
   LL/LH/HL/HH are modulated through image-adaptive, spatial-adaptive, and depth-conditioned competitive weights.
4) The old PBT-like local branch x attention branch multiplicative gate is removed.

The public model interface remains close to the original code:
    model = FADMFormer(input_channels, nb_filter, depth, num_heads, win_size=8, img_size=256)
    out = model(x)

This file intentionally does not include the former data augmentation component.
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth, implemented locally to avoid a hard dependency on timm."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    try:
        return nn.init.trunc_normal_(tensor, std=std)
    except AttributeError:
        return nn.init.normal_(tensor, std=std)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def blc_to_bchw(x: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None) -> torch.Tensor:
    b, l, c = x.shape
    if h is None and w is None:
        h = int(math.sqrt(l))
        w = h
    elif h is None:
        assert w is not None
        h = l // w
    elif w is None:
        w = l // h
    assert h * w == l, f"Token length {l} cannot be reshaped to {h}x{w}."
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """B,H,W,C -> nW*B, win, win, C."""
    b, h, w, c = x.shape
    assert h % win_size == 0 and w % win_size == 0, (
        f"H={h}, W={w} must be divisible by win_size={win_size}."
    )
    x = x.view(b, h // win_size, win_size, w // win_size, win_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, c)
    return windows


def window_reverse(windows: torch.Tensor, win_size: int, h: int, w: int) -> torch.Tensor:
    """nW*B, win, win, C -> B,H,W,C."""
    assert h % win_size == 0 and w % win_size == 0, (
        f"H={h}, W={w} must be divisible by win_size={win_size}."
    )
    num_windows_per_img = (h // win_size) * (w // win_size)
    assert windows.shape[0] % num_windows_per_img == 0, (
        f"windows batch={windows.shape[0]} is not divisible by windows/image={num_windows_per_img}."
    )
    b = windows.shape[0] // num_windows_per_img
    x = windows.view(b, h // win_size, w // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# -----------------------------------------------------------------------------
# Haar wavelet transform used by the dynamic subband-depth modulator
# -----------------------------------------------------------------------------

class HaarDWT2D(nn.Module):
    """Fast differentiable one-level Haar DWT/IDWT without external dependencies."""

    @staticmethod
    def decompose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: B,C,H,W; H and W should be even for the main IRSTD settings.
        b, c, h, w = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        # Orthonormal Haar scaling up to a constant factor. The same inverse below restores x.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def reconstruct(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor,
                    out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5
        b, c, h, w = ll.shape
        out = torch.zeros(b, c, h * 2, w * 2, dtype=ll.dtype, device=ll.device)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        if out_hw is not None:
            out = out[:, :, :out_hw[0], :out_hw[1]]
        return out


# -----------------------------------------------------------------------------
# New WFDA core: dynamic subband-depth competitive modulation
# -----------------------------------------------------------------------------

class DynamicSubbandDepthModulator(nn.Module):
    """
    Subband Interaction Router (SIR) for wavelet-domain attention.

    This version removes scalar subband/depth addition completely.
    Instead of using independently generated subband weights, it treats
    LL/LH/HL/HH as a coupled subband set and learns competitive weights
    through nonlinear subband interaction routing.

    The router uses:
        1) subband self-energy: |B_s|
        2) cross-subband contrast: |B_s - mean(B_{r!=s})|
        3) global frequency context: GAP(|LL|, |LH|, |HL|, |HH|)

    The modulation remains residual:
        B_s' = B_s * (1 + alpha * (W_s - 1/4))
    """

    def __init__(self, dim: int, total_depth: int = 4, init_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.total_depth = total_depth  # kept for compatibility, not used as a lookup
        self.wavelet = HaarDWT2D()

        hidden = max(dim // 2, 16)

        # Local subband-interaction router.
        # Input contains both self-energy and cross-subband contrast:
        #   self_energy:     B, 4C, H/2, W/2
        #   contrast_energy: B, 4C, H/2, W/2
        # total: B, 8C, H/2, W/2
        self.local_router = nn.Sequential(
            nn.Conv2d(dim * 8, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

        # Global frequency-condition controller.
        # It does not produce subband weights directly.
        # It modulates the router feature by FiLM, so the routing rule
        # changes according to the global frequency state of the input.
        self.global_controller = nn.Sequential(
            nn.Linear(dim * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )

        # Initialize the last layer to zero so the model starts from
        # local routing and learns global conditioning gradually.
        nn.init.zeros_(self.global_controller[-1].weight)
        nn.init.zeros_(self.global_controller[-1].bias)

        # Output one logit map for each subband and channel:
        # B, 4C, H/2, W/2 -> B, 4, C, H/2, W/2
        self.router_out = nn.Conv2d(hidden, dim * 4, kernel_size=1, bias=True)

        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # For visualization and analysis.
        self.last_subband_weights = None
        self.last_subband_logits = None
        self.last_self_energy = None
        self.last_contrast_energy = None
        self.last_global_descriptor = None

    def forward(self, x_tokens: torch.Tensor, depth_level=None) -> torch.Tensor:
        # depth_level is intentionally unused.
        # Different depths naturally have different feature statistics and
        # independent module parameters, instead of relying on a hard index.
        _ = depth_level

        b, l, c = x_tokens.shape
        h = w = int(math.sqrt(l))
        assert h * w == l, "DynamicSubbandDepthModulator expects square token maps."

        x = blc_to_bchw(x_tokens, h, w)

        ll, lh, hl, hh = self.wavelet.decompose(x)
        bands = torch.stack([ll, lh, hl, hh], dim=1)  # B, 4, C, H/2, W/2
        _, _, _, hs, ws = bands.shape
        # ---------------------------------------------------------
        # 1) Relative subband saliency
        # ---------------------------------------------------------
        abs_bands = bands.abs()  # B, 4, C, H/2, W/2

        # Mean energy of the other three subbands.
        other_energy = (abs_bands.sum(dim=1, keepdim=True) - abs_bands) / 3.0

        # Cross-subband contrast. This keeps the implementation consistent with
        # local_router's 8C input design: 4C self-energy + 4C contrast-energy.
        contrast_energy = (abs_bands - other_energy).abs()

        # Relative subband saliency:
        # A subband is emphasized only when it is both energetic and
        # distinctive compared with the other three subbands.
        eps = 1e-6
        relative_score = (abs_bands - other_energy) / (other_energy + eps)
        subband_saliency = abs_bands * torch.sigmoid(relative_score)

        # B, 8C, H/2, W/2
        self_energy = abs_bands.flatten(1, 2)
        contrast_energy_flat = contrast_energy.flatten(1, 2)
        router_input = torch.cat([self_energy, contrast_energy_flat], dim=1)

        # Local nonlinear subband interaction.
        router_feat = self.local_router(router_input)  # B, hidden, H/2, W/2

        # ---------------------------------------------------------
        # 2) Global frequency context
        # ---------------------------------------------------------
        # B, 4, C
        global_descriptor = subband_saliency.mean(dim=(3, 4))

        # B, 4C
        global_vector = global_descriptor.flatten(1)

        # FiLM modulation: B, 2*hidden
        gamma_beta = self.global_controller(global_vector)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma.view(b, -1, 1, 1)
        beta = beta.view(b, -1, 1, 1)

        # Global-conditioned routing, not scalar addition.
        router_feat = router_feat * (1.0 + torch.tanh(gamma)) + beta

        # ---------------------------------------------------------
        # 3) Competitive subband routing
        # ---------------------------------------------------------
        logits = self.router_out(router_feat).view(b, 4, c, hs, ws)

        # Competition over LL/LH/HL/HH at each channel and spatial position.
        weights = F.softmax(logits, dim=1)

        # Residual modulation.
        centered = weights - 0.25
        scale = self.res_scale.tanh()
        modulated = bands * (1.0 + scale * centered)

        # Save for visualization / analysis.
        self.last_subband_weights = weights.detach()
        self.last_subband_logits = logits.detach()
        self.last_self_energy = abs_bands.detach()
        self.last_contrast_energy = contrast_energy.detach()
        self.last_subband_saliency = subband_saliency.detach()
        self.last_global_descriptor = global_descriptor.detach()

        out = self.wavelet.reconstruct(
            modulated[:, 0],
            modulated[:, 1],
            modulated[:, 2],
            modulated[:, 3],
            out_hw=(h, w),
        )

        out = self.out_proj(out)
        return bchw_to_blc(out)


# -----------------------------------------------------------------------------
# Attention and FFN blocks
# -----------------------------------------------------------------------------

class QKVProjection(nn.Module):
    """
    QKV projection for window attention.

    Self-attention:
        The query is first modulated by DynamicSubbandDepthModulator.

    Cross-attention:
        Each semantic source has an independent K/V projection, and the
        guidance feature is projected into source-wise query groups. The
        j-th query group only interacts with the j-th K/V source. No dynamic
        head-source routing or source-level bias is used.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block: str,
        total_depth: int = 4,
        max_sources: int = 4,
        route_dim: Optional[int] = None,  # kept only for backward-compatible calls
        attn_mode: str = "self",
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block = block
        assert attn_mode in ("self", "cross"), f"Unsupported attn_mode={attn_mode}."
        self.attn_mode = attn_mode
        self.max_sources = max_sources
        self.route_dim = route_dim  # unused

        # Keep these attributes for compatibility with _reset_special_inits().
        self.source_kv_delta = None
        self.route_q = None
        self.route_k = None
        self.head_source_prior = None
        self.source_embed = None

        if self.attn_mode == "cross":
            # Produce S query groups from the same deep guidance feature.
            # Shape after projection: B_, N, max_sources * C.
            self.cross_to_q = nn.Linear(dim, dim * max_sources, bias=True)

            # Independent K/V projection for each semantic source.
            # This matches W_{CKV,j} in the paper.
            self.cross_to_kv = nn.ModuleList([
                nn.Linear(dim, dim * 2, bias=True) for _ in range(max_sources)
            ])

            self.to_q = None
            self.to_kv = None
            self.wavelet_modulator = None
        else:
            self.to_q = nn.Linear(dim, dim, bias=True)
            self.to_kv = nn.Linear(dim, dim * 2, bias=True)
            self.wavelet_modulator = DynamicSubbandDepthModulator(dim, total_depth=total_depth)
            self.cross_to_q = None
            self.cross_to_kv = None

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        kv_input: Union[torch.Tensor, List[torch.Tensor]],
        q_input: Optional[torch.Tensor] = None,
        depth_level: int = 0,
        is_cross: bool = False,
    ):
        if is_cross:
            assert self.attn_mode == "cross", "This QKVProjection was initialized for self attention."
            assert q_input is not None, "Cross attention requires q_input."
            assert isinstance(kv_input, (list, tuple)), "Cross attention expects a list of KV sources."
            assert self.cross_to_q is not None and self.cross_to_kv is not None

            sources = list(kv_input)
            num_sources = len(sources)
            assert 1 <= num_sources <= self.max_sources, (
                f"num_sources={num_sources} exceeds max_sources={self.max_sources}."
            )

            b_, nq, _ = q_input.shape

            # B_, Nq, max_sources*C -> max_sources, B_, H, Nq, Dh
            q_all = self.cross_to_q(q_input).view(
                b_, nq, self.max_sources, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4).contiguous()

            q_list, k_list, v_list = [], [], []
            for sid, src in enumerate(sources):
                q_j = q_all[sid]  # B_, H, Nq, Dh

                kv_j = self.cross_to_kv[sid](src).view(
                    src.shape[0], src.shape[1], 2, self.num_heads, self.head_dim
                ).permute(2, 0, 3, 1, 4).contiguous()

                k_j, v_j = kv_j[0], kv_j[1]
                q_list.append(q_j)
                k_list.append(k_j)
                v_list.append(v_j)

            return q_list, k_list, v_list, None, None

        else:
            assert self.attn_mode == "self", "This QKVProjection was initialized for cross attention."
            assert isinstance(kv_input, torch.Tensor), "Self attention expects tensor input."
            assert self.wavelet_modulator is not None and self.to_q is not None and self.to_kv is not None

            # SIR/WFDA-style modulation only for self-attention query.
            q_tokens = self.wavelet_modulator(kv_input, depth_level=depth_level)
            q = self._reshape_heads(self.to_q(q_tokens))

            kv = self.to_kv(kv_input).view(
                kv_input.shape[0],
                kv_input.shape[1],
                2,
                self.num_heads,
                self.head_dim,
            )
            kv = kv.permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]

            return q, k, v, None, None


class WindowResidualAttention(nn.Module):
    """
    Window attention used by SREB/CRRB.

    For SREB:
        self-attention with DynamicSubbandDepthModulator on Q.

    For CRRB/WCVA:
        source-wise cross attention. Each semantic source is processed by an
        independent Q/K/V interaction, and all source-specific responses are
        concatenated and projected back to C channels.
    """

    def __init__(
        self,
        dim: int,
        win_size: int,
        num_heads: int,
        block: str,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        total_depth: int = 4,
        max_sources: int = 4,
        attn_mode: str = "self",
    ):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        self.block = block
        assert attn_mode in ("self", "cross"), f"Unsupported attn_mode={attn_mode}."
        self.attn_mode = attn_mode
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_sources = max_sources

        self.qkv = QKVProjection(
            dim=dim,
            num_heads=num_heads,
            block=block,
            total_depth=total_depth,
            max_sources=max_sources,
            attn_mode=attn_mode,
        )

        self.rel_pos = nn.Parameter(
            torch.zeros(1, num_heads, win_size * win_size, win_size * win_size)
        )
        _trunc_normal_(self.rel_pos, std=0.02)

        self.attn_drop = nn.Dropout(attn_drop)

        # Cross attention concatenates max_sources responses. Missing source
        # slots are zero-padded before projection.
        proj_in_dim = dim * max_sources if self.attn_mode == "cross" else dim
        self.proj = nn.Linear(proj_in_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # These are kept only for compatibility with analysis scripts.
        self.last_source_weights = None
        self.last_source_logits = None
        self.last_source_attention_mass = None
        self.last_source_entropy = None

    def _single_source_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + rel_pos
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(q.shape[0], q.shape[2], self.dim)
        return out

    def forward(
        self,
        kv_input: Union[torch.Tensor, List[torch.Tensor]],
        q_input: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        depth_level: int = 0,
        is_cross: bool = False,
    ) -> torch.Tensor:

        if is_cross:
            assert self.attn_mode == "cross", "This attention layer was initialized for self attention."
            assert q_input is not None
            assert isinstance(kv_input, (list, tuple))

            num_sources = len(kv_input)
            assert 1 <= num_sources <= self.max_sources

            q_list, k_list, v_list, _, _ = self.qkv(
                kv_input,
                q_input=q_input,
                depth_level=depth_level,
                is_cross=True,
            )

            # Independent source-wise attention: Q_j only interacts with K_j,V_j.
            source_outputs = []
            for q_j, k_j, v_j in zip(q_list, k_list, v_list):
                source_outputs.append(
                    self._single_source_attention(q_j, k_j, v_j, self.rel_pos)
                )

            x = torch.cat(source_outputs, dim=-1)  # B_, Nq, S*C

            # Pad to max_sources*C so each DMA block has a fixed projection size.
            if num_sources < self.max_sources:
                pad_dim = (self.max_sources - num_sources) * self.dim
                x = F.pad(x, (0, pad_dim), mode="constant", value=0.0)

            self.last_source_weights = None
            self.last_source_logits = None
            self.last_source_entropy = None
            self.last_source_attention_mass = None

        else:
            assert self.attn_mode == "self", "This attention layer was initialized for cross attention."
            assert isinstance(kv_input, torch.Tensor)

            q, k, v, _, _ = self.qkv(
                kv_input,
                q_input=None,
                depth_level=depth_level,
                is_cross=False,
            )

            b_, nq, _ = kv_input.shape
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn + self.rel_pos

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(b_ // nW, nW, self.num_heads, nq, nq)
                attn = attn + mask.unsqueeze(0).unsqueeze(2)
                attn = attn.view(-1, self.num_heads, nq, nq)

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

            self.last_source_weights = None
            self.last_source_logits = None
            self.last_source_entropy = None
            self.last_source_attention_mass = None

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        block: str,
        input_resolution: int,
        num_heads: int,
        win_size: int = 8,
        shift_size: int = 0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        layer_index: int = 0,
        total_depth: int = 4,
        max_sources: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.h = input_resolution
        self.w = input_resolution
        self.win_size = min(win_size, input_resolution)
        self.shift_size = 0 if input_resolution <= win_size else shift_size
        if self.shift_size >= self.win_size:
            self.shift_size = self.win_size // 2
        self.block = block
        self.layer_index = layer_index
        self.attn_mode = "cross" if (block == "DMA" and layer_index == 0) else "self"

        self.norm_kv = norm_layer(dim)
        self.norm_q = norm_layer(dim) if self.attn_mode == "cross" else nn.Identity()
        self.attn = WindowResidualAttention(
            dim=dim,
            win_size=self.win_size,
            num_heads=num_heads,
            block=block,
            attn_drop=attn_drop,
            proj_drop=drop,
            total_depth=total_depth,
            max_sources=max_sources,
            attn_mode=self.attn_mode,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _make_shift_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size <= 0:
            return None
        img_mask = torch.zeros((1, self.h, self.w, 1), device=device)
        h_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.win_size)
        mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _partition_tokens(self, x_tokens: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        b, l, c = x_tokens.shape
        x = norm(x_tokens).view(b, self.h, self.w, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, c)
        return x_windows

    def forward(self,
                kv_input: Union[torch.Tensor, List[torch.Tensor]],
                q_input: Optional[torch.Tensor] = None,
                depth_level: int = 0,
                is_cross: bool = False) -> torch.Tensor:
        expected_cross = self.attn_mode == "cross"
        assert is_cross == expected_cross, (
            f"Layer mode mismatch: attn_mode={self.attn_mode}, is_cross={is_cross}."
        )

        q_windows = None
        if q_input is not None:
            q_windows = self._partition_tokens(q_input, self.norm_q)

        if is_cross:
            assert isinstance(kv_input, (list, tuple)), "CRRB cross attention expects a list of sources."
            kv_windows = [self._partition_tokens(kv, self.norm_kv) for kv in kv_input]
            attn_mask = None  # cross-source token length differs; no shifted mask here.
        else:
            assert isinstance(kv_input, torch.Tensor)
            kv_windows = self._partition_tokens(kv_input, self.norm_kv)
            attn_mask = self._make_shift_mask(kv_windows.device)

        attn_windows = self.attn(
            kv_windows, q_input=q_windows, mask=attn_mask, depth_level=depth_level, is_cross=is_cross
        )
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, self.dim)
        shifted = window_reverse(attn_windows, self.win_size, self.h, self.w)
        if self.shift_size > 0:
            shifted = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        out = shifted.view(-1, self.h * self.w, self.dim)
        return self.drop_path(out)
# -----------------------------------------------------------------------------
# New FAA/DMA outer structures: detail/context residual decomposition and fusion
# -----------------------------------------------------------------------------
class LightweightChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ConvFFN(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.gamma.tanh() * self.net(x))

class ContextGuidedDetailPreparation(nn.Module):
    """
    Context-guided Detail Preparation Gate (CDPG).

    This module prepares features before SIR/CRRB by:
        1) sharing local information to reduce window partition artifacts;
        2) extracting a smooth context reference;
        3) extracting high-frequency detail candidates;
        4) using context to gate reliable details.

    It is not the main attention mechanism. It is a lightweight
    preconditioning block before SIR/CRRB.
    """

    def __init__(self, dim: int):
        super().__init__()

        # 1) Information sharing branch.
        # Conv + dilated depthwise conv improves local interaction before window attention.
        self.info_share = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 2) Context reference branch.
        # Smooth context is further modeled by a larger depthwise filter.
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=5,
                padding=2,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 3) Detail candidate branch.
        # High-frequency residual is projected into target/clutter detail candidates.
        self.detail_branch = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 4) Context-guided detail gate.
        # It decides which high-frequency residuals are reliable.
        self.detail_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local smooth context approximation.
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # High-frequency detail candidate.
        detail_raw = x - smooth

        # Information sharing before window-based SIR/CRRB.
        shared = self.info_share(x)

        # Stable context reference.
        context = self.context_branch(smooth)

        # Target/clutter detail candidates.
        detail = self.detail_branch(detail_raw)

        # Context-guided selection of reliable details.
        gate = self.detail_gate(torch.cat([shared, context, detail], dim=1))

        # Prepared feature for SIR/CRRB.
        prepared = shared + context + gate * detail

        return self.act(x + self.gamma.tanh() * self.out_proj(prepared))

class EvidenceCalibratedResidualUpdate(nn.Module):
    """
    Evidence-Calibrated Residual Update (ECRU).

    CDPG first prepares context-detail features.
    SIR/CRRB then produces frequency-aware or cross-source evidence.
    ECRU aligns the current feature and the evidence into the same
    residual space, estimates evidence reliability, and updates the
    feature through a calibrated residual path.
    """

    def __init__(self, dim: int):
        super().__init__()

        # Project current feature into the residual evidence space.
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Project SIR/CRRB response into the same residual evidence space.
        self.update_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Reliability gate estimated from:
        # 1) current reference feature,
        # 2) SIR/CRRB evidence,
        # 3) their discrepancy.
        self.gate_gen = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.tensor(0.01))

        # Make the gate start from around 0.5 for stable training.
        nn.init.zeros_(self.gate_gen[-2].weight)
        nn.init.zeros_(self.gate_gen[-2].bias)

    def forward(self, x: torch.Tensor, attn_response: torch.Tensor) -> torch.Tensor:
        x_ref = self.x_proj(x)
        update = self.update_proj(attn_response)

        discrepancy = torch.abs(x_ref - update)

        gate = self.gate_gen(torch.cat([x_ref, update, discrepancy], dim=1))
        calibrated_update = gate * update

        return x + self.gamma.tanh() * calibrated_update
class FAADMAblock(nn.Module):
    """
    Backward-compatible block name.

    block='FAA' -> SREB: spectral residual encoding block.
    block='DMA' -> CRRB: cross-source residual refinement block.
    """

    def __init__(self,
                 channels_sumdim: int,
                 channels_list: Sequence[int],
                 dim: int,
                 input_resolution: int,
                 win_size: int = 8,
                 shift_size: int = 0,
                 num_heads: int = 1,
                 depth: int = 2,
                 drop_ratio: float = 0.0,
                 attn_drop_ratio: float = 0.0,
                 drop_path: Union[float, Sequence[float]] = 0.0,
                 norm_layer: nn.Module = nn.LayerNorm,
                 block: Optional[str] = None,
                 total_depth: int = 4):
        super().__init__()
        self.block = block or "FAA"
        self.depth = depth
        self.dim = dim
        self.input_resolution = input_resolution
        self.channels_list = list(channels_list)

        if isinstance(drop_path, (list, tuple)):
            dp = list(drop_path)
            if len(dp) < depth:
                dp = dp + [dp[-1] if dp else 0.0] * (depth - len(dp))
        else:
            dp = [float(drop_path)] * depth

        self.in_proj = nn.Sequential(
            nn.Conv2d(channels_sumdim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.pre_mixers = nn.ModuleList([
            ContextGuidedDetailPreparation(dim) for _ in range(depth)
        ])

        max_sources = max(len(self.channels_list), 1)
        self.attn_blocks = nn.ModuleList([
            WindowAttentionBlock(
                dim=dim,
                block=self.block,
                input_resolution=input_resolution,
                num_heads=num_heads,
                win_size=win_size,
                shift_size=shift_size if (i % 2 == 0) else win_size // 2,
                drop=drop_ratio,
                attn_drop=attn_drop_ratio,
                drop_path=dp[i],
                norm_layer=norm_layer,
                layer_index=i,
                total_depth=total_depth,
                max_sources=max_sources,
            )
            for i in range(depth)
        ])
        # Remove the extra local context-detail branch.
        # CDPG/pre_mixer already performs context-detail preparation.

        self.attn_updates = nn.ModuleList([
            EvidenceCalibratedResidualUpdate(dim) for _ in range(depth)
        ])
        self.ffns = nn.ModuleList([ConvFFN(dim, hidden_ratio=2.0, drop=drop_ratio) for _ in range(depth)])

        # Used only by CRRB/DMA: project all sources to the current resolution/channel.
        # FAA/SREB blocks do not instantiate these unused parameters, which keeps the
        # reported parameter count consistent with the actually executed graph.
        if self.block == "DMA":
            self.kv_input_proj = nn.ModuleDict({
                str(ch): nn.Sequential(
                    nn.Conv2d(ch, dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                ) for ch in sorted(set(self.channels_list))
            })
            self.guidance_up = nn.ConvTranspose2d(
                in_channels=2 * dim, out_channels=dim, kernel_size=2, stride=2
            )

            # Branch-wise CGDP/CDPG before CRRB cross attention.
            # The query guidance branch and every KV source branch are prepared
            # independently, so source identity is preserved before WCVA/MRCA.
            self.cross_q_prep = ContextGuidedDetailPreparation(dim)
            self.cross_kv_preps = nn.ModuleList([
                ContextGuidedDetailPreparation(dim) for _ in self.channels_list
            ])
        else:
            self.kv_input_proj = nn.ModuleDict()
            self.guidance_up = None
            self.cross_q_prep = None
            self.cross_kv_preps = nn.ModuleList()

    def _project_sources(self, kv_input: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        assert self.block == "DMA", "_project_sources is only used by DMA/CRRB blocks."
        sources = []
        for ch, kv in zip(self.channels_list, kv_input):
            key = str(ch)
            if key not in self.kv_input_proj:
                raise KeyError(f"No projection layer for source channel {ch}.")
            sources.append(self.kv_input_proj[key](kv))
        return sources

    def _prepare_guidance(self, q_input: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # In the original topology, q_input normally has 2*dim channels and half resolution.
        if q_input.shape[1] == 2 * self.dim:
            assert self.guidance_up is not None
            q = self.guidance_up(q_input)
        elif q_input.shape[1] == self.dim:
            q = q_input
            if q.shape[-2:] != target_hw:
                q = F.interpolate(q, size=target_hw, mode="bilinear", align_corners=False)
        else:
            # Robust fallback for non-doubling channel settings.
            q = F.interpolate(q_input, size=target_hw, mode="bilinear", align_corners=False)
            q = q[:, :self.dim] if q.shape[1] >= self.dim else F.pad(q, (0, 0, 0, 0, 0, self.dim - q.shape[1]))
        if q.shape[-2:] != target_hw:
            q = F.interpolate(q, size=target_hw, mode="bilinear", align_corners=False)
        return q

    def forward(self,
                x: torch.Tensor,
                kv_input: Optional[Sequence[torch.Tensor]] = None,
                q_input: Optional[torch.Tensor] = None,
                depth_level: int = 0) -> torch.Tensor:
        x = self.in_proj(x)
        h, w = x.shape[-2:]
        if self.block == "DMA":
            assert kv_input is not None and q_input is not None, "DMA/CRRB requires kv_input and q_input."
            kv_sources = self._project_sources(kv_input)
            q_guidance = self._prepare_guidance(q_input, target_hw=(h, w))
        else:
            kv_sources = None
            q_guidance = None
        for i in range(self.depth):
            # 1) Context-detail preparation for the main residual carrier.
            x = self.pre_mixers[i](x)

            # 2) SIR/CRRB evidence extraction.
            if self.block == "DMA" and i == 0:
                assert self.cross_q_prep is not None
                assert len(self.cross_kv_preps) == len(kv_sources)

                # Prepare the query guidance branch independently.
                q_guidance_p = self.cross_q_prep(q_guidance)
                q_tokens = bchw_to_blc(q_guidance_p)

                # Prepare each KV source branch independently.
                # Do not concatenate them before CGDP/CDPG; otherwise the
                # multi-source identity used by WCVA/MRCA would be weakened.
                kv_sources_p = [
                    prep(src) for prep, src in zip(self.cross_kv_preps, kv_sources)
                ]
                kv_tokens = [bchw_to_blc(src) for src in kv_sources_p]

                attn_tokens = self.attn_blocks[i](
                    kv_input=kv_tokens,
                    q_input=q_tokens,
                    depth_level=depth_level,
                    is_cross=True,
                )
            else:
                x_tokens = bchw_to_blc(x)
                attn_tokens = self.attn_blocks[i](
                    kv_input=x_tokens,
                    q_input=None,
                    depth_level=depth_level,
                    is_cross=False,
                )

            attn_response = blc_to_bchw(attn_tokens, h, w)

            # 3) Evidence-calibrated residual update.
            x = self.attn_updates[i](x, attn_response)

            # 4) ConvFFN refinement.
            x = self.ffns[i](x)
        return x


# -----------------------------------------------------------------------------
# Full model: FADMFormer topology retained, internals refactored
# -----------------------------------------------------------------------------

class FADMFormer(nn.Module):
    def __init__(self,
                 input_channels: int,
                 nb_filter: Sequence[int],
                 depth: Sequence[int],
                 num_heads: Sequence[Sequence[int]],
                 win_size: int = 8,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 img_size: int = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.down1 = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=4, stride=2, padding=1)

        dpr3 = [x.item() for x in torch.linspace(0, drop_path, depth[2] * 3)]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path, depth[1] * 4)]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path, depth[0] * 5)]

        self.FAA1_1 = FAADMAblock(
            channels_sumdim=input_channels, channels_list=[nb_filter[0]], dim=nb_filter[0], depth=2,
            num_heads=num_heads[0][0], block="FAA", drop_ratio=drop, attn_drop_ratio=attn_drop,
            drop_path=[0, 0], win_size=win_size, input_resolution=img_size, total_depth=4,
        )
        self.FAA2_1 = FAADMAblock(
            channels_sumdim=nb_filter[1], channels_list=[nb_filter[1]], dim=nb_filter[1], depth=2,
            num_heads=num_heads[1][0], block="FAA", drop_ratio=drop, attn_drop_ratio=attn_drop,
            drop_path=[0, 0], win_size=win_size, input_resolution=img_size // 2, total_depth=4,
        )
        self.FAA3_1 = FAADMAblock(
            channels_sumdim=nb_filter[2], channels_list=[nb_filter[2]], dim=nb_filter[2], depth=2,
            num_heads=num_heads[2][0], block="FAA", drop_ratio=drop, attn_drop_ratio=attn_drop,
            drop_path=[0, 0], win_size=win_size * 2, input_resolution=img_size // 4, total_depth=4,
        )
        self.FAA4_1 = FAADMAblock(
            channels_sumdim=nb_filter[3], channels_list=[nb_filter[3]], dim=nb_filter[3], depth=2,
            num_heads=num_heads[3][0], block="FAA", drop_ratio=drop, attn_drop_ratio=attn_drop,
            drop_path=[0, 0], win_size=win_size * 4, input_resolution=img_size // 8, total_depth=4,
        )

        self.DMA1_2 = FAADMAblock(
            channels_sumdim=nb_filter[0] + nb_filter[1], channels_list=[nb_filter[0], nb_filter[1]],
            dim=nb_filter[0], depth=depth[0], num_heads=num_heads[0][1], block="DMA", drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr1[0:depth[0]], win_size=win_size, input_resolution=img_size,
        )
        self.DMA2_2 = FAADMAblock(
            channels_sumdim=nb_filter[0] + nb_filter[1] + nb_filter[2], channels_list=[nb_filter[1], nb_filter[2], nb_filter[0]],
            dim=nb_filter[1], block="DMA", depth=depth[1], num_heads=num_heads[1][1], drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr2[0:depth[1]], win_size=win_size, input_resolution=img_size // 2,
        )
        self.DMA3_2 = FAADMAblock(
            channels_sumdim=nb_filter[1] + nb_filter[2] + nb_filter[3], channels_list=[nb_filter[2], nb_filter[3], nb_filter[1]],
            dim=nb_filter[2], block="DMA", depth=depth[2], num_heads=num_heads[2][1], drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr3[0:depth[2]], win_size=win_size * 2, input_resolution=img_size // 4,
        )
        self.DMA1_3 = FAADMAblock(
            channels_sumdim=nb_filter[0] * 2 + nb_filter[1], channels_list=[nb_filter[0], nb_filter[0], nb_filter[1]],
            dim=nb_filter[0], block="DMA", depth=depth[0], num_heads=num_heads[0][2], drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr1[depth[0]:depth[0] * 2], win_size=win_size, input_resolution=img_size,
        )
        self.DMA2_3 = FAADMAblock(
            channels_sumdim=nb_filter[0] + nb_filter[1] * 2 + nb_filter[2], channels_list=[nb_filter[1], nb_filter[1], nb_filter[2], nb_filter[0]],
            dim=nb_filter[1], block="DMA", depth=depth[1], num_heads=num_heads[1][2], drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr2[depth[1]:depth[1] * 2], win_size=win_size, input_resolution=img_size // 2,
        )
        self.DMA1_4 = FAADMAblock(
            channels_sumdim=nb_filter[0] * 3 + nb_filter[1], channels_list=[nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]],
            dim=nb_filter[0], depth=depth[0], block="DMA", num_heads=num_heads[0][3], drop_ratio=drop,
            attn_drop_ratio=attn_drop, drop_path=dpr1[depth[0] * 2:depth[0] * 3],
            win_size=win_size // 2 if img_size == 256 else win_size, input_resolution=img_size,
        )

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.apply(self._init_weights)
        self._reset_special_inits()

    def _reset_special_inits(self) -> None:
        for m in self.modules():
            if isinstance(m, DynamicSubbandDepthModulator):
                nn.init.zeros_(m.global_controller[-1].weight)
                nn.init.zeros_(m.global_controller[-1].bias)

            if isinstance(m, EvidenceCalibratedResidualUpdate):
                nn.init.zeros_(m.gate_gen[-2].weight)
                nn.init.zeros_(m.gate_gen[-2].bias)

            if isinstance(m, QKVProjection) and m.attn_mode == "cross":
                if m.source_kv_delta is not None:
                    for layer in m.source_kv_delta:
                        nn.init.zeros_(layer.weight)
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, iptimage: torch.Tensor) -> torch.Tensor:
        F1_1 = self.FAA1_1(x=iptimage, depth_level=0)
        F2_1 = self.FAA2_1(self.down1(F1_1), depth_level=1)
        F3_1 = self.FAA3_1(self.down2(F2_1), depth_level=2)
        F4_1 = self.FAA4_1(self.down3(F3_1), depth_level=3)

        D3_2 = self.DMA3_2(
            torch.cat([F3_1, self.up(F4_1), self.down(F2_1)], dim=1),
            [F3_1, self.up(F4_1), self.down(F2_1)],
            F4_1,
            depth_level=2,
        )
        D2_2 = self.DMA2_2(
            torch.cat([F2_1, self.up(F3_1), self.down(F1_1)], dim=1),
            [F2_1, self.up(F3_1), self.down(F1_1)],
            D3_2,
            depth_level=1,
        )
        D1_2 = self.DMA1_2(
            torch.cat([F1_1, self.up(F2_1)], dim=1),
            [F1_1, self.up(F2_1)],
            D2_2,
            depth_level=0,
        )
        D2_3 = self.DMA2_3(
            torch.cat([F2_1, D2_2, self.up(D3_2), self.down(D1_2)], dim=1),
            [F2_1, D2_2, self.up(D3_2), self.down(D1_2)],
            D3_2,
            depth_level=1,
        )
        D1_3 = self.DMA1_3(
            torch.cat([F1_1, D1_2, self.up(D2_2)], dim=1),
            [F1_1, D1_2, self.up(D2_2)],
            D2_3,
            depth_level=0,
        )
        D1_4 = self.DMA1_4(
            torch.cat([F1_1, D1_2, D1_3, self.up(D2_3)], dim=1),
            [F1_1, D1_2, D1_3, self.up(D2_3)],
            D2_3,
            depth_level=0,
        )
        return self.final(D1_4)


# Suggested names for the paper draft:
#   FAA -> SREB: Spectral Residual Encoding Block
#   DMA -> CRRB: Cross-source Residual Refinement Block
#   WFDA -> SDCA: Subband-Depth Competitive Attention
#   WCVA -> MRCA: Multi-source Residual Cross Attention


if __name__ == "__main__":
    # A light shape test. CPU thread overhead can dominate small-window attention.
    torch.set_num_threads(1)
    # Use smaller channels/resolution for CPU sanity.
    model = FADMFormer(
        input_channels=1,
        nb_filter=[4, 8, 16, 32],
        depth=[1, 1, 1],
        num_heads=[[1, 1, 1, 1], [1, 1, 1], [1, 1, 1], [1]],
        win_size=4,
        img_size=32,
    )
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    print("output:", tuple(y.shape))
if __name__ == "__main__":
    import time
    import torch

    # =========================
    # 1. Model config
    # =========================
    # 这个配置对应你现在表里大约 Params=5.86M 的口径
    input_channels = 1
    img_size = 256
    nb_filter = [16, 32, 64, 128]
    depth = [1, 1, 1]
    num_heads = [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1],
    ]
    win_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FADMFormer(
        input_channels=input_channels,
        nb_filter=nb_filter,
        depth=depth,
        num_heads=num_heads,
        win_size=win_size,
        img_size=img_size,
    ).to(device)

    model.eval()

    dummy = torch.randn(1, input_channels, img_size, img_size).to(device)

    # =========================
    # 2. Params(M)
    # =========================
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = params / 1e6

    # =========================
    # 3. FLOPs(G)
    # =========================
    flops_g = None

    # 推荐优先用 fvcore，attention / matmul 统计相对更完整
    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flops_g = flops.total() / 1e9

        print("FLOPs counted by fvcore.")
    except Exception as e:
        print("fvcore failed:", e)
        print("Try thop instead...")

        try:
            from thop import profile

            with torch.no_grad():
                macs, _ = profile(model, inputs=(dummy,), verbose=False)

            # 注意：很多论文表里直接把 MACs / 1e9 当 FLOPs(G) 写
            flops_g = macs / 1e9

            print("FLOPs counted by thop. This is MACs/1e9 style.")
        except Exception as e2:
            print("thop failed:", e2)
            flops_g = -1

    # =========================
    # 4. Time(s)
    # =========================
    warmup = 50
    repeats = 200

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)

        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(dummy)
        ender.record()

        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender)
        time_s = elapsed_ms / repeats / 1000.0

    else:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(dummy)
        end = time.perf_counter()

        time_s = (end - start) / repeats

    # =========================
    # 5. Print results
    # =========================
    print("=" * 60)
    print(f"Input size : 1 x {input_channels} x {img_size} x {img_size}")
    print(f"Device     : {device}")
    print(f"Params(M)  : {params_m:.2f}")
    print(f"FLOPs(G)   : {flops_g:.2f}")
    print(f"Time(s)    : {time_s:.3f}")
    print("=" * 60)

    # 直接输出 LaTeX 表格最后一行
    print("LaTeX row:")
    print(
        f"FADMFormer & {params_m:.2f} & {flops_g:.2f} & {time_s:.3f}\\\\"
    )