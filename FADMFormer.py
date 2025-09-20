import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath,trunc_normal_
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
idwt = DWTInverse(mode='zero', wave='haar').to(device)
def dwt_decompose(x):
    x = x.permute(0, 3, 1, 2)
    Yl, Yh = dwt(x)
    Yl = Yl.to(device)
    Yh = [h.to(device) for h in Yh]
    LH, HL, HH = Yh[0][:, :, 0], Yh[0][:, :, 1], Yh[0][:, :, 2]
    return Yl, LH, HL, HH
def dwt_reconstruct(LL, LH, HL, HH):
    Yh = [torch.stack([LH, HL, HH], dim=2)]
    Yh = [y.to(device) for y in Yh]
    out = idwt((LL.to(device), Yh))
    return out
class ConditionalSubbandModulator(nn.Module):
    def __init__(self, in_channels, subband_name, total_depth=5):
        super().__init__()
        self.subband_name = subband_name
        self.subband_embed = nn.Parameter(torch.randn(4))
        self.depth_embed = nn.Parameter(torch.randn(total_depth))
        self.modulator = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x, depth_level):
        subband_idx = {'LL': 0, 'LH': 1, 'HL': 2, 'HH': 3}[self.subband_name]
        cond = self.subband_embed[subband_idx] + self.depth_embed[depth_level]
        cond_weight = torch.sigmoid(cond).view(1, 1, 1, 1)
        modulated = self.modulator(x) * cond_weight
        return modulated
class WaveletAttentionModulator(nn.Module):
    def __init__(self, in_channels, total_depth=5):
        super().__init__()
        self.subbands = nn.ModuleDict({
            name: ConditionalSubbandModulator(in_channels, name, total_depth)
            for name in ['LL', 'LH', 'HL', 'HH']})
    def forward(self, x, depth_level):
        x_img = blc_to_bchw(x).permute(0, 2, 3, 1)
        LL, LH, HL, HH = dwt_decompose(x_img)
        LL_ = self.subbands['LL'](LL, depth_level)
        LH_ = self.subbands['LH'](LH, depth_level)
        HL_ = self.subbands['HL'](HL, depth_level)
        HH_ = self.subbands['HH'](HH, depth_level)
        x_mod = dwt_reconstruct(LL_, LH_, HL_, HH_)
        x_mod = x_mod.permute(0, 2, 3, 1)
        return bchw_to_blc(x_mod.permute(0, 3, 1, 2))
def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows
def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
def bchw_to_blc(x):
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x
def blc_to_bchw(x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).view(B, C, H, W)
    return x
class LinearQKV(nn.Module):
    def __init__(self, dim, block, co_dim, heads=8, i=1):
        super().__init__()
        self.heads = heads
        self.block = block
        if self.block == 'DMA' and i == 0:
            self.to_q = nn.Linear(dim, dim * heads)
        else:
            self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(co_dim, dim * 2)
        self.block = block
        self.wavelet_modulator = WaveletAttentionModulator(dim, total_depth=5)
    def forward(self, x, attn_kv=None, depth_level=None, i=1):
        if self.block == 'DMA' and i == 0:
            for kv in x:
                B_, N, C = kv.shape
        else:
            B_, N, C = x.shape
        if self.block == 'FAA':
            attn_kv = self.wavelet_modulator(x, depth_level=depth_level)
        else:
            pass
        N_kv = attn_kv.size(1)
        if self.block == 'DMA' and i == 0:
            q = self.to_q(attn_kv).reshape(B_, N, 1, self.heads, -1).permute(2, 0, 3, 1, 4)
            kv = torch.stack(x, dim=1)
            kv = self.to_kv(kv).reshape(B_, N_kv, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
            q = q[0]
            v, k = kv[0], kv[1]
            return q, k, v
        else:
            q = self.to_q(attn_kv).reshape(B_, N, 1, 1, -1).permute(2, 0, 3, 1, 4)
            kv = self.to_kv(x).reshape(B_, N_kv, 2, 1, -1).permute(2, 0, 3, 1, 4)
            q = q[0]
            v, k = kv[0], kv[1]
            return q, k, v
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        if in_planes <= ratio:
            ratio = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)
class FastLeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, outdim=32, downsample=1):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim // downsample, hidden_dim),nn.GELU())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),nn.GELU())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, outdim))
    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        return x
class FAADMAblock(nn.Module):
    def __init__(self,channels_sumdim,channels_list,dim,input_resolution,win_size=8,shift_size=0,heads=1,depth=2,
                 drop_ratio=0.,attn_drop_ratio=0.,drop_path=[0],norm_layer=nn.LayerNorm,block=None):
        super(FAADMAblock, self).__init__()
        self.shift_size = shift_size
        self.cbr1 = nn.Sequential(nn.Conv2d(channels_sumdim, dim, 3, 1, 1),nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.channels_list = channels_list
        self.depth = depth
        self.attenbranch = nn.ModuleList([])
        self.block = block
        self.attn_proj = nn.ConvTranspose2d(in_channels=2 * dim, out_channels=dim, kernel_size=2, stride=2, padding=0)
        for i in range(depth):
            num_heads = heads[i] if isinstance(heads, list) else heads
            dp = drop_path[i]
            self.attenbranch.append(WinLNattnFaLeFF(dim=dim, input_resolution=input_resolution, num_heads=num_heads, win_size=win_size,
                                     shift_size=self.shift_size if (i % 2 == 0) else win_size // 2, drop=drop_ratio,
                                     attn_drop=attn_drop_ratio, drop_path=dp, norm_layer=norm_layer, block=block, i=i))
        self.cbrpar1=nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.diacbrpar2=nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=2, dilation=2), nn.BatchNorm2d(dim),nn.ReLU(inplace=True))
        self.cbr2=nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1),nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.cbam=CBAM(dim)
        self.actfun=nn.ReLU(inplace=True)
        self.grconv3br=nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, 1, dim),nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.conv1b=nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        self.kv_inputcbr = nn.ModuleDict({str(n): nn.Sequential(
                nn.Conv2d(n, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ) for n in self.channels_list})
    def forward(self, x, kv_input=None, q_input=None, depth_level=0):
        x = self.cbr1(x)
        if self.block == 'DMA':
            orgn = x
            cbrpar1 = self.cbrpar1(x)
            diacbrpar2 = self.diacbrpar2(x)
            x = self.cbr2(torch.cat([cbrpar1, diacbrpar2], dim=1))
            mainbranch = x
            q_input = bchw_to_blc(self.attn_proj(q_input))
            kv_stack1, kv_stack2 = [], []
            for j, kv in zip(self.channels_list, kv_input):
                kv = self.kv_inputcbr[str(j)](kv)
                kv = bchw_to_blc(kv)
                kv_stack1.append(kv)
            for i in range(self.depth):
                if i == 0:
                    attenbranch = self.attenbranch[i](kv_input=kv_stack1, q_input=q_input, depth_level=depth_level, i=i)
                    attenbranch = blc_to_bchw(attenbranch)
                    mainbranch = self.grconv3br(mainbranch)
                    x = mainbranch * attenbranch
                    x = self.conv1b(x)
                    x = self.cbam(x)
                    x = x + orgn
                    x = self.actfun(x)
                else:
                    orgn = x
                    cbrpar1 = self.cbrpar1(x)
                    diacbrpar2 = self.diacbrpar2(x)
                    x = self.cbr2(torch.cat([cbrpar1, diacbrpar2], dim=1))
                    mainbranch = x
                    x = bchw_to_blc(x)
                    attenbranch = self.attenbranch[i](kv_input=x, q_input=q_input, depth_level=depth_level, i=i)
                    attenbranch = blc_to_bchw(attenbranch)
                    mainbranch = self.grconv3br(mainbranch)
                    x = mainbranch * attenbranch
                    x = self.conv1b(x)
                    x = self.cbam(x)
                    x = x + orgn
                    x = self.actfun(x)
            return x
        else:
            for i in range(self.depth):
                orgn = x
                cbrpar1 = self.cbrpar1(x)
                diacbrpar2 = self.diacbrpar2(x)
                x = self.cbr2(torch.cat([cbrpar1, diacbrpar2], dim=1))
                mainbranch = x
                x = bchw_to_blc(x)
                attenbranch = self.attenbranch[i](kv_input=x, q_input=q_input, depth_level=depth_level)
                attenbranch = blc_to_bchw(attenbranch)
                mainbranch = self.grconv3br(mainbranch)
                x = mainbranch * attenbranch
                x = self.conv1b(x)
                x = self.cbam(x)
                x = x + orgn
                x = self.actfun(x)
            return x
class WFDAWCVA(nn.Module):
    def __init__(self, dim, co_dim, win_size, num_heads, block, attn_drop=0., proj_drop=0., i=1):
        super().__init__()
        self.win_size = win_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.re_pos = nn.Parameter(torch.zeros(1, 1, self.win_size ** 2, self.win_size ** 2))
        trunc_normal_(self.re_pos, std=.02)
        self.block = block
        self.linearqkv = LinearQKV(dim=dim, co_dim=co_dim, heads=num_heads, block=block, i=i)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_kv = nn.Linear(dim, dim * 2)
        if self.block == 'DMA' and i == 0:
            self.proj = nn.Linear(dim * num_heads, dim)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, attn_kv=None, mask=None, depth_level=0, i=0):
        if self.block == 'DMA' and i == 0:
            for kv in x:
                B_, N, C = kv.shape
        else:
            B_, N, C = x.shape
        if self.block == 'DMA' and i == 0:
            q, k, v = self.linearqkv(x, attn_kv, i=i)
            attn = (q / (q.shape[-1] ** 0.5)) @ k.transpose(-2, -1)
            attn = attn + self.re_pos
        else:
            q, k, v = self.linearqkv(x, attn_kv, depth_level=depth_level)
            q = q * self.scale
            attn = (q / (q.shape[-1] ** 0.5)) @ k.transpose(-2, -1)
            attn = attn + self.re_pos
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=1)
            attn = attn.view(B_ // nW, nW, 1, N, N * 1) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, 1, N, N * 1)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class WinLNattnFaLeFF(nn.Module):
    def __init__(self, dim, block, input_resolution, num_heads, win_size=8, shift_size=0,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, i=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.win_size = win_size
        self.shift_size = shift_size
        self.block = block
        self.H = self.input_resolution
        self.W = self.input_resolution
        if self.input_resolution <= self.win_size:
            self.shift_size = 0
            self.win_size = self.input_resolution
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = WFDAWCVA(dim=dim, co_dim=dim, win_size=self.win_size,num_heads=num_heads, block=block, attn_drop=attn_drop, proj_drop=drop, i=i)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.leff = FastLeFF(dim, dim * 4, outdim=dim, downsample=1)
    def forward(self, kv_input, q_input=None,depth_level=0, i=1):
        if q_input is not None:
            B, L, C_co = q_input.shape
            q_input = self.norm2(q_input)
            q_input = q_input.view(B, self.H, self.W, C_co)
        if self.shift_size > 0:
            if q_input is not None:
                shifted_co = torch.roll(q_input, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            if q_input is not None:
                shifted_co = q_input
        co_windows = None
        if q_input is not None:
            co_windows = window_partition(shifted_co, self.win_size)
            co_windows = co_windows.view(-1, self.win_size * self.win_size, C_co)
        self.attn_mask = None
        if self.shift_size > 0:
            shift_mask = torch.zeros((1, self.H, self.W, 1)).to(device)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            self.attn_mask = shift_attn_mask
        if self.block == 'DMA' and i == 0:
            x_windowstack = []
            for kv in kv_input:
                B, L, C = kv.shape
                kv = self.norm1(kv)
                kv = kv.view(B, self.H, self.W, C)
                if self.shift_size > 0:
                    shifted_x = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                else:
                    shifted_x = kv
                x_windows = window_partition(shifted_x, self.win_size)
                x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
                x_windowstack.append(x_windows)
        else:
            B, L, C = kv_input.shape
            x = self.norm1(kv_input)
            x = x.view(B, self.H, self.W, C)
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
            x_windowstack = window_partition(shifted_x, self.win_size)
            x_windowstack = x_windowstack.view(-1, self.win_size * self.win_size, C)
        attn_windows = self.attn(x_windowstack, co_windows, mask=self.attn_mask, depth_level=depth_level, i=i)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, self.H, self.W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, self.H * self.W, -1)
        x = self.drop_path(x)
        x = torch.sigmoid(self.leff(self.norm3(x)))
        return x
class FADMFormer(nn.Module):
    def __init__(self, input_channels, channel,depth, num_heads, win_size=8, drop=0., attn_drop=0., drop_path=0., img_size=512):
        super(FADMFormer, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down1 = nn.Conv2d(channel[0],channel[1], kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(channel[1], channel[2], kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(channel[2], channel[3], kernel_size=4, stride=2, padding=1)
        dpr2 = [x.item() for x in torch.linspace(0, drop_path, depth[2] * 1)]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path, depth[1] * 2)]
        dpr0 = [x.item() for x in torch.linspace(0, drop_path, depth[0] * 3)]
        self.FAA1_1 = FAADMAblock(channels_sumdim=input_channels, channels_list=[channel[0]], dim=channel[0], depth=2,heads=num_heads[0][0],
                                      block='FAA', drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=[0,0],win_size=win_size, input_resolution=img_size)
        self.FAA2_1 = FAADMAblock(channels_sumdim=channel[1], channels_list=[channel[1]], dim=channel[1], depth=2,heads=num_heads[1][0],
                                      block='FAA', drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=[0,0],win_size=win_size, input_resolution=img_size // 2)
        self.FAA3_1 = FAADMAblock(channels_sumdim=channel[2], channels_list=[channel[2]], dim=channel[2], depth=2,heads=num_heads[2][0],
                                      block='FAA', drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=[0,0],win_size=win_size * 2, input_resolution=img_size // 4)
        self.FAA4_1 = FAADMAblock(channels_sumdim=channel[3], channels_list=[channel[3]], dim=channel[3], depth=2,heads=num_heads[3][0],
                                      block='FAA', drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=[0,0],win_size=win_size * 4, input_resolution=img_size // 8)
        self.DMA1_2 = FAADMAblock(channels_sumdim=channel[0] + channel[1], channels_list=[channel[0], channel[1]],
                                      dim=channel[0], depth=depth[0], heads=num_heads[0][1], block='DMA', drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr0[0:depth[0]], win_size=win_size, input_resolution=img_size)
        self.DMA2_2 = FAADMAblock(channels_sumdim=channel[0] + channel[1] + channel[2],channels_list=[channel[1], channel[2], channel[0]],
                                      dim=channel[1], block='DMA', depth=depth[1], heads=num_heads[1][1], drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr1[0:depth[1]], win_size=win_size, input_resolution=img_size // 2)
        self.DMA3_2 = FAADMAblock(channels_sumdim=channel[1] +channel[2] + channel[3],channels_list=[channel[2], channel[3], channel[1]],
                                      dim=channel[2], block='DMA', depth=depth[2], heads=num_heads[2][1], drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr2[0:depth[2]], win_size=win_size * 2, input_resolution=img_size // 4)
        self.DMA1_3 = FAADMAblock(channels_sumdim=channel[0] * 2 + channel[1],channels_list=[channel[0], channel[0], channel[1]],
                                      dim=channel[0], block='DMA', depth=depth[0], heads=num_heads[0][2], drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr0[depth[0]:depth[0] * 2], win_size=win_size,input_resolution=img_size)
        self.DMA2_3 = FAADMAblock(channels_sumdim=channel[0] +channel[1] * 2 +channel[2],channels_list=[channel[1], channel[1], channel[2], channel[0]],
                                      dim=channel[1], block='DMA', depth=depth[1], heads=num_heads[1][2], drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr1[depth[1]:depth[1] * 2], win_size=win_size,input_resolution=img_size // 2)
        self.DMA1_4 = FAADMAblock(channels_sumdim=channel[0] * 3 + channel[1],channels_list=[channel[0],channel[0], channel[0], channel[1]],
                                      dim=channel[0], depth=depth[0], block='DMA', heads=num_heads[0][3], drop_ratio=drop,
                                      attn_drop_ratio=attn_drop,drop_path=dpr0[depth[0] * 2:depth[0] * 3],
                                      win_size=win_size // 2 if img_size == 256 else win_size,input_resolution=img_size)
        self.final = nn.Conv2d(channel[0], 1, kernel_size=1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    def forward(self, iptimage):
        F1_1 = self.FAA1_1(x=iptimage, depth_level=0)
        F2_1 = self.FAA2_1(self.down1(F1_1), depth_level=1)
        F3_1 = self.FAA3_1(self.down2(F2_1), depth_level=2)
        F4_1 = self.FAA4_1(self.down3(F3_1), depth_level=3)
        D3_2 = self.DMA3_2(torch.cat([F3_1, self.up(F4_1), self.down(F2_1)], 1),[F3_1, self.up(F4_1), self.down(F2_1)], F4_1)
        D2_2 = self.DMA2_2(torch.cat([F2_1, self.up(F3_1), self.down(F1_1)], 1),[F2_1, self.up(F3_1), self.down(F1_1)], D3_2)
        D1_2 = self.DMA1_2(torch.cat([F1_1, self.up(F2_1)], 1), [F1_1, self.up(F2_1)], D2_2)
        D2_3 = self.DMA2_3(torch.cat([F2_1, D2_2, self.up(D3_2), self.down(D1_2)], 1),[F2_1, D2_2, self.up(D3_2), self.down(D1_2)], D3_2)
        D1_3 = self.DMA1_3(torch.cat([F1_1, D1_2, self.up(D2_2)], 1), [F1_1, D1_2, self.up(D2_2)], D2_3)
        D1_4 = self.DMA1_4(torch.cat([F1_1, D1_2, D1_3, self.up(D2_3)], 1), [F1_1, D1_2, D1_3, self.up(D2_3)], D2_3)
        output = self.final(D1_4)
        return output