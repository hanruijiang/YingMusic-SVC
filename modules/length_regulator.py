from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.commons import sequence_mask
import numpy as np

f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def f0_to_coarse(f0, f0_bin):
  f0_mel = 1127 * (1 + f0 / 700).log()
  a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
  b = f0_mel_min * a - 1.
  f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
  f0_coarse = torch.round(f0_mel).long()
  f0_coarse = f0_coarse * (f0_coarse > 0)
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  f0_coarse = f0_coarse * (f0_coarse < f0_bin)
  f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
  return f0_coarse

class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            in_channels: int,
            out_channels: int = None,
            groups: int = 1,
            f0_condition: bool = False,
            n_f0_bins: int = 512,
            # f0 -> style 残差（方案2）相关开关与参数（仅在本模块内部实现，不改上游）
            use_style_residual: bool = False,
            style_dim: int = 192,
            residual_alpha: float = 0.2,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        else:
            self.interpolate = False
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)

        # 统一记录 f0 bins（供残差分支使用）
        self.n_f0_bins = n_f0_bins

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.register_buffer("f0_mask", torch.zeros(1, channels))
        else:
            self.f0_condition = False

        self.content_in_proj = nn.Linear(in_channels, channels)

        # ---------------- f0 -> style 残差（方案2）----------------
        self.use_style_residual = use_style_residual
        self.style_dim = style_dim
        # 作为标量上限因子（不训练），便于稳态控制
        self.register_buffer("residual_alpha", torch.tensor(residual_alpha, dtype=torch.float32))
        if self.use_style_residual:
            # 直接复用已计算的 f0_emb（channels 维），先投影到 style_dim，再过小 MLP
            self.f0_to_style_proj = nn.Linear(channels, style_dim)
            self.f02style_mlp = nn.Sequential(
                nn.Linear(style_dim*2, style_dim),
                nn.Mish(),
                nn.Linear(style_dim, style_dim),
            )

    @staticmethod
    def _rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-6) -> torch.Tensor:
        return torch.sqrt((x.pow(2).mean(dim=dim, keepdim=keepdim)).clamp_min(eps))

    def forward(self, x, ylens, f0=None, style=None):
        x = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')    # 插值到mel长度
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, :x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:
                quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
                quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(quantized_f0)
                f0_emb = F.interpolate(f0_emb.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')

                x = x + f0_emb

        # ============== f0 -> style 残差（方案2）：仅依赖 f0 ==============
        style_residual = None
        if self.use_style_residual and style is not None and ylens is not None:
            B = style.size(0)
            # x 当前为 (B, C, T_out) 形式
            T_out = x.size(2)
            # 时域有效帧 mask（用于缩放与抑制无效帧）
            mask_bt = sequence_mask(ylens).unsqueeze(-1).float()  # (B, T_out, 1)
            if mask_bt.size(1) != T_out:
                # 对齐到当前 T_out（非插值路径时会发生）
                mask_bt = F.interpolate(mask_bt.transpose(1, 2), size=T_out, mode='nearest').transpose(1, 2)
            # voiced 掩码（静音=0），先在 f0 时间轴，再插值到 T_out
            if f0 is not None:
                voiced = (f0 > 0).float().unsqueeze(1)  # (B, 1, T_f0)
                voiced = F.interpolate(voiced, size=T_out, mode='nearest').squeeze(1)  # (B, T_out)
            else:
                voiced = torch.zeros(B, T_out, device=x.device, dtype=x.dtype)
            voiced = voiced.unsqueeze(-1)  # (B, T_out, 1)

            # 直接使用上面已生成并对齐到 T_out 的 f0_emb（若不可用则残差置零）
            # f0_emb 当前是 (B, channels, T_out) 形状；投影到 style_dim
            f0s = None
            if 'f0_emb' in locals():
                f0e = f0_emb.transpose(1, 2).contiguous()  # (B, T_out, channels)
                f0s = self.f0_to_style_proj(f0e)           # (B, T_out, style_dim)
            else:
                f0s = torch.zeros(B, T_out, self.style_dim, device=x.device, dtype=x.dtype)

            # 依赖 f0和spk 的小 MLP 变换，先做幅度限幅
            style_seq = style[:, None, :].repeat(1, T_out, 1)  # (B, T_out, Ds)
            res_in = torch.cat([style_seq, f0s], dim=-1)  # (B, T_out, 2*Ds)
            res = self.f02style_mlp(res_in)  # (B, T_out, Ds)
            res = torch.tanh(res)

            # 以全局 style 的 RMS 为标尺，缩放到 alpha * RMS(style)
            s_ref = self._rms(style, dim=-1, keepdim=True)  # (B, 1)
            # 计算残差在时域+特征域的 RMS（仅有效帧）
            res_rms_t = self._rms(res * mask_bt, dim=-1)  # (B, T_out, 1)
            valid_counts = mask_bt.sum(1, keepdim=True).clamp_min(1e-6)  # (B, 1, 1)
            res_rms = (res_rms_t.sum(1, keepdim=True) / valid_counts)  # (B, 1, 1)

            target = self.residual_alpha * s_ref[:, None, :]  # (B, 1, 1)
            scale = (target / res_rms.clamp_min(1e-6)).clamp(max=1.0)  # 不放大，只削弱
            res = res * scale

            # 静音与越界掩码
            res = res * voiced * mask_bt
            style_residual = res  # (B, T_out, Ds)

        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, style_residual
