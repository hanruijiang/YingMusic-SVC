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
        n_f0_bins: int,
        residual_alpha: float,
        style_dim: int = 192,
    ):
        super().__init__()
        layers = []
        for _ in sampling_ratios:
            layers.extend([
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.GroupNorm(1, channels),
                nn.Mish(),
            ])
        layers.append(nn.Conv1d(channels, channels, 1, 1))
        self.model = nn.Sequential(*layers)
        self.n_f0_bins = n_f0_bins
        self.f0_embedding = nn.Embedding(n_f0_bins, channels)
        self.register_buffer("f0_mask", torch.zeros(1, channels))
        self.content_in_proj = nn.Linear(in_channels, channels)
        self.register_buffer("residual_alpha", torch.tensor(residual_alpha, dtype=torch.float32))
        self.f0_to_style_proj = nn.Linear(channels, style_dim)
        self.f02style_mlp = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.Mish(),
            nn.Linear(style_dim, style_dim),
        )

    @staticmethod
    def _rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-6) -> torch.Tensor:
        return torch.sqrt((x.pow(2).mean(dim=dim, keepdim=keepdim)).clamp_min(eps))

    def forward(self, x, ylens, f0, style):
        x = self.content_in_proj(x)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')

        quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
        quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
        f0_emb = self.f0_embedding(quantized_f0)
        f0_emb = F.interpolate(f0_emb.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        x = x + f0_emb

        T_out = x.size(2)
        mask_bt = mask.float()
        voiced = F.interpolate((f0 > 0).float().unsqueeze(1), size=T_out, mode='nearest')
        voiced = voiced.transpose(1, 2)
        f0s = self.f0_to_style_proj(f0_emb.transpose(1, 2).contiguous())
        style_seq = style[:, None, :].repeat(1, T_out, 1)
        res = torch.tanh(self.f02style_mlp(torch.cat([style_seq, f0s], dim=-1)))

        # Limit the residual RMS relative to the global speaker embedding.
        s_ref = self._rms(style, dim=-1, keepdim=True)
        res_rms_t = self._rms(res * mask_bt, dim=-1)
        valid_counts = mask_bt.sum(1, keepdim=True).clamp_min(1e-6)
        res_rms = res_rms_t.sum(1, keepdim=True) / valid_counts
        target = self.residual_alpha * s_ref[:, None, :]
        scale = (target / res_rms.clamp_min(1e-6)).clamp(max=1.0)
        style_residual = res * scale * voiced * mask_bt

        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, style_residual
