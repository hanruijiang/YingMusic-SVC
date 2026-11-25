from abc import ABC

import torch
from tqdm import tqdm

from modules.diffusion_transformer import DiT


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.estimator = None

        self.in_channels = args.DiT.in_channels

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, style_r=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate, style_r=style_r)

    def solve_euler(self, x, x_lens, prompt, mu, style, t_span, inference_cfg_rate=0.5, style_r=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t = t_span[0]

        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)    # torch.Size([1, 128, 1640])
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]  # torch.Size([1, 128, 349])
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)       # prompt mel
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)                # timbre token
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)                         # content
                stacked_x = torch.cat([x, x], dim=0)                                              # noise
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)                    # time step
                if style_r is not None:
                    stacked_style_r = torch.cat([style_r, torch.zeros_like(style_r)], dim=0)
                else:
                    stacked_style_r = None

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,style_r=stacked_style_r
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu, style_r=style_r)


            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0

        return x


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        self.estimator = DiT(args)
