import argparse

import torch
from munch import Munch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def build_model(args):
    from modules.flow_matching import CFM
    from modules.length_regulator import InterpolateRegulator

    lr = args.length_regulator
    length_regulator = InterpolateRegulator(
        channels=lr.channels,
        sampling_ratios=lr.sampling_ratios,
        in_channels=lr.in_channels,
        n_f0_bins=lr.n_f0_bins,
        residual_alpha=lr.residual_alpha,
        style_dim=args.style_encoder.dim,
    )
    cfm = CFM(args)
    nets = Munch(
        cfm=cfm,
        length_regulator=length_regulator,
    )
    return nets

def load_checkpoint(model, path):
    params = torch.load(path, map_location="cpu")["net"]
    for name, module in model.items():
        state_dict = {
            key.removeprefix("module."): value
            for key, value in params[name].items()
        }
        module.load_state_dict(state_dict, strict=True)
        print(f"{name} loaded")
    return model


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
