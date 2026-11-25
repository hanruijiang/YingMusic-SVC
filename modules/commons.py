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

    length_regulator = InterpolateRegulator(
        channels=args.length_regulator.channels,
        sampling_ratios=args.length_regulator.sampling_ratios,
        in_channels=args.length_regulator.in_channels,
        f0_condition=args.length_regulator.f0_condition if hasattr(args.length_regulator, "f0_condition") else False,
        n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512,
        use_style_residual = args.length_regulator.use_style_residual if hasattr(args.length_regulator, "use_style_residual") else False,
        residual_alpha = args.length_regulator.residual_alpha if hasattr(args.length_regulator, "residual_alpha") else 0,
    )
    cfm = CFM(args)
    nets = Munch(
        cfm=cfm,
        length_regulator=length_regulator,
    )
    return nets

def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("net"), dict):
        raise RuntimeError("Checkpoint must contain a 'net' mapping")
    params = state["net"]
    for name, module in model.items():
        if name not in params or not isinstance(params[name], dict):
            raise RuntimeError(f"Missing checkpoint module: {name}")
        state_dict = {}
        for key, value in params[name].items():
            key = key.removeprefix("module.")
            if key in state_dict:
                raise RuntimeError(f"Duplicate checkpoint key: {name}.{key}")
            state_dict[key] = value
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        print(f"{name} loaded")
    return model

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
