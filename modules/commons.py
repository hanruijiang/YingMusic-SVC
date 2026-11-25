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

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts



def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def build_model(args, stage="DiT"):
    if stage == "DiT":
        from modules.flow_matching import CFM
        from modules.length_regulator import InterpolateRegulator

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None,
            codebook_size=args.length_regulator.content_codebook_size,
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
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return nets

def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key in params:
            # Strip distributed-training prefixes in released checkpoints.
            for k in list(params[key].keys()):
                if k.startswith("module."):
                    params[key][k[len("module.") :]] = params[key][k]
                    del params[key][k]
            model_state_dict = model[key].state_dict()
            # 过滤出形状匹配的键值对
            filtered_state_dict = {
                k: v
                for k, v in params[key].items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(
                    f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
                )
            print("%s loaded" % key)
            model[key].load_state_dict(filtered_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    return model

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
