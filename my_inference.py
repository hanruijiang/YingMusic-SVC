import argparse
import os
import time

os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import librosa
import numpy as np
import torch
import torchaudio
import yaml

from hf_utils import load_custom_model_from_hf
from mm4 import preprocess_voice_conversion
from modules.commons import build_model, load_checkpoint, recursive_munch, str2bool


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

def validate_config(config):
    pre = config["preprocess_params"]
    spect = pre["spect_params"]
    params = config["model_params"]
    expected = (
        (pre, {"sr": 44100}),
        (spect, {"hop_length": 512, "n_mels": 128, "n_fft": 2048, "win_length": 2048, "fmin": 0}),
        (params["vocoder"], {"type": "bigvgan"}),
        (params["speech_tokenizer"], {"type": "whisper"}),
        (params["style_encoder"], {"dim": 192}),
        (params["length_regulator"], {
            "channels": 768, "in_channels": 768, "sampling_ratios": [1, 1, 1, 1],
            "f0_condition": True, "n_f0_bins": 256, "use_style_residual": True,
        }),
        (params["DiT"], {
            "hidden_dim": 768, "num_heads": 12, "depth": 17, "in_channels": 128,
            "content_dim": 768, "style_condition": True, "uvit_skip_connection": True,
        }),
    )
    for section, values in expected:
        for name, value in values.items():
            if section.get(name) != value:
                raise ValueError(f"Official SVC configuration requires {name}={value}")
    if spect.get("fmax") not in (None, 22050):
        raise ValueError("Official SVC configuration requires fmax=null or 22050")

    for key, name in (
        ("vocoder", "nvidia/bigvgan_v2_44khz_128band_512x"),
        ("speech_tokenizer", "openai/whisper-small"),
    ):
        model_name = params[key]["name"]
        if model_name != name and not os.path.isdir(model_name):
            raise ValueError(f"{key} requires {name} or its local model directory")


def load_models_api(args, device=torch.device("cuda")):
    dit_checkpoint_path = args.checkpoint
    print(f'load model from {dit_checkpoint_path}')
    dit_config_path = args.config
    print(f'load config from {dit_config_path}')
    with open(dit_config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    validate_config(config)
    model_params = recursive_munch(config["model_params"])
    spect = config["preprocess_params"]["spect_params"]
    fmin = spect["fmin"]
    fmax = spect["fmax"]

    from modules.rmvpe import RMVPE

    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt")

    f0_extractor = RMVPE(model_path, device=device)
    f0_fn = f0_extractor.infer_from_audio

    model = build_model(model_params)
    sr = config["preprocess_params"]["sr"]

    model = load_checkpoint(model, dit_checkpoint_path)

    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches()

    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin"
    )

    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    from modules.bigvgan.bigvgan import BigVGAN

    bigvgan_model = BigVGAN.from_pretrained(model_params.vocoder.name)
    h = bigvgan_model.h
    vocoder_audio = (h.sampling_rate, h.num_mels, h.hop_size, h.n_fft, h.win_size, h.fmin, h.fmax or sr / 2)
    svc_audio = (sr, spect["n_mels"], spect["hop_length"], spect["n_fft"], spect["win_length"], fmin, fmax or sr / 2)
    if vocoder_audio != svc_audio:
        raise ValueError("BigVGAN and SVC audio configurations must match")
    bigvgan_model.remove_weight_norm()
    vocoder_fn = bigvgan_model.eval().to(device)

    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors="pt",
                                               return_attention_mask=True)
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": fmin,
        "fmax": fmax,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


@torch.no_grad()
def run_inference(args, model_bundle, device=torch.device("cuda")):

    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = model_bundle
    fp16 = args.fp16
    sr = mel_fn_args['sampling_rate']
    force_pitch_shift = args.semi_tone_shift

    source = args.source

    target_name = args.target
    print(source, target_name)
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    exp_path = os.path.join(args.output , args.expname)
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]


    hop_length = mel_fn_args['hop_size']
    for name, audio in (("source", source_audio), ("reference", ref_audio)):
        if audio.size < sr // 10 or not np.isfinite(audio).all():
            raise ValueError(f"{name} audio must be finite and at least 0.1 seconds long")
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    time_vc_start = time.time()

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)


    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    F0_ori = torch.from_numpy(F0_ori).to(device)[None]
    F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    shifted_f0_alt, pitch_shift = preprocess_voice_conversion(
        voiced_f0_ori=voiced_F0_ori,
        voiced_f0_alt=voiced_F0_alt,
        shifted_f0_alt=F0_alt,
        enable_adaptive=True,
        max_shift_semitones=24,
        force_pitch_shift=force_pitch_shift,
    )
    print(f'automatic pitch shift {pitch_shift} semi tones')

    cond, style_cond = model.length_regulator(
        S_alt, ylens=target_lengths, f0=shifted_f0_alt, style=style2
    )
    prompt_condition, style_prompt = model.length_regulator(
        S_ori, ylens=target2_lengths, f0=F0_ori, style=style2
    )

    max_source_window = max_context_window - mel2.size(2)

    processed_frames = 0
    generated_wave_chunks = []

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        chunk_style_cond = style_cond[:, processed_frames:processed_frames + max_source_window]
        cat_style_cond = torch.cat([style_prompt, chunk_style_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):

            vc_target = model.cfm.inference(cat_condition,
                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                        mel2, style2, diffusion_steps,
                                                       inference_cfg_rate=inference_cfg_rate, style_r=cat_style_cond)

            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")
    os.makedirs(exp_path, exist_ok=True)
    src_name = os.path.splitext(os.path.basename(source))[0]
    tgt_name = os.path.splitext(os.path.basename(target_name))[0]
    vc_name = f"{tgt_name}_{src_name}_{pitch_shift}.wav"
    output_path = os.path.join(exp_path, vc_name)
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Source vocal audio")
    parser.add_argument("--target", type=str, required=True, help="Reference singer audio")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--expname", type=str, default="inference")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--config", type=str, default="./configs/YingMusic-SVC.yml")
    args = parser.parse_args()

    if args.diffusion_steps <= 0:
        parser.error("--diffusion-steps must be greater than zero")
    for name in ("source", "target", "checkpoint", "config"):
        if not os.path.isfile(getattr(args, name)):
            parser.error(f"--{name}: file does not exist: {getattr(args, name)}")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and an NVIDIA GPU are required")
    if not 0 <= args.cuda < torch.cuda.device_count():
        parser.error(f"Invalid CUDA device index: {args.cuda}")
    args.cuda = torch.device(f"cuda:{args.cuda}")
    if args.fp16:
        print('DiT FP16 inference enabled')

    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.semi_tone_shift = None

    args.output = './outputs'
    os.makedirs(args.output, exist_ok=True)

    models = load_models_api(args, device=args.cuda)
    vc = run_inference(args, models, device=args.cuda)
    print(f"Saved converted vocal: {vc}")
