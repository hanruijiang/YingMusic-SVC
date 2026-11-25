# YingMusic-SVC — CLI inference

Zero-shot singing voice conversion: convert a source vocal to the timbre of a reference singer while preserving the lyrics and melody. This repository contains only the SVC command-line inference implementation.

Original project: [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) · [Paper](https://arxiv.org/abs/2512.04793) · [Model weights](https://huggingface.co/GiantAILab/YingMusic-SVC).

## Installation

Use Python 3.10 and an NVIDIA GPU with a compatible CUDA-enabled PyTorch installation.

```bash
git clone https://github.com/hanruijiang/YingMusic-SVC.git
cd YingMusic-SVC

conda create -n ymsvc python=3.10
conda activate ymsvc
# Example: PyTorch 2.4.0 with CUDA 12.1; choose a wheel compatible with your driver.
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For compressed audio formats, install FFmpeg if your audio backend requires it. WAV inputs are recommended.

## Checkpoint

Download [YingMusic-SVC-full.pt](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt) into `checkpoints/`:

```bash
mkdir -p checkpoints
huggingface-cli download GiantAILab/YingMusic-SVC YingMusic-SVC-full.pt --local-dir checkpoints
```

RMVPE, CAMPPlus, Whisper and BigVGAN weights are downloaded automatically from Hugging Face on first use. Use trusted checkpoints only. All required SVC state-dict entries must load successfully.

## Inference

Run from the repository root. Provide an isolated source vocal and a reference vocal; the reference is truncated to 25 seconds internally. Inputs must be finite and at least 0.1 seconds long. The configuration supports the official 44.1 kHz, 128-mel, 512-hop SVC model only; other model architectures, CPU inference and accompaniment separation are not supported.

```bash
python my_inference.py \
    --source path/to/source_vocal.wav \
    --target path/to/reference_vocal.wav \
    --checkpoint checkpoints/YingMusic-SVC-full.pt \
    --config configs/YingMusic-SVC.yml \
    --diffusion-steps 100 \
    --cuda 0 \
    --fp16 True \
    --expname example
```

The converted 44.1 kHz vocal WAV is saved under `outputs/example/`. `--fp16 False` disables DiT autocast; Whisper still uses FP16. Use `--help` to list arguments, or edit the paths in `my_infer.sh` and run `bash my_infer.sh`.

## Acknowledgements and license

Based on [YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) and [Seed-VC](https://github.com/Plachtaa/seed-vc). See [LICENSE.txt](LICENSE.txt) for the MIT license. If using the model in research, please cite the [original paper](https://arxiv.org/abs/2512.04793).
