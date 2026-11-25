# YingMusic-SVC

Zero-shot singing voice conversion: convert a source vocal to the timbre of a reference singer while preserving its lyrics and melody.

[Paper](https://arxiv.org/abs/2512.04793) · [Model weights](https://huggingface.co/GiantAILab/YingMusic-SVC)

## Installation

Python 3.10, an NVIDIA GPU and CUDA-enabled PyTorch are required.

```bash
git clone -b cleanup/cli-inference-only https://github.com/hanruijiang/YingMusic-SVC.git
cd YingMusic-SVC

conda create -n ymsvc python=3.10
conda activate ymsvc
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The example uses CUDA 12.1. Use a compatible PyTorch wheel for your GPU and driver. WAV inputs are recommended; compressed audio may require FFmpeg.

## Models

Download the [YingMusic-SVC checkpoint](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt):

```bash
huggingface-cli download GiantAILab/YingMusic-SVC YingMusic-SVC-full.pt --local-dir checkpoints
```

RMVPE, CAMPPlus, Whisper-small and BigVGAN weights are downloaded automatically on first use. The default Hugging Face cache is `checkpoints/hf_cache/`; set `HF_HUB_CACHE` to use another directory.

## Inference

Run from the repository root with an isolated source vocal and a reference vocal. The reference is limited to 25 seconds. The model uses 44.1 kHz audio, 128 mel channels and a 512-sample hop.

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

Converted audio is saved as a WAV file in `outputs/example/`. `--fp16 False` disables DiT autocast; Whisper uses FP16. Use `--help` for arguments, or edit and run `my_infer.sh`.

## Acknowledgements and licenses

Based on [YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) and [Seed-VC](https://github.com/Plachtaa/seed-vc). See [LICENSE.txt](LICENSE.txt) and [third-party notices](incl_licenses/README.md) for source licenses. Model weights are subject to their respective model licenses.
