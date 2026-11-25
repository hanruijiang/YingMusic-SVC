#!/bin/bash
set -euo pipefail

source='path/to/source_vocal.wav'
target='path/to/target_timbre.wav'
checkpoint='checkpoints/YingMusic-SVC-full.pt'
diffusion_step=100
fp16=True
config='./configs/YingMusic-SVC.yml'
expname='your_exp_name'
cuda=0

python my_inference.py \
    --source "${source}" \
    --target "${target}" \
    --diffusion-steps "${diffusion_step}" \
    --checkpoint "${checkpoint}" \
    --expname "${expname}" \
    --cuda "${cuda}" \
    --fp16 "${fp16}" \
    --config "${config}"
