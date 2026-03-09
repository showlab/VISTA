#!/bin/bash
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --config=configs/inference_example.yaml \
  --ckpt_path /path/to/checkpoints/20000.ckpt \
  --siglip-model /path/to/siglip-so400m-patch14-384 \
  --ipa-checkpoint /path/to/checkpoints/20000.ckpt
