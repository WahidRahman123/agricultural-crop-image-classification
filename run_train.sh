#!/bin/bash
# Quick training launcher for MSc project
cd "$(dirname "$0")"
python src/train.py \
  --epochs 12 \
  --batch-size 8 \
  --img-size 160 \
  --max-per-class 80 \
  --model resnet18 \
  "$@"
