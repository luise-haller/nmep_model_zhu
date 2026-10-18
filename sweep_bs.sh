#!/bin/bash
# 2.2.3: AlexNet batch-size sweep (LR fixed at 3e-4, everything else fixed)
for bs in 32 64 100 128 256 300 512; do
  CUDA_VISIBLE_DEVICES=6 python main.py --cfg=configs/alexnet_base.yaml \
    --output output/q223_bs$bs --batch-size $bs --opts TRAIN.LR 3e-4
done
