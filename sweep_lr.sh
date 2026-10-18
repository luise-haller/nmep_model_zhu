#!/bin/bash
# 2.2.2: AlexNet learning-rate sweep (everything else fixed)
for lr in 3e-5 1e-4 3e-4 1e-3 3e-3; do
  CUDA_VISIBLE_DEVICES=0 python main.py --cfg=configs/alexnet_base.yaml \
    --output output/q222_lr$lr --opts TRAIN.LR $lr
done
