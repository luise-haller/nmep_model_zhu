#!/bin/bash
# 2.2.4: throughput vs batch size (powers of 2 until CUDA OOM), 5 epochs each
for exp in $(seq 0 16); do
  bs=$((2**exp))
  echo "===== batch size $bs ====="
  CUDA_VISIBLE_DEVICES=0 python main.py --cfg=configs/alexnet_base.yaml \
    --output output/q224_bs$bs --batch-size $bs --opts TRAIN.LR 3e-4 TRAIN.EPOCHS 5 \
    || { echo "FAILED at batch size $bs (likely CUDA OOM)"; break; }
done
