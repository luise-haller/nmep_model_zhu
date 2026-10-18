#!/bin/bash
# 2.2.5: augmentation schemes (AlexNet, LR 3e-4, batch size 256)
for scheme in crop_erase; do
  CUDA_VISIBLE_DEVICES=1 python main.py --cfg=configs/alexnet_base.yaml \
    --output output/q225_$scheme --opts TRAIN.LR 3e-4 AUG.SCHEME $scheme SAVE_CKPT False
done
