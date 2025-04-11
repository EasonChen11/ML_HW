#!/bin/bash

log_file="test_log.jsonl"

epochs=(20 40 80)
batch_sizes=(8 16 32)
learning_rates=(0.1 0.01 0.001)
losses=(mm ce mm)

for epoch in "${epochs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for loss in "${losses[@]}"; do
        suffix="e${epoch}_bs${bs}_lr${lr}_loss${loss}"
        echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr, loss=$loss"
        # 訓練與測試
        python3 train.py --epochs $epoch --bs $bs --lr $lr --loss $loss
        python3 test.py --weight weight_${suffix}.pth --log $log_file --loss $loss
      done
    done
  done
done