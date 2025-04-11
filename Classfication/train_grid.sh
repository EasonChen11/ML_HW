#!/bin/bash

log_file="test_log.jsonl"

epochs=(20 40 80 160)
batch_sizes=(16 32)
learning_rates=(0.1 0.01 0.001)
losses=(ce mm)
# epochs=(2)
# batch_sizes=(16)
# learning_rates=(0.1)
# losses=(ce)

for epoch in "${epochs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for loss in "${losses[@]}"; do
        suffix="e${epoch}_bs${bs}_lr${lr}_loss${loss}"

        echo "{\"config\": \"$suffix\", \"stage\": \"start\"}" >> "$log_file"

        echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr, loss=$loss" | tee -a "$log_file"
        python3 train.py --epochs $epoch --bs $bs --lr $lr --loss $loss >> "$log_file" 2>&1

        echo -e "\nRunning test with epochs=$epoch, batch_size=$bs, lr=$lr, loss=$loss" | tee -a "$log_file"
        python3 test.py --weight weight_${suffix}.pth --loss $loss >> "$log_file" 2>&1

        echo "{\"config\": \"$suffix\", \"stage\": \"end\"}" >> "$log_file"
      done
    done
  done
done
