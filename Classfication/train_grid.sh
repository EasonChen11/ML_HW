#!/bin/bash
# 相對路徑
base_dir=$(dirname "$0")
log_file="${base_dir}/3model_session_log.jsonl"

epochs=(2 40 80 160)
batch_sizes=(32 64)
learning_rates=(0.01 0.001)
losses=(ce mm)

for epoch in "${epochs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for loss in "${losses[@]}"; do
        suffix="e${epoch}_bs${bs}_lr${lr}_loss${loss}"

        echo "{\"config\": \"$suffix\", \"stage\": \"start\"}" >> "$log_file"

        echo "Running with epochs=$epoch, batch_size=$bs, lr=$lr, loss=$loss" | tee -a "$log_file"
        python3 train.py --epochs $epoch --bs $bs --lr $lr --loss $loss >> "$log_file" 2>&1

        echo -e "\nRunning test with epochs=$epoch, batch_size=$bs, lr=$lr, loss=$loss" | tee -a "$log_file"
        python3 test.py --weight weight_${suffix}.pth >> "$log_file" 2>&1

        # kaggle competitions submit -c plant-seedlings-classification -f predictions/predictions_${suffix}.csv -m "submit e${epoch}_bs${bs}_lr${lr}_loss${loss}"
        echo "{\"config\": \"$suffix\", \"stage\": \"end\"}" >> "$log_file"
      done
    done
  done
done
