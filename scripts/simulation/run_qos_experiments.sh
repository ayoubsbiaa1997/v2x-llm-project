#!/bin/bash

NS3_DIR=~/ns-3-dev
OUT_DIR=~/v2x-llm-project/datasets/raw/qos_logs

mkdir -p $OUT_DIR
cd $NS3_DIR

vehicles_list=(5 10 20 30 40 50 60)

for v in "${vehicles_list[@]}"; do
  echo "Running scenario vehicles=$v"

  ./ns3 run "scratch/v2x-5g-sumo --ueNumPergNb=$v" \
    > "$OUT_DIR/qos_v${v}.txt"
done
