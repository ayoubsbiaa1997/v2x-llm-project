#!/bin/bash

NS3_DIR=~/ns-3-dev
OUT_DIR=~/v2x-llm-project/datasets/raw/qos_logs_rich

mkdir -p "$OUT_DIR"
cd "$NS3_DIR"

vehicles_list=(5 10 20 30 40 50 60)
lambda_list=(50 100 300 600 1000)
packet_list=(64 128 256 512)
distance_list=(100 200 300)

for v in "${vehicles_list[@]}"; do
  for l in "${lambda_list[@]}"; do
    for p in "${packet_list[@]}"; do
      for d in "${distance_list[@]}"; do

        echo "Running scenario vehicles=$v lambda=$l packet=$p distance=$d"

        ./ns3 run "scratch/v2x-5g-sumo --ueNumPergNb=$v --lambdaUll=$l --lambdaBe=$l --packetSizeUll=$p --packetSizeBe=$p --interSiteDistance=$d" \
          > "$OUT_DIR/qos_v${v}_l${l}_p${p}_d${d}.txt"

      done
    done
  done
done
