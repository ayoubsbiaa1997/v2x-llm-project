#!/bin/bash

NS3_DIR="$HOME/ns-3-dev"
OUT_DIR="$HOME/v2x-llm-project/datasets/processed/handover_campaign_v3"

mkdir -p "$OUT_DIR"
cd "$NS3_DIR" || exit 1

run_scenario () {
    NAME=$1
    UES=$2
    SPEED=$3
    DIST=$4
    LAMBDA=$5
    PACKET=$6

    echo "======================================"
    echo "Running $NAME"
    echo "UEs=$UES speed=$SPEED distance=$DIST lambda=$LAMBDA packet=$PACKET"
    echo "======================================"

    timeout 8m ./ns3 run "scratch/v2x-5g-handover-v2 --simTime=2s --ueNumPergNb=$UES --enableUeMobility=true --ueSpeed=$SPEED --bsDistance=$DIST --lambdaUll=$LAMBDA --lambdaBe=$LAMBDA --packetSizeUll=$PACKET --packetSizeBe=$PACKET --simTag=$NAME --outputDir=$OUT_DIR" \
    > "$OUT_DIR/log_${NAME}.txt" 2>&1

    echo "Finished $NAME"
}

for UES in 15 25 35; do
  for SPEED in 20 35 50; do
    for DIST in 250 320 400; do

      NAME="HO_V3_U${UES}_S${SPEED}_D${DIST}"

      if [ "$UES" -eq 15 ]; then
        LAMBDA=300
        PACKET=256
      elif [ "$UES" -eq 25 ]; then
        LAMBDA=500
        PACKET=512
      else
        LAMBDA=700
        PACKET=1024
      fi

      run_scenario "$NAME" "$UES" "$SPEED" "$DIST" "$LAMBDA" "$PACKET"

    done
  done
done

echo "Campagne V3 terminée : $OUT_DIR"
