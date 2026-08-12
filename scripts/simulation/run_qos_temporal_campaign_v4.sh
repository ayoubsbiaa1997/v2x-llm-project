#!/bin/bash

NS3_DIR="$HOME/ns-3-dev"
OUT_DIR="$HOME/v2x-llm-project/datasets/processed/mobility/qos_campaign_v4"

mkdir -p "$OUT_DIR"

cd "$NS3_DIR" || exit 1

run_scenario () {
    NAME=$1
    UES=$2
    LAMBDA=$3
    PACKET=$4
    DIST=$5

    echo "======================================"
    echo "Running scenario: $NAME"
    echo "UEs=$UES Lambda=$LAMBDA Packet=$PACKET Distance=$DIST"
    echo "======================================"

    ./ns3 run "scratch/v2x-5g-sumo --simTime=60s --ueNumPergNb=$UES --lambdaUll=$LAMBDA --lambdaBe=$LAMBDA --packetSizeUll=$PACKET --packetSizeBe=$PACKET --interSiteDistance=$DIST"

    cp qos_temporal_windows.csv "$OUT_DIR/qos_${NAME}.csv"
}

# Scénarios stables
run_scenario "D50_STABLE" 5 50 64 50
run_scenario "D100_STABLE" 5 80 64 100
run_scenario "D150_STABLE" 10 100 128 150

# Scénarios intermédiaires
run_scenario "D200_MEDIUM" 20 300 256 200
run_scenario "D250_MEDIUM" 30 500 512 250
run_scenario "D300_MEDIUM" 40 700 512 300

# Scénarios congestionnés
run_scenario "D350_HIGH" 50 900 1024 350
run_scenario "D400_HIGH" 60 1200 1024 400
run_scenario "D450_HIGH" 70 1500 1024 450

echo "Campagne terminée. Fichiers générés dans : $OUT_DIR"
