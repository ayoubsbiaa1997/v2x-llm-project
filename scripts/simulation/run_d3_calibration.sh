#!/usr/bin/env bash

set -u
set -o pipefail

NS3_DIR="$HOME/ns-3-dev"
PROJECT_DIR="$HOME/v2x-llm-project"
OUT_DIR="$PROJECT_DIR/datasets/raw/d3_calibration"

mkdir -p "$OUT_DIR"

SIM_TIME=20
PACKET_SIZE_BE=1200
SEED=2026
RUN_NUMBER=1

# Phases fixes
PHASE1_END=5
PHASE2_END=10
PHASE3_END=15

# Profil dynamique volontairement exigeant
STABLE_LAMBDA=200
MEDIUM_LAMBDA=1200
HIGH_LAMBDA=4000
RECOVERY_LAMBDA=400

# Format :
# nom ue_par_gnb distance bw1 bw2 puissance
SCENARIOS=(
    "baseline          8 100 50000000 50000000 35"
    "bw20              8 100 20000000 20000000 35"
    "bw10              8 100 10000000 10000000 35"
    "bw5               8 100  5000000  5000000 35"

    "ue16_bw50        16 100 50000000 50000000 35"
    "ue16_bw20        16 100 20000000 20000000 35"
    "ue16_bw10        16 100 10000000 10000000 35"

    "ue24_bw50        24 100 50000000 50000000 35"
    "ue24_bw20        24 100 20000000 20000000 35"
    "ue24_bw10        24 100 10000000 10000000 35"

    "dist300_bw50      8 300 50000000 50000000 35"
    "dist300_bw20      8 300 20000000 20000000 35"
    "dist300_bw10      8 300 10000000 10000000 35"

    "power30_bw20      8 300 20000000 20000000 30"
    "power25_bw20      8 300 20000000 20000000 25"
)

SUCCESS=0
FAILED=0
TOTAL=0

cd "$NS3_DIR" || exit 1

for scenario in "${SCENARIOS[@]}"; do

    read -r NAME UE DISTANCE BW1 BW2 TX_POWER <<< "$scenario"

    TOTAL=$((TOTAL + 1))

    RUN_ID="d3cal_${NAME}_s${SEED}"

    CSV_FILE="$OUT_DIR/${RUN_ID}_qos_temporal.csv"
    STDOUT_FILE="$OUT_DIR/${RUN_ID}_stdout.txt"

    echo
    echo "============================================================"
    echo "Scénario   : $NAME"
    echo "UE/gNB     : $UE"
    echo "Distance   : $DISTANCE m"
    echo "BW1 / BW2  : $BW1 / $BW2 Hz"
    echo "Puissance  : $TX_POWER dBm"
    echo "============================================================"

    rm -f "$CSV_FILE"

    timeout 30m ./ns3 run "scratch/v2x-5g-dynamic \
--dynamicTraffic=1 \
--simTime=${SIM_TIME}s \
--ueNumPergNb=$UE \
--lambdaUll=100 \
--stableLambdaBe=$STABLE_LAMBDA \
--mediumLambdaBe=$MEDIUM_LAMBDA \
--highLambdaBe=$HIGH_LAMBDA \
--recoveryLambdaBe=$RECOVERY_LAMBDA \
--phase1End=$PHASE1_END \
--phase2End=$PHASE2_END \
--phase3End=$PHASE3_END \
--qosLogInterval=0.5 \
--packetSizeUll=128 \
--packetSizeBe=$PACKET_SIZE_BE \
--interSiteDistance=$DISTANCE \
--bandwidthBand1=$BW1 \
--bandwidthBand2=$BW2 \
--totalTxPower=$TX_POWER \
--dynamicSeed=$SEED \
--dynamicRun=$RUN_NUMBER \
--simTag=$RUN_ID \
--outputDir=$OUT_DIR" \
        > "$STDOUT_FILE" 2>&1

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 && -s "$CSV_FILE" ]]; then
        echo "Statut : SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    elif [[ $EXIT_CODE -eq 124 ]]; then
        echo "Statut : TIMEOUT"
        FAILED=$((FAILED + 1))
    else
        echo "Statut : FAILED ($EXIT_CODE)"
        echo "Voir : $STDOUT_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "============================================================"
echo "CALIBRATION D3 TERMINÉE"
echo "Total   : $TOTAL"
echo "Succès  : $SUCCESS"
echo "Échecs  : $FAILED"
echo "Sortie  : $OUT_DIR"
echo "============================================================"
