#!/usr/bin/env bash

set -u
set -o pipefail

NS3_DIR="$HOME/ns-3-dev"
PROJECT_DIR="$HOME/v2x-llm-project"
OUT_DIR="$PROJECT_DIR/datasets/raw/dynamic_qos_calibration"

mkdir -p "$OUT_DIR"

# Format :
# tag | nombre UE | lambda MEDIUM | lambda HIGH | distance
CONFIGURATIONS=(
  "cal_u8_m300_h900_d200|8|300|900|200"
  "cal_u8_m500_h1500_d200|8|500|1500|200"
  "cal_u12_m300_h900_d200|12|300|900|200"
  "cal_u12_m500_h1500_d200|12|500|1500|200"
  "cal_u16_m300_h900_d200|16|300|900|200"
  "cal_u16_m500_h1500_d200|16|500|1500|200"
)

cd "$NS3_DIR" || exit 1

RUN_NUMBER=1
SUCCESS_COUNT=0
FAILURE_COUNT=0

for CONFIG in "${CONFIGURATIONS[@]}"; do
    IFS="|" read -r TAG UE_COUNT MEDIUM_RATE HIGH_RATE DISTANCE \
        <<< "$CONFIG"

    STDOUT_FILE="$OUT_DIR/${TAG}_stdout.txt"
    CSV_FILE="$OUT_DIR/${TAG}_qos_temporal.csv"
    SUMMARY_FILE="$OUT_DIR/${TAG}"

    rm -f \
        "$STDOUT_FILE" \
        "$CSV_FILE" \
        "$SUMMARY_FILE"

    echo
    echo "============================================================"
    echo "Run          : $TAG"
    echo "UE           : $UE_COUNT"
    echo "Medium rate  : $MEDIUM_RATE paquets/s"
    echo "High rate    : $HIGH_RATE paquets/s"
    echo "Packet BE    : 1200 octets"
    echo "Distance     : $DISTANCE m"
    echo "============================================================"

    timeout 20m ./ns3 run "scratch/v2x-5g-dynamic \
--dynamicTraffic=1 \
--simTime=12s \
--ueNumPergNb=$UE_COUNT \
--lambdaUll=100 \
--stableLambdaBe=50 \
--mediumLambdaBe=$MEDIUM_RATE \
--highLambdaBe=$HIGH_RATE \
--recoveryLambdaBe=100 \
--phase1End=3 \
--phase2End=6 \
--phase3End=9 \
--qosLogInterval=0.5 \
--packetSizeUll=128 \
--packetSizeBe=1200 \
--interSiteDistance=$DISTANCE \
--dynamicSeed=12345 \
--dynamicRun=$RUN_NUMBER \
--simTag=$TAG \
--outputDir=$OUT_DIR" \
        > "$STDOUT_FILE" 2>&1

    STATUS=$?

    if [[ $STATUS -eq 0 && -s "$CSV_FILE" ]]; then
        echo "Statut : SUCCÈS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif [[ $STATUS -eq 124 ]]; then
        echo "Statut : TIMEOUT après 20 minutes"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    else
        echo "Statut : ÉCHEC, code=$STATUS"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))

        echo "Dernières lignes du journal :"
        tail -n 30 "$STDOUT_FILE"
    fi

    RUN_NUMBER=$((RUN_NUMBER + 1))
done

echo
echo "============================================================"
echo "FIN DE LA CALIBRATION"
echo "Succès : $SUCCESS_COUNT"
echo "Échecs : $FAILURE_COUNT"
echo "Sortie : $OUT_DIR"
echo "============================================================"

if [[ $SUCCESS_COUNT -eq 0 ]]; then
    exit 1
fi
