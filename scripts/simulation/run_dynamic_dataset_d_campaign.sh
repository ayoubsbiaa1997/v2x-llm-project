#!/usr/bin/env bash

set -u
set -o pipefail


NS3_DIR="$HOME/ns-3-dev"
PROJECT_DIR="$HOME/v2x-llm-project"

OUT_DIR="$PROJECT_DIR/datasets/raw/dynamic_dataset_d"
MANIFEST="$OUT_DIR/campaign_manifest.csv"

mkdir -p "$OUT_DIR"


UE_COUNTS=(8 12 16)
DISTANCES=(100 200 300)
SEEDS=(101 202 303 404)


echo \
"run_id,ue_count,distance_m,seed,run_number,stable_lambda,medium_lambda,high_lambda,recovery_lambda,packet_size_be,phase1_end,phase2_end,phase3_end,sim_time,status,exit_code" \
> "$MANIFEST"


cd "$NS3_DIR" || exit 1


TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0


for UE_COUNT in "${UE_COUNTS[@]}"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for SEED_INDEX in "${!SEEDS[@]}"; do

            SEED="${SEEDS[$SEED_INDEX]}"
            RUN_NUMBER=$((SEED_INDEX + 1))

            RUN_ID="dyn_u${UE_COUNT}_d${DISTANCE}_s${SEED}"

            # Variation contrôlée des changements de phase.
            PHASE1_END=$((3 + SEED_INDEX % 2))
            PHASE2_END=$((7 + SEED_INDEX % 3))
            PHASE3_END=$((11 + SEED_INDEX % 2))
            SIM_TIME=15

            STABLE_LAMBDA=50

            if [[ "$UE_COUNT" -eq 8 ]]; then
                MEDIUM_LAMBDA=400
                HIGH_LAMBDA=1400
            elif [[ "$UE_COUNT" -eq 12 ]]; then
                MEDIUM_LAMBDA=350
                HIGH_LAMBDA=1200
            else
                MEDIUM_LAMBDA=300
                HIGH_LAMBDA=1000
            fi

            # Légère variation entre seeds.
            MEDIUM_LAMBDA=$((MEDIUM_LAMBDA + 50 * SEED_INDEX))

            HIGH_LAMBDA=$((HIGH_LAMBDA + 150 * SEED_INDEX))

            RECOVERY_LAMBDA=$((100 + 25 * SEED_INDEX))

            PACKET_SIZE_BE=1200

            STDOUT_FILE="$OUT_DIR/${RUN_ID}_stdout.txt"
            CSV_FILE="$OUT_DIR/${RUN_ID}_qos_temporal.csv"
            FLOW_FILE="$OUT_DIR/${RUN_ID}"

            rm -f \
                "$STDOUT_FILE" \
                "$CSV_FILE" \
                "$FLOW_FILE"

            TOTAL_RUNS=$((TOTAL_RUNS + 1))

            echo
            echo "============================================================"
            echo "Run              : $RUN_ID"
            echo "UE               : $UE_COUNT"
            echo "Distance         : $DISTANCE"
            echo "Medium lambda    : $MEDIUM_LAMBDA"
            echo "High lambda      : $HIGH_LAMBDA"
            echo "Transitions      : $PHASE1_END / $PHASE2_END / $PHASE3_END"
            echo "============================================================"

            timeout 30m ./ns3 run "scratch/v2x-5g-dynamic \
--dynamicTraffic=1 \
--simTime=${SIM_TIME}s \
--ueNumPergNb=$UE_COUNT \
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
--dynamicSeed=$SEED \
--dynamicRun=$RUN_NUMBER \
--simTag=$RUN_ID \
--outputDir=$OUT_DIR" \
                > "$STDOUT_FILE" 2>&1

            EXIT_CODE=$?

            if [[ $EXIT_CODE -eq 0 && -s "$CSV_FILE" ]]; then
                STATUS="SUCCESS"
                SUCCESS_RUNS=$((SUCCESS_RUNS + 1))

                echo "Statut : SUCCESS"
            elif [[ $EXIT_CODE -eq 124 ]]; then
                STATUS="TIMEOUT"
                FAILED_RUNS=$((FAILED_RUNS + 1))

                echo "Statut : TIMEOUT"
            else
                STATUS="FAILED"
                FAILED_RUNS=$((FAILED_RUNS + 1))

                echo "Statut : FAILED ($EXIT_CODE)"
                tail -n 30 "$STDOUT_FILE"
            fi

            echo \
"$RUN_ID,$UE_COUNT,$DISTANCE,$SEED,$RUN_NUMBER,$STABLE_LAMBDA,$MEDIUM_LAMBDA,$HIGH_LAMBDA,$RECOVERY_LAMBDA,$PACKET_SIZE_BE,$PHASE1_END,$PHASE2_END,$PHASE3_END,$SIM_TIME,$STATUS,$EXIT_CODE" \
                >> "$MANIFEST"

        done
    done
done


echo
echo "============================================================"
echo "CAMPAGNE TERMINÉE"
echo "Runs totaux : $TOTAL_RUNS"
echo "Succès       : $SUCCESS_RUNS"
echo "Échecs       : $FAILED_RUNS"
echo "Manifest     : $MANIFEST"
echo "============================================================"


if [[ $SUCCESS_RUNS -eq 0 ]]; then
    exit 1
fi
