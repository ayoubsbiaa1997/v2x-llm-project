#!/usr/bin/env bash

set -u
set -o pipefail

NS3_DIR="$HOME/ns-3-dev"
PROJECT_DIR="$HOME/v2x-llm-project"

OUT_DIR="$PROJECT_DIR/datasets/raw/dynamic_dataset_d2"
MANIFEST="$OUT_DIR/campaign_manifest.csv"

mkdir -p "$OUT_DIR"

echo \
"run_id,profile,ue_count,distance_m,seed,run_number,stable_lambda,medium_lambda,high_lambda,recovery_lambda,packet_size_be,phase1_end,phase2_end,phase3_end,sim_time,status,exit_code" \
> "$MANIFEST"

UE_COUNTS=(8 16)
DISTANCES=(100 300)
SEEDS=(111 222)

PROFILES=(
    "EARLY_DEGRADATION"
    "LATE_DEGRADATION"
    "ABRUPT_DEGRADATION"
    "MEDIUM_CONTROL"
    "STABLE_CONTROL"
    "PERSISTENT_HIGH"
)

cd "$NS3_DIR" || exit 1

TOTAL=0
SUCCESS=0
FAILED=0

for PROFILE in "${PROFILES[@]}"; do
    for UE_COUNT in "${UE_COUNTS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for SEED_INDEX in "${!SEEDS[@]}"; do

                SEED="${SEEDS[$SEED_INDEX]}"
                RUN_NUMBER=$((SEED_INDEX + 1))

                STABLE_LAMBDA=50
                PACKET_SIZE_BE=1200
                SIM_TIME=20

                case "$UE_COUNT" in
                    8)
                        BASE_MEDIUM=400
                        BASE_HIGH=1400
                        ;;
                    12)
                        BASE_MEDIUM=350
                        BASE_HIGH=1200
                        ;;
                    16)
                        BASE_MEDIUM=300
                        BASE_HIGH=1000
                        ;;
                    *)
                        echo "UE_COUNT non supporté : $UE_COUNT"
                        exit 1
                        ;;
                esac

                MEDIUM_LAMBDA=$((BASE_MEDIUM + 50 * SEED_INDEX))

                HIGH_LAMBDA=$((BASE_HIGH + 150 * SEED_INDEX))

                case "$PROFILE" in
                    EARLY_DEGRADATION)
                        PHASE1_END=$((3 + SEED_INDEX))
                        PHASE2_END=$((6 + SEED_INDEX))
                        PHASE3_END=$((11 + SEED_INDEX))
                        RECOVERY_LAMBDA=100
                        ;;

                    LATE_DEGRADATION)
                        PHASE1_END=$((6 + SEED_INDEX))
                        PHASE2_END=$((12 + SEED_INDEX))
                        PHASE3_END=$((16 + SEED_INDEX))
                        RECOVERY_LAMBDA=100
                        ;;

                    ABRUPT_DEGRADATION)
                        PHASE1_END=$((5 + SEED_INDEX))
                        PHASE2_END=$((6 + SEED_INDEX))
                        PHASE3_END=$((13 + SEED_INDEX))
                        MEDIUM_LAMBDA=$HIGH_LAMBDA
                        RECOVERY_LAMBDA=100
                        ;;

                    MEDIUM_CONTROL)
                        PHASE1_END=$((4 + SEED_INDEX))
                        PHASE2_END=$((9 + SEED_INDEX))
                        PHASE3_END=$((14 + SEED_INDEX))
                        HIGH_LAMBDA=$MEDIUM_LAMBDA
                        RECOVERY_LAMBDA=100
                        ;;

                    STABLE_CONTROL)
                        PHASE1_END=$((4 + SEED_INDEX))
                        PHASE2_END=$((9 + SEED_INDEX))
                        PHASE3_END=$((14 + SEED_INDEX))
                        MEDIUM_LAMBDA=$STABLE_LAMBDA
                        HIGH_LAMBDA=$STABLE_LAMBDA
                        RECOVERY_LAMBDA=$STABLE_LAMBDA
                        ;;

                    PERSISTENT_HIGH)
                        PHASE1_END=$((4 + SEED_INDEX))
                        PHASE2_END=$((8 + SEED_INDEX))
                        PHASE3_END=$((14 + SEED_INDEX))
                        RECOVERY_LAMBDA=$HIGH_LAMBDA
                        ;;

                    *)
                        echo "Profil inconnu : $PROFILE"
                        exit 1
                        ;;
                esac

                RUN_ID=$(
                    echo \
                    "d2_${PROFILE}_u${UE_COUNT}_d${DISTANCE}_s${SEED}" \
                    | tr '[:upper:]' '[:lower:]'
                )

                CSV_FILE="$OUT_DIR/${RUN_ID}_qos_temporal.csv"
                STDOUT_FILE="$OUT_DIR/${RUN_ID}_stdout.txt"
                FLOW_FILE="$OUT_DIR/${RUN_ID}"

                rm -f \
                    "$CSV_FILE" \
                    "$STDOUT_FILE" \
                    "$FLOW_FILE"

                TOTAL=$((TOTAL + 1))

                echo
                echo "============================================================"
                echo "Run       : $RUN_ID"
                echo "Profil    : $PROFILE"
                echo "UE        : $UE_COUNT"
                echo "Distance  : $DISTANCE"
                echo "Lambdas   : $STABLE_LAMBDA / $MEDIUM_LAMBDA / $HIGH_LAMBDA / $RECOVERY_LAMBDA"
                echo "Phases    : $PHASE1_END / $PHASE2_END / $PHASE3_END"
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
                    SUCCESS=$((SUCCESS + 1))
                    echo "Statut : SUCCESS"
                elif [[ $EXIT_CODE -eq 124 ]]; then
                    STATUS="TIMEOUT"
                    FAILED=$((FAILED + 1))
                    echo "Statut : TIMEOUT"
                else
                    STATUS="FAILED"
                    FAILED=$((FAILED + 1))
                    echo "Statut : FAILED ($EXIT_CODE)"
                    tail -n 20 "$STDOUT_FILE"
                fi

                echo \
"$RUN_ID,$PROFILE,$UE_COUNT,$DISTANCE,$SEED,$RUN_NUMBER,$STABLE_LAMBDA,$MEDIUM_LAMBDA,$HIGH_LAMBDA,$RECOVERY_LAMBDA,$PACKET_SIZE_BE,$PHASE1_END,$PHASE2_END,$PHASE3_END,$SIM_TIME,$STATUS,$EXIT_CODE" \
                    >> "$MANIFEST"

            done
        done
    done
done

echo
echo "============================================================"
echo "CAMPAGNE D2 TERMINÉE"
echo "Total  : $TOTAL"
echo "Succès : $SUCCESS"
echo "Échecs : $FAILED"
echo "============================================================"

if [[ $SUCCESS -eq 0 ]]; then
    exit 1
fi
