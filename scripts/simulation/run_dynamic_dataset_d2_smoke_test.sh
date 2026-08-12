#!/usr/bin/env bash

set -u
set -o pipefail

NS3_DIR="$HOME/ns-3-dev"
PROJECT_DIR="$HOME/v2x-llm-project"

OUT_DIR="$PROJECT_DIR/datasets/raw/dynamic_dataset_d2_pilot"
MANIFEST="$OUT_DIR/campaign_manifest.csv"

UE_COUNTS=(8)
DISTANCES=(100)
SEEDS=(111)

PROFILES=(
    "EARLY_DEGRADATION"
)

SIM_TIME=20
PACKET_SIZE_BE=1200

mkdir -p "$OUT_DIR"

echo \
"run_id,profile,ue_count,distance_m,seed,run_number,stable_lambda,medium_lambda,high_lambda,recovery_lambda,packet_size_be,phase1_end,phase2_end,phase3_end,sim_time,status,exit_code" \
> "$MANIFEST"

cd "$NS3_DIR" || {
    echo "Erreur : dossier ns-3 introuvable : $NS3_DIR"
    exit 1
}

TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

for PROFILE in "${PROFILES[@]}"; do
    for UE_COUNT in "${UE_COUNTS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for SEED_INDEX in "${!SEEDS[@]}"; do

                SEED="${SEEDS[$SEED_INDEX]}"
                RUN_NUMBER=$((SEED_INDEX + 1))

                STABLE_LAMBDA=50
                RECOVERY_LAMBDA=100

                case "$UE_COUNT" in
                    8)
                        BASE_MEDIUM=400
                        BASE_HIGH=1400
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
                        ;;

                    LATE_DEGRADATION)
                        PHASE1_END=$((6 + SEED_INDEX))
                        PHASE2_END=$((12 + SEED_INDEX))
                        PHASE3_END=$((16 + SEED_INDEX))
                        ;;

                    ABRUPT_DEGRADATION)
                        PHASE1_END=$((5 + SEED_INDEX))
                        PHASE2_END=$((6 + SEED_INDEX))
                        PHASE3_END=$((13 + SEED_INDEX))

                        MEDIUM_LAMBDA=$HIGH_LAMBDA
                        ;;

                    MEDIUM_CONTROL)
                        PHASE1_END=$((4 + SEED_INDEX))
                        PHASE2_END=$((9 + SEED_INDEX))
                        PHASE3_END=$((14 + SEED_INDEX))

                        HIGH_LAMBDA=$MEDIUM_LAMBDA
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

                PROFILE_LOWER=$(
                    echo "$PROFILE" |
                    tr '[:upper:]' '[:lower:]'
                )

                RUN_ID="d2_${PROFILE_LOWER}_u${UE_COUNT}_d${DISTANCE}_s${SEED}"

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
                echo "Profil           : $PROFILE"
                echo "UE               : $UE_COUNT"
                echo "Distance         : $DISTANCE"
                echo "Seed             : $SEED"
                echo "Stable lambda    : $STABLE_LAMBDA"
                echo "Medium lambda    : $MEDIUM_LAMBDA"
                echo "High lambda      : $HIGH_LAMBDA"
                echo "Recovery lambda  : $RECOVERY_LAMBDA"
                echo "Transitions      : $PHASE1_END / $PHASE2_END / $PHASE3_END"
                echo "Simulation       : $TOTAL_RUNS / 48"
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

                    if [[ -f "$STDOUT_FILE" ]]; then
                        tail -n 30 "$STDOUT_FILE"
                    fi
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
echo "CAMPAGNE PILOTE D2 TERMINÉE"
echo "Runs attendus : 48"
echo "Runs exécutés : $TOTAL_RUNS"
echo "Succès        : $SUCCESS_RUNS"
echo "Échecs        : $FAILED_RUNS"
echo "Manifest      : $MANIFEST"
echo "Dossier       : $OUT_DIR"
echo "============================================================"

if [[ $SUCCESS_RUNS -eq 0 ]]; then
    exit 1
fi

if [[ $FAILED_RUNS -gt 0 ]]; then
    exit 2
fi

exit 0
