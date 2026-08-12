# Scientific Manifest — Final V2X ML–LLM Pipeline

## 1. Reference Version

Reference Git commit:

0caddf0

This commit corresponds to the consolidated and validated experimental version
of the V2X ML–LLM PFE pipeline.

---

## 2. Scientific Objective

The project investigates a hybrid architecture for intelligent decision-making
in 5G/6G V2X communications.

The retained architecture combines:

1. ns-3 / 5G-LENA network simulation;
2. SUMO vehicular mobility;
3. structured V2X dataset generation;
4. machine-learning-based network-state prediction;
5. deterministic expert decision policy;
6. Large Language Model reasoning and explanation;
7. external mobility validation using NGSIM US-101.

The LLM is considered a high-level decision-support and explanation layer,
not a direct low-level radio controller.

---

## 3. Main Experimental Pipeline

~~~text
SUMO / ns-3 / 5G-LENA
          |
          v
Experimental dataset
          |
          v
ML network-state prediction
          |
          +---------------------+
          |          |          |
          v          v          v
      QoSScore  CongestionScore HORiskScore
          |          |          |
          +----------+----------+
                     |
                     v
             State discretization
                     |
                     v
               Expert policy
                     |
                     v
             Prompt construction
                     |
                     v
                    LLM
                     |
                     v
           Decision + Explanation
                     |
                     v
           Consistency evaluation
~~~

---

## 4. Reference Datasets

### 4.1 Main simulation dataset

File:

datasets/processed/final_dataset.csv

Size:

190 experimental observations + header.

Purpose:

Reference dataset containing simulated V2X configuration, QoS, congestion,
handover and decision information.

### 4.2 Decision-oriented dataset

File:

datasets/processed/final_dataset_v3.csv

Size:

190 experimental observations + header.

Important variables include:

- QoSScore
- CongestionScore
- HORiskScore
- NetworkSeverityScore
- QoSLevel
- CongestionLevel
- HORisk
- Decision
- PolicyRule
- PolicyReason
- Explanation

This dataset is the principal decision-oriented representation of the internal
simulation campaign.

### 4.3 Handover dataset

File:

datasets/processed/handover_dataset_v6.csv

Purpose:

Aggregated handover information derived from the retained handover campaign V3.

### 4.4 Main temporal dataset

File:

datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4.csv

Observations:

1071 + header.

### 4.5 Handover-enriched temporal dataset

File:

datasets/processed/mobility/temporal_v2x_synchronized_dataset_v6.csv

Observations:

1071 + header.

---

## 5. Machine-Learning Network-State Prediction

### 5.1 Input variables

The retained ML models use seven simulation configuration variables:

- gNbNum
- ueNumPerGnb
- totalUE
- bsDistance
- ueSpeed
- lambdaULL
- lambdaBe

Constant variables removed:

- scenarioLength
- packetULL
- packetBE

Measured QoS outputs, qualitative levels and final decisions are excluded from
the ML inputs in order to reduce direct target leakage.

### 5.2 Validation protocol

Number of observations:

190

Validation strategy:

RepeatedKFold — 5 folds × 5 repetitions.

The final validation metrics are based on out-of-fold predictions.

### 5.3 Retained models

QoSScore:
HistGradientBoosting

CongestionScore:
HistGradientBoosting

HORiskScore:
ExtraTrees

The HORiskScore model was retained according to the selected CV MAE criterion.

---

## 6. Main ML Results

### QoSScore

OOF MAE:

0.0772

OOF RMSE:

0.1003

OOF R²:

0.7412

After discretization:

Accuracy = 0.7316

Macro-F1 = 0.7131

Interpretation:

QoSScore presents a satisfactory predictive structure within the simulated
experimental domain.

### CongestionScore

OOF MAE:

0.1065

OOF RMSE:

0.1385

OOF R²:

0.6163

After discretization:

Accuracy = 0.6368

Macro-F1 = 0.6083

Interpretation:

CongestionScore is the most difficult of the three continuous network-state
targets to estimate.

### HORiskScore

OOF MAE:

0.0178

OOF RMSE:

0.0289

OOF R²:

0.8990

After discretization:

Accuracy = 0.9368

Global Macro-F1 = 0.4663

Active-class Macro-F1 = 0.9325

Important limitation:

The handover-risk result must be interpreted cautiously because the valid
simulation observations contain no real handover failures and only two
qualitative classes are effectively active.

The continuous regression result is therefore more scientifically meaningful
than the apparent classification accuracy.

---

## 7. Temporal Prediction Branch

The temporal prediction branch is retained as an exploratory component.

Initial models produced near-perfect scores:

- Decision Tree: 1.00
- Random Forest: 1.00
- XGBoost: 1.00
- MLP: 1.00

These scores are NOT retained as evidence of strong temporal prediction.

### Temporal audit

Dataset:

1071 observations

Number of scenarios:

9

Total observed decision transitions:

1

Total degradations:

1

Total recoveries:

0

Eight of the nine scenarios contain no decision transition.

### Persistence baseline

Horizon 1:

Accuracy = 0.9991

Macro-F1 = 0.9992

Horizon 10:

Accuracy = 0.9898

Macro-F1 = 0.9909

Scientific conclusion:

The near-perfect temporal results are mainly caused by label persistence and
insufficient transition diversity.

The temporal branch must therefore be presented as exploratory and not as a
main proof of predictive capability.

---

## 8. Expert Decision Policy

The expert policy provides a deterministic decision reference derived from
predicted network-state conditions.

Its objectives are:

- decision reproducibility;
- controlled decision space;
- separation between prediction and reasoning;
- reference policy for LLM evaluation;
- improved interpretability.

The LLM does not replace this policy.

---

## 9. Internal LLM Benchmark

Reference:

results/llm_evaluation/final_benchmark_190/

Number of scenarios per model:

190

Models:

- Llama 3.2 1B
- Qwen 2.5 3B
- DeepSeek-R1 8B

All three models achieved:

JSONValidRate = 1.0

### Llama 3.2 1B

Accuracy:

0.3474

Macro-F1:

0.1031

Mean inference latency:

19.72 s

Interpretation:

Fastest evaluated model, but lowest agreement with the expert-policy reference.

### Qwen 2.5 3B

Accuracy:

0.3737

Macro-F1:

0.1621

Mean inference latency:

39.37 s

Interpretation:

Intermediate performance and strong structured-output robustness.

### DeepSeek-R1 8B

Accuracy:

0.3947

Macro-F1:

0.1976

Mean inference latency:

101.78 s

Interpretation:

Highest agreement and Macro-F1 among the evaluated models, but with a very
high inference cost.

---

## 10. LLM Multicriteria Ranking

Weights used:

- Accuracy: 0.25
- Macro-F1: 0.35
- Macro-Precision: 0.10
- Macro-Recall: 0.10
- JSON valid rate: 0.05
- Latency score: 0.15

Ranking:

1. DeepSeek-R1 8B — 0.8065
2. Qwen 2.5 3B — 0.6349
3. Llama 3.2 1B — 0.2000

Important methodological note:

This ranking is conditional on the selected weights.

It must not be presented as an absolute ranking of the models.

---

## 11. External Validation — NGSIM

Dataset:

NGSIM US-101.

NGSIM contains real vehicular trajectories but does not contain 5G radio
ground truth.

The external-validation pipeline therefore evaluates functional transferability
under an independent mobility distribution rather than direct real-world radio
accuracy.

### External workflow

~~~text
NGSIM trajectories
       |
       v
Mobility preprocessing
       |
       v
Adaptive mapping
       |
       v
ML network-state prediction
       |
       v
Transferred expert policy
       |
       v
LLM evaluation
~~~

### Common evaluable subset

The final common subset contains:

20 scenarios.

Transferred policy distribution:

- MONITOR_NETWORK: 5
- PREPARE_HANDOVER: 5
- HANDOVER_NOW: 10

### Main external LLM results

DeepSeek-R1 8B:

Best decision agreement = 0.417

Llama 3.2 1B:

Fastest model = 13.42 s

Qwen 2.5 3B:

Best parse/exploitable-response rate = 100%

Scientific interpretation:

The qualitative performance–latency trade-off observed internally remains
visible under an external mobility distribution.

This is evidence of behavioral transferability, not proof of 5G radio
generalization.

---

## 12. Main Scientific Claims Supported by the Experiments

The following conclusions are considered defensible:

1. Network-state indicators can be estimated from simulation configuration
   variables with meaningful OOF predictive performance.

2. QoSScore and CongestionScore show moderate-to-good predictability within the
   simulated domain.

3. HORiskScore is highly predictable as a continuous pseudo-score, but its
   qualitative classification is limited by insufficient handover-failure
   diversity.

4. The expert policy provides a reproducible decision reference for LLM
   evaluation.

5. The evaluated LLMs exhibit a clear trade-off between decision agreement,
   structured-output robustness and latency.

6. DeepSeek achieves the strongest agreement under the retained evaluation
   criteria, while Llama is substantially faster.

7. External NGSIM experiments support functional transferability of the
   pipeline to an independent mobility distribution.

8. The temporal-model 100% scores are not considered evidence of genuine
   temporal prediction because a persistence baseline obtains nearly equivalent
   performance.

---

## 13. Claims That Must NOT Be Made

The following statements are not scientifically supported by the experiments:

- The temporal model predicts future network decisions with 100% reliability.
- The HORisk classifier detects real handover failures with 93.7% generalization.
- DeepSeek is universally the best LLM.
- NGSIM validates real 5G QoS performance.
- The LLM can safely replace low-level network-control mechanisms.
- The current pipeline is production-ready for real-time V2X control.

---

## 14. Main Limitations

- Internal simulation dataset limited to 190 observations.
- Network-state targets are constructed scores / pseudo-labels.
- Limited real handover-failure diversity.
- Temporal labels are strongly persistent.
- LLM agreement metrics remain modest in absolute terms.
- DeepSeek inference latency is high.
- NGSIM provides mobility ground truth but no radio ground truth.
- External network parameters require domain mapping.
- External common LLM subset is limited to 20 fully comparable scenarios.

---

## 15. Scientific Status

The retained experimental evidence supports the hybrid architecture:

~~~text
ML network-state estimation
          +
Controlled expert policy
          +
LLM reasoning / explanation
          +
External mobility validation
~~~

The principal evidence for the final PFE is based on:

1. OOF ML network-state validation;
2. expert-policy decision construction;
3. 190-scenario internal LLM benchmark;
4. external NGSIM transferability evaluation.

The temporal branch remains exploratory.

