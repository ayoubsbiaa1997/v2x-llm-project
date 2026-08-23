# Reproducibility Guide

## 1. Scope

This document describes the main reproducibility path of the V2X ML–LLM project. Commands are intended to be executed from the repository root:

```bash
cd ~/v2x-llm-project
```

The repository contains additional exploratory, temporal, calibration, and handover experiments. The sequence below focuses on the final decision-support pipeline used for the main ML, LLM, and NGSIM validation results.

---

## 2. Environment

Create/activate the Python environment and install the project dependencies according to `ENVIRONMENT.md` and `requirements.txt`.

```bash
cd ~/v2x-llm-project
source .venv/bin/activate
pip install -r requirements.txt
```

The complete simulation stage additionally requires a compatible ns-3 / 5G-LENA environment. SUMO and Ollama are required for the corresponding mobility and local-LLM experiments.

---

## 3. V2X Simulation

The repository keeps the principal ns-3 sources in:

```text
ns3/v2x-5g-sumo.cc
ns3/v2x-5g-handover.cc
```

Simulation campaign scripts are located under:

```text
scripts/simulation/
scripts/handover/
```

The final decision-dataset builder expects the raw simulation metrics at:

```text
~/ns-3-dev/dataset_metrics.csv
```

Therefore, the simulation campaign must produce this file before rebuilding the final V3 dataset.

---

## 4. Build the Final Decision Dataset

Run:

```bash
python3 scripts/dataset/build_decision_dataset_v3.py
```

Main outputs:

```text
datasets/processed/final_dataset_v3.csv
datasets/llm/dataset_llm_v3.jsonl
datasets/processed/dataset_summary_v3.txt
datasets/processed/decision_policy_v3.csv
```

`final_dataset_v3.csv` is the principal dataset used by the final network-state ML training and LLM evaluation pipeline.

---

## 5. Train the Network-State Models

Run:

```bash
python3 scripts/ml/train_network_state_models.py
```

Input:

```text
datasets/processed/final_dataset_v3.csv
```

Main model directory:

```text
models/network_state_v1/
```

Main evaluation outputs:

```text
results/network_state_v1/model_metrics.csv
results/network_state_v1/out_of_fold_predictions.csv
results/network_state_v1/selected_features.json
results/network_state_v1/training_summary.txt
```

The pipeline predicts three decision-relevant continuous network-state variables:

- QoSScore
- CongestionScore
- HORiskScore

---

## 6. Validate the ML Pipeline

After training, run:

```bash
python3 scripts/ml/validate_ml_pipeline.py
```

The validation script uses:

```text
results/network_state_v1/out_of_fold_predictions.csv
results/network_state_v1/selected_features.json
```

and writes validation metrics, figures, tables, prediction-error statistics, and a validation manifest under:

```text
results/ml_validation_v2/
```

---

## 7. Run the LLM Evaluation

The final LLM evaluation script uses:

```text
datasets/processed/final_dataset_v3.csv
```

and relies on the ML inference and LLM-decision components.

Before running the benchmark, ensure that Ollama is available and that the required local models are installed.

The final experiments used:

- Llama 3.2 1B
- Qwen 2.5 3B
- DeepSeek-R1 8B

Check the exact command-line interface available in the repository with:

```bash
python3 scripts/llm/evaluate_llm_pilot.py --help
```

The completed 190-scenario evaluations are stored in:

```text
results/llm_evaluation/llama3_2_1b_pilot_190/
results/llm_evaluation/qwen2_5_3b_pilot_190/
results/llm_evaluation/deepseek-r1_8b_pilot_190/
```

Each experiment contains the selected scenarios, individual evaluation results, classification report, confusion matrix, and metrics.

---

## 8. Generate the Final LLM Benchmark

Once the three 190-scenario evaluation directories exist, run:

```bash
python3 scripts/llm/final_llm_benchmark.py
```

Default output directory:

```text
results/llm_evaluation/final_benchmark_190/
```

The benchmark aggregates the three LLM evaluations and produces comparison metrics and figures, including decision performance and performance–latency analyses.

Additional statistical and sensitivity analyses are implemented in:

```bash
python3 scripts/analysis/benchmark_statistics.py
python3 scripts/analysis/benchmark_sensitivity.py
python3 scripts/analysis/benchmark_visualization.py
```

---

## 9. NGSIM External Mobility Validation

### 9.1 Raw Data

The NGSIM raw dataset is intentionally excluded from the repository.

The preprocessing script expects:

```text
external_validation/ngsim/raw/us101.csv
```

The user must obtain the NGSIM US-101 dataset separately and place it at this location.

### 9.2 Audit and Preprocess NGSIM

The relevant scripts are:

```bash
python3 scripts/external_validation/audit_ngsim.py
python3 scripts/external_validation/preprocess_ngsim.py
```

The preprocessing stage generates, among other outputs:

```text
external_validation/ngsim/processed/ngsim_windows.csv
external_validation/ngsim/processed/ngsim_windows_ml_compatible.csv
results/external_validation/ngsim/preprocess_summary.json
```

NGSIM provides real vehicular trajectories but does not provide synchronized 5G radio measurements. This stage must therefore be interpreted as external mobility-domain validation rather than direct real-world radio validation.

### 9.3 Fixed Mapping ML Prediction

Run:

```bash
python3 scripts/external_validation/predict_ngsim_ml.py
```

This applies the trained network-state models to the ML-compatible NGSIM representation.

The standard prediction output is stored under:

```text
results/external_validation/ngsim/
```

### 9.4 Adaptive Mapping

The project also includes an adaptive scenario-mapping stage:

```bash
python3 scripts/external_validation/adaptive_scenario_generator.py
```

It transforms NGSIM mobility indicators into scenario variables compatible with the learned V2X model domain.

After generating the adaptive representation, apply the ML prediction pipeline according to the command-line options exposed by:

```bash
python3 scripts/external_validation/predict_ngsim_ml.py --help
```

The adaptive prediction results used by the external LLM validation are stored under:

```text
results/external_validation/ngsim/adaptive_ml/
```

### 9.5 Compare Fixed and Adaptive Mapping

Run:

```bash
python3 scripts/external_validation/compare_fixed_vs_adaptive_ngsim.py
```

The comparison outputs are stored under:

```text
results/external_validation/ngsim/comparison/
```

This comparison analyzes score dispersion and saturation. It is not a supervised radio-performance evaluation because NGSIM does not contain 5G ground-truth metrics.

### 9.6 External LLM Validation

The external LLM validation uses by default:

```text
results/external_validation/ngsim/adaptive_ml/ml_predictions_ngsim.csv
```

Check its execution options with:

```bash
python3 scripts/external_validation/validate_external_llm_pipeline.py --help
```

The corresponding results are written under:

```text
results/external_validation/ngsim/llm_validation/
```

Then run the common-subset comparison and final external analysis:

```bash
python3 scripts/external_validation/compare_external_llm_common_subset.py
python3 scripts/external_validation/external_validation_analysis.py
```

---

## 10. Main Reproducibility Chain

The final experimental chain can be summarized as:

```text
ns-3 / 5G-LENA simulations
        |
        v
dataset_metrics.csv
        |
        v
build_decision_dataset_v3.py
        |
        v
final_dataset_v3.csv
        |
        v
train_network_state_models.py
        |
        +--> trained ML models
        +--> out-of-fold predictions
        |
        v
validate_ml_pipeline.py
        |
        v
LLM evaluation on 190 scenarios
        |
        v
final_llm_benchmark.py
        |
        +-------------------------------+
        |                               |
        v                               v
Internal benchmark              NGSIM mobility validation
                                        |
                                        v
                              Adaptive domain mapping
                                        |
                                        v
                              External LLM evaluation
```

---

## 11. Reproducibility Notes

- Run project scripts from the repository root unless explicitly stated otherwise.
- Raw third-party mobility datasets are not distributed with this repository.
- The final V3 decision dataset is generated from `~/ns-3-dev/dataset_metrics.csv`.
- The main final ML models are stored under `models/network_state_v1/`.
- The primary LLM comparison uses the three 190-scenario result directories.
- LLM inference latency depends on local hardware and Ollama configuration and should not be expected to reproduce exactly across machines.
- NGSIM validation assesses mobility-domain transferability, not direct real-world 5G radio accuracy.
- Several exploratory and temporal experiments remain in the repository for scientific traceability but are not required to reproduce the main final benchmark.
