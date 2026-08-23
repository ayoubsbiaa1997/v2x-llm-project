# Intelligent V2X Decision Support using ML and LLMs over 5G/6G Networks

## Overview

This repository contains the implementation, datasets, experimental results, and validation pipeline developed for the research project:

**“Large Language Models for Intelligent Decision-Making in V2X Communications over 5G/6G Networks.”**

The project investigates a hybrid architecture combining:

- 5G V2X network simulation using **ns-3 / 5G-LENA**;
- mobility-aware and traffic-aware network characterization;
- Machine Learning models for network-state prediction;
- deterministic expert policies for reference decision-making;
- Large Language Models (LLMs) for high-level network reasoning;
- external mobility validation using **NGSIM** real-world vehicle trajectories.

The main objective is to study whether predicted V2X network conditions can be transformed into reliable and interpretable high-level network-management decisions.

---

## Proposed Architecture

The proposed pipeline follows the sequence:

```text
V2X Scenario
     |
     v
ns-3 / 5G-LENA Simulation
     |
     v
Dataset Construction
     |
     v
ML Network-State Prediction
     |
     +--> QoS Score
     +--> Congestion Score
     +--> Handover Risk Score
     |
     v
State Discretization
     |
     +----------------------+
     |                      |
     v                      v
Expert Policy          Structured LLM Prompt
     |                      |
     v                      v
Reference Decision      LLM Decision + Explanation
     |                      |
     +----------+-----------+
                |
                v
        Comparative Evaluation
```

The expert-policy decision is kept independent from the LLM prompt in order to avoid direct decision-label leakage.

---

## Main Research Questions

The project investigates four main questions:

1. Can ML models predict decision-relevant V2X network states from network, traffic, and mobility conditions?
2. Can LLMs transform predicted V2X states into high-level decisions coherent with an independent expert policy?
3. How do different LLMs compare in terms of decision quality, robustness, explainability, and inference latency?
4. Does the pipeline preserve meaningful behavior when exposed to mobility patterns outside the original simulation domain?

---

## Experimental Dataset

The final V2X dataset contains **190 valid observations** generated from heterogeneous ns-3 / 5G-LENA scenarios.

Main scenario variables include:

- number of gNBs;
- UEs per gNB;
- total number of UEs;
- inter-site distance;
- vehicle speed;
- ULL traffic load;
- BE traffic load.

The final ML feature set contains seven scenario-level variables.

Post-simulation QoS and handover measurements are excluded from the ML input features to reduce information leakage.

---

## Network-State Prediction

Three continuous targets are predicted:

- **QoSScore**
- **CongestionScore**
- **HORiskScore**

The evaluated regression algorithms include:

- Random Forest;
- Extra Trees;
- Histogram-based Gradient Boosting.

Model evaluation uses repeated 5-fold cross-validation and out-of-fold predictions.

### Selected Models

| Target | Selected Model | OOF R² |
|---|---|---:|
| QoS | Histogram-based Gradient Boosting | 0.7412 |
| Congestion | Histogram-based Gradient Boosting | 0.6163 |
| Handover Risk | Extra Trees | 0.8990 |

The handover-risk result must be interpreted cautiously because the final valid simulation set does not contain handover failures.

---

## LLM Decision Benchmark

Three open-weight LLMs were evaluated under the same zero-shot protocol:

- **Llama 3.2 1B**
- **Qwen 2.5 3B**
- **DeepSeek-R1 8B**

All models were executed locally through Ollama.

### Main Results

| Model | Accuracy | Macro-F1 | Mean Latency |
|---|---:|---:|---:|
| Llama 3.2 1B | 0.3474 | 0.1031 | 19.72 s |
| Qwen 2.5 3B | 0.3737 | 0.1621 | 39.37 s |
| DeepSeek-R1 8B | **0.3947** | **0.1976** | 101.78 s |

All three models produced valid structured outputs for all 190 evaluated scenarios.

The results reveal a clear trade-off between decision performance and inference latency.

---

## External Validation using NGSIM

The pipeline was additionally evaluated using mobility traces derived from the **NGSIM US-101** dataset.

NGSIM is used as an external mobility source and not as a real 5G radio ground truth.

A fixed mapping initially caused strong prediction saturation.

An adaptive mapping strategy reduced the dominant-value concentration for QoS and congestion predictions from:

```text
97.89% -> 67.89%
```

This experiment evaluates mobility-domain transferability rather than full real-world 5G V2X validation.

---

## Repository Structure

```text
v2x-llm-project/
│
├── datasets/
│   ├── analysis/
│   ├── llm/
│   ├── ml/
│   ├── mobility/
│   ├── processed/
│   └── raw/
│
├── external_validation/
│   └── ngsim/
│
├── figures/
│   ├── llm_benchmark/
│   ├── ml/
│   ├── ml_comparison/
│   └── temporal/
│
├── models/
│   ├── network_state_v1/
│   ├── preprocessing/
│   └── trained/
│
├── ns3/
│   ├── v2x-5g-handover.cc
│   └── v2x-5g-sumo.cc
│
├── report/
│   └── latex/
│
├── results/
│   ├── external_validation/
│   ├── inference/
│   ├── llm_evaluation/
│   ├── ml/
│   ├── ml_audit_v1/
│   ├── ml_validation_v2/
│   └── network_state_v1/
│
├── scripts/
│   ├── analysis/
│   ├── dataset/
│   ├── external_validation/
│   ├── handover/
│   ├── llm/
│   ├── ml/
│   ├── mobility/
│   └── simulation/
│
├── src/
│   └── ml/
│
├── ENVIRONMENT.md
├── SCIENTIFIC_MANIFEST.md
├── requirements.txt
└── README.md
```

---

## Important Result Directories

### Network-State Prediction

`results/network_state_v1/`

Contains:

- model metrics;
- out-of-fold predictions;
- selected features;
- training summary.

### Final LLM Benchmark

`results/llm_evaluation/final_benchmark_190/`

Contains:

- final model comparison;
- class-wise metrics;
- benchmark summary;
- latency analysis;
- generated figures.

### Statistical LLM Evaluation

`results/llm_evaluation/benchmark_statistics_190/`

### Performance-Latency Sensitivity Analysis

`results/llm_evaluation/benchmark_sensitivity_190/`

### NGSIM External Validation

`results/external_validation/ngsim/`

---

## Environment

The project was developed using:

- Ubuntu / WSL2;
- Python;
- ns-3;
- 5G-LENA;
- SUMO;
- scikit-learn;
- Ollama;
- Llama 3.2;
- Qwen 2.5;
- DeepSeek-R1.

See `ENVIRONMENT.md` and `requirements.txt` for environment and dependency information.

---

## Reproducibility

The project separates:

```text
Simulation
   -> Dataset generation
   -> ML prediction
   -> Expert policy
   -> LLM inference
   -> Statistical evaluation
   -> External validation
```

Experiment scripts are organized by function inside the `scripts/` directory.

Raw third-party datasets such as NGSIM and LuSTScenario are not included in this repository and must be obtained separately from their original sources.

---

## Reproducibility

The complete experimental workflow, including dataset construction, ML training, LLM evaluation, benchmarking, and NGSIM external validation, is documented in:

➡️ **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**

This guide provides the main execution sequence, expected inputs and outputs, and important notes required to reproduce the principal experimental results.

---

## Scientific Limitations

The current study has several limitations:

- the primary dataset contains 190 valid observations;
- no handover failures are present in the final valid simulation set;
- LLM decision agreement remains limited under zero-shot prompting;
- local LLM inference latency is too high for direct real-time V2X control;
- NGSIM provides real mobility trajectories but not synchronized real-world 5G radio measurements.

The proposed framework should therefore be interpreted as a **high-level intelligent decision-support architecture**, rather than a replacement for low-level real-time cellular control mechanisms.

---

## Future Work

Future work includes:

- expanding the ns-3 simulation campaign;
- introducing richer handover-failure scenarios;
- integrating real radio and mobility measurements;
- fine-tuning lightweight LLMs for V2X decision alignment;
- uncertainty-aware ML prediction;
- adaptive decision policies;
- additional external validation datasets.

---

## Author

**Ayoub Sbiaa**

National Institute of Posts and Telecommunications (INPT)  
Rabat, Morocco

---

## Repository Status

This repository contains the experimental implementation and results associated with an ongoing scientific research manuscript.

The repository is currently maintained as a **private research repository**.
