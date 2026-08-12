# Intelligent Decision-Making for V2X Communications Using ML and Large Language Models

## Overview

This repository contains the experimental implementation developed for the engineering final-year project:

**“Utilisation des Large Language Models pour la prise de décision intelligente dans les communications V2X sur réseau 5G/6G.”**

The project investigates a hybrid intelligent decision-making architecture for V2X communications combining:

- vehicular mobility simulation;
- 5G V2X network simulation;
- QoS and handover analysis;
- machine-learning-based network-state prediction;
- an expert decision policy;
- Large Language Models (LLMs);
- explainable decision generation;
- external validation using real vehicular mobility data.

The proposed architecture separates network-state prediction, controlled decision-making, and high-level reasoning.

## Proposed Architecture

The experimental pipeline can be summarized as follows:

~~~text
V2X Simulation / Mobility Data
            |
            v
      Data Processing
            |
            v
   ML State Prediction
      |      |      |
      v      v      v
     QoS  Congestion  HO Risk
      \      |      /
             v
      State Discretization
             |
             v
        Expert Policy
             |
             v
      Prompt Construction
             |
             v
            LLM
             |
             v
   Decision + Explanation
             |
             v
    Consistency Validation
~~~

The machine-learning layer estimates the network state, while the expert policy provides a deterministic reference decision. The LLM operates as a high-level reasoning and explanation layer rather than directly controlling low-level radio procedures.

## Main Technologies

The experimental environment relies on:

- **ns-3** for discrete-event network simulation;
- **5G-LENA** for 5G NR modeling;
- **SUMO** for vehicular mobility;
- **Python 3.12** for data processing and experimentation;
- **Scikit-learn** for machine-learning pipelines;
- **XGBoost** for gradient-boosted models;
- **PyTorch** for temporal neural models;
- **SHAP** for model explainability;
- **NumPy** and **Pandas** for data processing;
- **Matplotlib** and **Seaborn** for experimental visualization.

External mobility validation is based on the **NGSIM US-101** vehicular trajectory dataset.

## Repository Structure

~~~text
v2x-llm-project/
├── datasets/
├── external_validation/
├── figures/
├── models/
├── ns3/
├── report/
├── results/
├── scripts/
│   ├── analysis/
│   ├── dataset/
│   ├── external_validation/
│   ├── handover/
│   ├── llm/
│   ├── ml/
│   ├── mobility/
│   └── simulation/
├── src/
│   └── ml/
├── ENVIRONMENT.md
├── requirements.txt
└── README.md
~~~

## V2X Simulation

The network experimentation stage is based on ns-3 and 5G-LENA.

The simulations investigate variations in:

- number of vehicles;
- gNB deployment conditions;
- vehicle speed;
- traffic intensity;
- packet size;
- network load.

The main network indicators include:

- throughput;
- delay;
- jitter;
- packet loss;
- packet delivery ratio;
- congestion;
- handover attempts;
- handover success and failure;
- handover delay.

The principal ns-3 source files are:

~~~text
ns3/v2x-5g-sumo.cc
ns3/v2x-5g-handover.cc
~~~

## Dataset Construction

The main processed dataset is:

~~~text
datasets/processed/final_dataset.csv
~~~

The decision-oriented enriched version is:

~~~text
datasets/processed/final_dataset_v3.csv
~~~

This second dataset incorporates the information required by the decision pipeline, including network-state scores and expert-policy outputs.

## Machine Learning

The ML pipeline predicts three network-state indicators:

- **QoSScore**
- **CongestionScore**
- **HORiskScore**

The corresponding trained models are stored in:

~~~text
models/network_state_v1/
~~~

The main ML implementation is located in:

~~~text
src/ml/
scripts/ml/
~~~

## Mobility and Temporal Modeling

The mobility pipeline uses SUMO-derived vehicular information and temporal windows.

The retained experiments include:

- synchronized QoS/mobility datasets;
- temporal sequence generation;
- Random Forest models;
- LSTM models;
- GRU models;
- handover-aware temporal enrichment.

Related scripts are located in:

~~~text
scripts/mobility/
scripts/ml/
~~~

## Handover Analysis

The retained handover branch is based on:

~~~text
datasets/processed/handover_campaign_v3/
~~~

The aggregated handover dataset is:

~~~text
datasets/processed/handover_dataset_v6.csv
~~~

Associated scripts are available in:

~~~text
scripts/handover/
~~~

## Expert Decision Policy

A deterministic expert policy is used as a controlled decision reference.

It maps predicted QoS, congestion and handover-risk conditions to a constrained decision space.

This provides:

- reproducibility;
- decision consistency;
- a reference for LLM evaluation;
- separation between prediction and reasoning.

The LLM therefore does not directly replace lower-layer network control.

## Large Language Model Evaluation

Three LLM families were evaluated:

- **Llama 3.2 1B**
- **Qwen 2.5 3B**
- **DeepSeek-R1 8B**

The evaluation considers:

- agreement with the expert policy;
- robustness;
- structured-output validity;
- Macro-F1;
- explanation quality;
- inference latency.

Final benchmark results are available in:

~~~text
results/llm_evaluation/
~~~

including:

~~~text
final_benchmark_190/
benchmark_statistics_190/
benchmark_sensitivity_190/
benchmark_visualization_190/
~~~

The final benchmark contains 190 scenarios.

## External Validation with NGSIM

An external validation stage is included to evaluate the generalization of the proposed decision pipeline beyond the internally generated simulation scenarios.

The validation uses mobility information derived from the NGSIM US-101 trajectory dataset.

The corresponding workflow is located in:

~~~text
scripts/external_validation/
external_validation/ngsim/
results/external_validation/
~~~

The raw NGSIM trajectory data are not redistributed with this repository. They must be obtained separately from the original data source.

The external-validation workflow includes:

1. preprocessing of NGSIM trajectories;
2. mobility feature extraction;
3. adaptation to the V2X decision pipeline;
4. ML-based network-state prediction;
5. expert-policy evaluation;
6. LLM decision evaluation;
7. comparison between internal and external scenarios.

This validation is intended to assess the robustness and transferability of the proposed architecture under mobility conditions not directly generated by the primary simulation campaign.

## Experimental Environment

The validated Python environment is documented in:

~~~text
ENVIRONMENT.md
~~~

The main software environment includes:

- Python 3.12.3
- NumPy 2.5.1
- Pandas 3.0.5
- Matplotlib 3.11.1
- Seaborn 0.13.2
- Scikit-learn 1.9.0
- XGBoost 3.4.0
- Joblib 1.5.3
- SHAP 0.52.0
- PyTorch 2.13.0

The local PyTorch environment was validated with a CUDA-enabled build.

## Installation

Create and activate a Python virtual environment:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

Upgrade pip:

~~~bash
python -m pip install --upgrade pip
~~~

Install the project dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

Verify the environment:

~~~bash
python -m pip check
~~~

Validate the Python source code:

~~~bash
python -m compileall -q scripts src
~~~

## Reproducibility

The project separates the experimental workflow into several stages:

~~~text
Simulation
    |
    v
Raw measurements
    |
    v
Dataset construction
    |
    v
Mobility / QoS / Handover synchronization
    |
    v
Network-state prediction
    |
    v
Expert decision policy
    |
    v
LLM reasoning
    |
    v
Consistency and performance evaluation
~~~

The directory organization reflects this separation to facilitate experimental reproducibility and traceability.

Simulation scripts are located in:

~~~text
scripts/simulation/
~~~

Dataset construction scripts are located in:

~~~text
scripts/dataset/
~~~

Mobility processing scripts are located in:

~~~text
scripts/mobility/
~~~

Handover processing scripts are located in:

~~~text
scripts/handover/
~~~

ML experiments are located in:

~~~text
scripts/ml/
~~~

LLM experiments are located in:

~~~text
scripts/llm/
~~~

Analysis and visualization scripts are located in:

~~~text
scripts/analysis/
~~~

## Scientific Scope

The proposed architecture follows a hybrid approach rather than delegating direct radio-resource control to a generative model.

The roles are separated as follows:

- **ns-3 / 5G-LENA** provides controlled network experimentation;
- **SUMO** provides vehicular mobility information;
- **ML models** estimate relevant network-state indicators;
- **the expert policy** constrains the admissible decision space;
- **the LLM** performs high-level contextual reasoning and generates an explanation;
- **the validation layer** checks decision consistency.

This separation is intended to improve interpretability, reproducibility and control of the decision process.

## Limitations

The experimental results must be interpreted within the scope of the implemented scenarios.

In particular:

- ns-3 simulations represent controlled approximations of real V2X environments;
- the generated datasets cover a finite set of network and mobility configurations;
- handover behavior depends on the implemented 5G-LENA configuration;
- ML performance depends on the representativeness of the training scenarios;
- LLM agreement with the expert policy does not by itself demonstrate optimal radio-resource management;
- external NGSIM validation primarily strengthens the mobility-generalization analysis and does not constitute a complete real-world 5G network validation.

Consequently, the proposed system should be interpreted as a research prototype for intelligent high-level V2X decision support rather than as a production-ready autonomous network controller.

## Repository Status

This repository contains the consolidated experimental implementation retained for the final PFE workflow.

Legacy experiments and intermediate artifacts that are not required by the retained pipeline are excluded from the active project structure.

The principal retained components are:

- ns-3 / 5G-LENA V2X simulations;
- QoS experimentation;
- SUMO mobility processing;
- handover analysis;
- synchronized temporal datasets;
- ML network-state prediction;
- expert decision policy;
- LLM evaluation;
- external NGSIM validation;
- scientific figures and experimental results;
- LaTeX report material.

## Academic Context

This work was developed as part of an engineering final-year project (PFE) on the use of Large Language Models for intelligent decision-making in V2X communications over 5G/6G networks.

The repository is intended to support experimental traceability, reproducibility of the retained pipeline, and the preparation of the associated academic report and scientific publication.
