# Health Imaging Red-Team Framework

A research framework for systematically **red-teaming multimodal health pipelines**, with an initial focus on **ophthalmology (AMD, GA, wet AMD)**.

This project simulates realistic failure modes in clinical imaging workflows and measures how **LLMs / multimodal models** behave when:

- Metadata is missing  
- Images are low-quality  
- Laterality is mismatched  
- ICD history contradicts imaging  
- Medications imply a different stage than imaging  

It also supports **safety critics**, **abstention policies**, and **summary metrics** for safety and reliability.

---

## ✨ Key Ideas

- Turn your **AMD phenotyping QC intuition** into a **formal red-team protocol**
- Generate **clean + corrupted scenarios** from real clinical data
- Query a **clinical LLM (or multimodal model)** on each scenario
- Plug in a **critic model** that scores safety and flags contradictions
- Aggregate:
  - Accuracy
  - Hallucination rate
  - Abstention rate
  - Safety scores
  - Scenario-wise breakdowns

---

## 🧱 Repository Structure

```text
src/health_redteam/
  config.py       # Global configuration
  data.py         # Data loading & normalization
  corruptions.py  # Failure modes & scenario generation
  prompts.py      # Prompt builders for main model + critic
  models/
    base.py       # Abstract interfaces for ClinicalModel & CriticModel
    dummy.py      # Dummy models for dry-runs
  eval/
    metrics.py    # EvaluationRecord, metrics, hallucination logic
    experiment.py # Orchestrates full experiment
  scripts/
    run_experiment.py   # Main entrypoint
    visualize_metrics.py# Plots safety & performance metrics
```

```text
health-imaging-redteam/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── health_redteam/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── corruptions.py
│       ├── prompts.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── dummy.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── experiment.py
│       └── scripts/
│           ├── run_experiment.py
│           └── visualize_metrics.py
└── data/
    └── clean_ophthalmology_dataset.parquet  (your dataset goes here)
```

## Bash
```text
git clone <your-repo-url> health-imaging-redteam
cd health-imaging-redteam

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt
pip install -e .
