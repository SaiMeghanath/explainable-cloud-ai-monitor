# Explainable Cloud AI Monitor

Integrating Google Cloud ML APIs with post-hoc Explainability (LIME & SHAP) to make cloud-deployed model predictions transparent and interpretable.


## Motivation

Cloud ML APIs (Google AutoML, Vertex AI, etc.) are powerful but opaque — they return predictions without explaining why. In high-stakes domains like healthcare, finance, or content moderation, a prediction without justification is often unusable. This project wraps Google Cloud ML predictions with LIME and SHAP explanations, creating a monitoring layer that answers the question every deployment should ask: *what is the model actually looking at?*

**Explainability is not a nice-to-have. It is the difference between a model you can trust and one you can only guess at.**

## System Design

```
Input Data
    │
    ▼
Google Cloud ML API
(AutoML / Vertex AI Prediction)
    │
    ▼
Prediction Output
    │
    ├──► LIME Explainer
    │    (local, perturbation-based)
    │    → Feature importance for individual prediction
    │
    └──► SHAP Explainer
         (global + local, game-theoretic)
         → Shapley values per feature
              │
              ▼
         Visualization Layer
         (feature importance plots,
          SHAP summary & force plots)
              │
              ▼
         Flask / Notebook Dashboard
```

## What It Does

- Sends input data to a Google Cloud ML API endpoint and retrieves predictions
- Applies **LIME** (Local Interpretable Model-agnostic Explanations) to explain individual predictions by perturbing inputs and observing output changes
- Applies **SHAP** (SHapley Additive exPlanations) to compute feature contributions using game-theoretic Shapley values — both locally (per prediction) and globally (across the dataset)
- Visualizes explanations as feature importance bar charts, SHAP summary plots, and force plots
- Provides a monitoring interface to track explanation drift over time

## LIME vs SHAP — When to Use Which

| Aspect | LIME | SHAP |
|--------|------|------|
| **Approach** | Local perturbation | Game-theoretic (Shapley values) |
| **Scope** | Individual predictions | Individual + global |
| **Speed** | Fast | Slower (exact) |
| **Consistency** | Can vary across runs | Theoretically consistent |
| **Best for** | Quick local explanations | Rigorous feature attribution |

Both are **model-agnostic** — they treat the underlying Cloud ML model as a black box, making this approach generalizable to any API-served model.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Cloud ML** | Google Cloud ML APIs / Vertex AI |
| **Explainability** | SHAP, LIME (lime) |
| **Visualization** | Matplotlib, SHAP plots |
| **Web Interface** | Flask |
| **Notebooks** | Jupyter |

## Project Structure

```
explainable-cloud-ai-monitor/
├── app/
│   ├── predict.py          # Cloud API call wrapper
│   ├── explainer.py        # LIME + SHAP explanation logic
│   └── visualize.py        # Plotting utilities
├── notebooks/              # Exploration & demonstration notebooks
├── tests/                  # Unit tests
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

## Setup & Usage

### 1. Clone

```bash
git clone https://github.com/SaiMeghanath/explainable-cloud-ai-monitor.git
cd explainable-cloud-ai-monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Google Cloud credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
```

### 4. Run

```bash
python app/predict.py       # Get prediction + explanations
# or
flask run                   # Launch monitoring dashboard
```

## Why This Matters for Research

XAI is an active research area — LIME and SHAP are the dominant post-hoc methods, but both have known limitations (LIME's instability, SHAP's computational cost for large models, faithfulness vs. interpretability trade-offs). This project is a practical implementation that surfaces those trade-offs in a real deployment context: cloud APIs with black-box internals, real data, real latency constraints.

## Future Directions

- [ ] Extend to NLP models — token-level SHAP attributions for text classification predictions
- [ ] Add explanation drift monitoring (alert when feature importance patterns shift)
- [ ] Integrate with Vertex AI Explainable AI for native cloud-side attributions
- [ ] Benchmark LIME vs. SHAP faithfulness on controlled synthetic datasets

---

## Author

**Aladurthi Sai Meghanath**  
MCA (AI Specialization) · Amrita Vishwa Vidyapeetham  
[LinkedIn](https://www.linkedin.com/in/meghanath03/) · [GitHub](https://github.com/SaiMeghanath) · [saimeghanath052@gmail.com](mailto:saimeghanath052@gmail.com)
