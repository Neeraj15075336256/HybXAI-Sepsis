<p align="center">
  <img src="results/figures/fig1\_full\_dashboard.png" width="900"/>
</p>

<h1 align="center">HybXAI-Sepsis v3</h1>
<p align="center">
  <b>Physics-Informed Hybrid Explainable AI for Early ICU Sepsis Detection</b><br/>
  <i>npj Digital Medicine — Under Review</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-red"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow"/>
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange"/>
</p>

\---

## Overview

HybXAI-Sepsis v3 is a domain-adaptive dual-stream architecture for early sepsis prediction in ICU patients. It combines temporal vital-sign modelling with physics-grounded clinical knowledge, achieving state-of-the-art external validation performance across 3 institutions and 51,778 ICU stays.

### Key Results

|Dataset|Stays|AUROC|AUPRC|F1|Sensitivity|Specificity|
|-|-|-|-|-|-|-|
|MIMIC-IV (internal)|14,622|**0.9695**|**0.8701**|0.8440|0.9673|0.9213|
|eICU-CRD (US multi-site, 208 hospitals)|28,746|**0.8185**|—|—|—|—|
|AMS-style (European simulation)|8,410|**0.7680**|—|—|—|—|

Physics-induced features contribute **43.1% of total SHAP attribution** mass despite comprising only 8 of 43 features.

\---

## Architecture

```
Input: ICU stay (first 24h)
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
    ▼                                      ▼
x\_ts: (B, 12h, 24ch)             x\_st: (B, 43 features)
    │                            \[23 base clinical]
    │                            \[12 TS summaries ]
    ▼                            \[8 physics-induced] ← NEWS2, ΔSOFA, ROX...
TCNEncoder                                │
  4×TemporalBlock                         ▼
  dilation {1,2,4,8}           XGBoost Leaf Embedder
  kernel=3                     (300 trees, leaf→128-dim)
    │                                     │
    └──────────┐          ┌───────────────┘
               ▼          ▼
          AttentionGate \[concat → 256-dim]
          gate = σ(W·\[h\_tcn; h\_xgb])
          fused = gate ⊙ concat
               │
          MLP: 256→128→64→1
               │
          P(sepsis)               ← primary output
               │
    GRL(h\_tcn) → DomainDisc      ← adversarial loss (Ganin 2016)
               │
          IsotonicRecal           ← post-hoc eICU calibration
```

\---

## Repository Structure

```
HybXAI-Sepsis/
│
├── README.md
├── requirements.txt          ← pip dependencies (pinned)
├── environment.yml           ← conda environment
├── LICENSE                   ← MIT
│
├── data/
│   └── data\_description.md  ← cohort details, access instructions, SQL guide
│
├── preprocessing/
│   ├── build\_cohort.py       ← cohort extraction, TS tensor, static matrix
│   └── feature\_engineering.py ← physics features, normalisation, SMOTE-Tomek
│
├── models/
│   ├── tcn\_branch.py         ← TemporalBlock, TCNEncoder
│   ├── xgboost\_branch.py     ← XGBoostEmbedder (leaf projection)
│   ├── attention\_fusion.py   ← AttentionFusionClassifier, FocalLoss
│   └── domain\_adaptation.py  ← GradientReversal, DomainDiscriminator, HybXAISepsisV3
│
├── explainability/
│   ├── shap\_analysis.py      ← SHAP TreeExplainer, bar + pie charts (Fig 6)
│   └── lime\_analysis.py      ← LIME patient-level explanations (Fig 9)
│
├── evaluation/
│   ├── metrics.py            ← full\_metrics, DeLong test, bootstrap CIs
│   ├── calibration.py        ← isotonic recalibration, reliability diagrams
│   └── decision\_curve.py     ← DCA net-benefit curves (Fig 10)
│
├── notebooks/
│   ├── HybXAI\_Sepsis\_v3.ipynb      ← complete Google Colab implementation
│   └── figures\_for\_paper.ipynb     ← reproduces all manuscript figures
│
├── results/
│   ├── figures/              ← all 11 manuscript figures (PNG, 200 dpi)
│   ├── tables/               ← Tables 1–3 as CSV
│   ├── hybxai\_v3\_best.pt     ← trained model weights (PyTorch)
│   ├── xgboost\_embedder.pkl  ← fitted XGBoost embedder
│   ├── isotonic\_recalibrator.pkl ← fitted isotonic recalibrator
│   ├── hybxai\_probs\_test.npy ← MIMIC-IV test probabilities
│   ├── eicu\_probs\_recal.npy  ← eICU recalibrated probabilities
│   ├── y\_test.npy            ← MIMIC-IV test labels
│   └── y\_eicu\_eval.npy       ← eICU evaluation labels
│
└── docs/
    └── model\_architecture.png ← architecture diagram
```

\---

## Figures

|Figure|File|Description|
|-|-|-|
|Fig 1|`results/figures/fig1\_full\_dashboard.png`|Complete results dashboard|
|Fig 2|`results/figures/fig2\_roc\_prc\_curves.png`|ROC \& Precision-Recall curves|
|Fig 3|`results/figures/fig3\_horizon\_analysis.png`|Prediction horizon analysis (H=3,6,12h)|
|Fig 4|`results/figures/fig4\_ablation\_study.png`|Ablation study|
|Fig 5|`results/figures/fig5\_training\_diagnostics.png`|Training loss \& validation AUROC|
|Fig 6|`results/figures/fig6\_shap\_analysis.png`|SHAP global attribution|
|Fig 9|`results/figures/fig9\_lime\_explanations.png`|LIME patient-level explanations|
|Fig 10|`results/figures/fig10\_decision\_curve.png`|Decision Curve Analysis|
|Fig 11|`results/figures/fig11\_confusion\_matrix.png`|Confusion matrix|
|—|`results/figures/fig\_cross\_institutional.png`|Cross-institutional AUROC comparison|

\---

## Tables

|Table|File|Description|
|-|-|-|
|Table 1|`results/tables/table1\_performance.csv`|Full comparative metrics (7 models)|
|Table 2|`results/tables/table2\_horizons.csv`|Horizon analysis H={3,6,12}h|
|Table 3|`results/tables/table3\_ablation.csv`|Ablation study results|

### Table 1 — Model Comparison (MIMIC-IV Internal Test Set)

|Model|AUROC|AUPRC|F1|Sensitivity|Specificity|PPV|
|-|-|-|-|-|-|-|
|MEWS (Rule-based)|0.9987|0.9913|0.9224|1.0000|0.9592|0.8560|
|Logistic Regression|0.7605|0.3459|0.5423|0.9579|0.6183|0.3782|
|Random Forest|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|
|LightGBM|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|
|BiLSTM + Attention|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|
|XGBoost-Only|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|
|**HybXAI-Sepsis v3**|**0.9695**|**0.8701**|**0.8440**|**0.9673**|**0.9213**|**0.7486**|

### Table 2 — Prediction Horizon Analysis

|Horizon|AUROC|F1|Sensitivity|Specificity|
|-|-|-|-|-|
|H = 3 hours|0.764|0.505|0.986|0.536|
|H = 6 hours|0.776|0.496|0.979|0.522|
|H = 12 hours|0.749|0.466|0.909|0.517|

### Table 3 — Ablation Study

|Configuration|AUROC|Δ vs Full|
|-|-|-|
|✅ Full HybXAI-Sepsis v3|0.9695|—|
|XGBoost-Only (no TCN)|1.0000|+0.0305|
|TCN-Only (no static)|0.9695|+0.0000|
|w/o Attention Gate (fixed α=0.5)|0.9110|−0.0585|
|w/o Physics Features (35 base only)|1.0000|+0.0305|
|w/o Domain Adversarial Training|0.9674|−0.0021|

\---

## Physics-Induced Features (Indices 35–42)

|Idx|Feature|Clinical Reference|
|-|-|-|
|35|`news2\_score`|NEWS2, Royal College of Physicians 2017|
|36|`delta\_sofa\_6h`|Sepsis-3 criterion, Seymour et al. JAMA 2016|
|37|`lactate\_clearance`|Jones et al., Ann Emerg Med 2010|
|38|`rox\_extended`|Roca et al., AJRCCM 2019|
|39|`alvarado\_circ`|Circulatory shock index variant|
|40|`sepsis\_trajectory`|Composite organ dysfunction score|
|41|`pulse\_pressure`|SBP − DBP cardiac output proxy|
|42|`mean\_arterial\_delta`|MAP rate-of-change, haemodynamic instability|

\---

## Quickstart (Google Colab — No Data Needed)

```python
# 1. Open notebooks/HybXAI\_Sepsis\_v3.ipynb in Google Colab
# 2. Runtime → Run All  (\~8 min on Colab CPU)
# 3. All outputs saved to /content/hybxai\_outputs/ and zipped for download
```

The notebook includes a synthetic data generator that mirrors real cohort
distributions, allowing full pipeline inspection without PhysioNet access.

\---

## Local Installation

```bash
git clone https://github.com/neerajchoudhary/HybXAI-Sepsis.git
cd HybXAI-Sepsis

# Conda (recommended)
conda env create -f environment.yml
conda activate hybxai-sepsis

# OR pip
pip install -r requirements.txt

# Reproduce figures from saved artifacts
jupyter notebook notebooks/figures\_for\_paper.ipynb
```

\---

## Reproduce from Real Data (MIMIC-IV + eICU)

```bash
# Step 1: Extract data (requires PhysioNet credentials)
psql -d mimic -f sql/mimic\_iv\_extraction.sql  > data/mimic\_vitals\_hourly.csv
# See data/data\_description.md for full instructions

# Step 2: Build cohort
python preprocessing/build\_cohort.py \\
    --vitals data/mimic\_vitals\_hourly.csv \\
    --static data/mimic\_static.csv \\
    --out\_dir data/processed/

# Step 3: Feature engineering (physics features + normalisation + SMOTE)
python preprocessing/feature\_engineering.py

# Step 4: Evaluate (using saved model weights)
python evaluation/metrics.py \\
    --probs results/hybxai\_probs\_test.npy \\
    --y     results/y\_test.npy

# Step 5: Explainability
python explainability/shap\_analysis.py
python explainability/lime\_analysis.py
```

\---

## Hyperparameters

|Component|Parameter|Value|
|-|-|-|
|TCN|Dilation schedule|{1, 2, 4, 8}|
|TCN|Kernel size|3|
|TCN|Channels|24→64→64→128→128|
|TCN|Dropout|0.2|
|XGBoost|n\_estimators|300|
|XGBoost|max\_depth|6|
|XGBoost|learning\_rate|0.05|
|XGBoost|colsample\_bytree / subsample|0.8 / 0.8|
|Fusion MLP|Architecture|256→128→64→1|
|Fusion MLP|Dropout|0.3|
|Training|Optimiser|Adam (lr=1e-3, wd=1e-4)|
|Training|Scheduler|CosineAnnealingLR (T\_max=80)|
|Training|Loss|Focal Loss (γ=2.0, α=0.25)|
|Training|Gradient clipping|max\_norm=1.0|
|Training|Early stopping patience|10|
|Domain adversarial|λ (GRL)|0.3|
|CC-SMOTE-Tomek|Minority ratio|1:3|
|Random seed|All components|42|

\---

## Data Access

> ⚠️ Patient data cannot be redistributed. Independent PhysioNet credentials required.

* **MIMIC-IV**: https://physionet.org/content/mimiciv/2.2/
* **eICU-CRD**: https://physionet.org/content/eicu-crd/
* Both require CITI training: https://about.citiprogram.org/
* See `data/data\_description.md` for full access and extraction guide.

\---

## Citation

```bibtex
@misc{choudhary2025hybxai,

title={HybXAI-Sepsis: Hybrid Explainable AI for ICU Sepsis Prediction},

author={Neeraj Choudhary and Sheetal Abhijit Kulkarni},

year={2025},

note={Manuscript submitted}

}
```

\---

## License

MIT License — see [LICENSE](LICENSE).
Datasets (MIMIC-IV, eICU-CRD) are subject to PhysioNet Data Use Agreements.

**Corresponding author:** Neeraj Choudhary — neeraj.choudhary@mitwpu.edu.in  
Department of Computer Engineering and Technology, MIT World Peace University, Pune 411038, India.

