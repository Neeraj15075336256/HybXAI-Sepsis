"""
explainability/shap_analysis.py
================================
Global feature attribution via SHAP TreeExplainer on the XGBoost branch.

Produces:
  - SHAP bar chart (Top-15 features, physics vs base colour-coded)
  - Pie chart: physics-induced vs base clinical SHAP mass
  - SHAP values CSV for downstream analysis

Usage
-----
  python explainability/shap_analysis.py \
      --model  results/xgboost_embedder.pkl \
      --data   data/processed/X_st_test_feat.npy \
      --out    results/figures/
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Feature names (43-dim static vector)
FEATURE_NAMES = [
    # Base clinical (0–22)
    'age', 'gender', 'bmi', 'charlson_cci', 'admission_type',
    'lactate', 'creatinine', 'bilirubin', 'platelet', 'wbc',
    'hemoglobin', 'glucose', 'ph', 'pao2_fio2', 'bun',
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_renal', 'sofa_neuro',
    'shock_index', 'rox_index',
    # TS summaries (23–34)
    'hr_mean', 'sbp_mean', 'spo2_mean',
    'hr_std',  'sbp_std',  'spo2_std',
    'hr_delta', 'sbp_delta', 'map_delta',
    'temp_max', 'resp_max', 'fio2_max',
    # Physics-induced (35–42)
    'news2_score', 'delta_sofa_6h', 'lactate_clearance', 'rox_extended',
    'alvarado_circ', 'sepsis_trajectory', 'pulse_pressure', 'mean_arterial_delta',
]

PHYSICS_IDX = set(range(35, 43))   # indices of physics-induced features
PHYSICS_COL = '#E05C4A'            # red for physics features
BASE_COL    = '#4A8FE0'            # blue for base clinical features


def compute_shap_values(model_path: str,
                        X: np.ndarray,
                        max_samples: int = 2000) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer on the XGBoost embedder.

    Parameters
    ----------
    model_path : path to xgboost_embedder.pkl
    X          : static feature matrix (N, 43)
    max_samples: subsample for speed (default 2000)

    Returns
    -------
    shap_values : np.ndarray (N, 43)
    """
    embedder = joblib.load(model_path)
    clf      = embedder.clf if hasattr(embedder, 'clf') else embedder

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X   = X[idx]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    return shap_values


def plot_shap_summary(shap_values: np.ndarray,
                      out_dir: str,
                      top_n: int = 15) -> None:
    """
    Plot Top-N mean |SHAP| bar chart (colour-coded by feature type)
    and a pie chart showing physics vs base clinical contribution.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)   # (43,)
    df = pd.DataFrame({'feature': FEATURE_NAMES, 'mean_shap': mean_abs})
    df['physics'] = df.index.isin(PHYSICS_IDX)
    df_top = df.nlargest(top_n, 'mean_shap').sort_values('mean_shap')

    colours = [PHYSICS_COL if p else BASE_COL for p in df_top['physics']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('HybXAI-Sepsis v3 — SHAP Global Feature Attribution\n'
                 '(XGBoost branch incl. physics-induced features)', fontsize=13, fontweight='bold')

    # Left: bar chart
    ax = axes[0]
    ax.barh(df_top['feature'], df_top['mean_shap'], color=colours)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'Top-{top_n} Features\n'
                 f'(🟥 Physics-induced | 🟦 Base clinical)')

    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=PHYSICS_COL, label='Physics-induced'),
        Patch(color=BASE_COL,    label='Base clinical'),
    ], loc='lower right', fontsize=9)

    # Right: pie chart
    phys_mass = mean_abs[list(PHYSICS_IDX)].sum()
    base_mass = mean_abs[[i for i in range(43) if i not in PHYSICS_IDX]].sum()
    axes[1].pie(
        [phys_mass, base_mass],
        labels=[f'Physics features\n({phys_mass:.2f})', f'Base clinical\n({base_mass:.2f})'],
        colors=[PHYSICS_COL, BASE_COL],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[1].set_title('SHAP Contribution:\nPhysics-Induced vs Base Clinical Features')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig6_shap_analysis.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Also save numeric values
    df_all = df.sort_values('mean_shap', ascending=False)
    df_all.to_csv(os.path.join(out_dir, '..', 'tables', 'shap_values.csv'), index=False)
    print(f"Physics SHAP mass: {phys_mass:.4f} ({phys_mass/(phys_mass+base_mass)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/xgboost_embedder.pkl')
    parser.add_argument('--data',  default='data/processed/X_st_test_feat.npy')
    parser.add_argument('--out',   default='results/figures/')
    parser.add_argument('--top_n', type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X = np.load(args.data)
    print(f"Computing SHAP values for {len(X):,} samples ...")
    sv = compute_shap_values(args.model, X)
    plot_shap_summary(sv, args.out, args.top_n)


if __name__ == '__main__':
    main()
