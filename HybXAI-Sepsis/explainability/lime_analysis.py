"""
explainability/lime_analysis.py
=================================
Patient-level explanations via LIME (Local Interpretable Model-Agnostic Explanations).
Reproduces Figure 9 of the manuscript: top-10 feature weights for the
3 highest-risk patients in the MIMIC-IV test set.

Reference: Ribeiro et al., "Why Should I Trust You?": Explaining the
Predictions of Any Classifier, KDD 2016.

Usage
-----
  python explainability/lime_analysis.py \
      --model      results/xgboost_embedder.pkl \
      --X_train    data/processed/X_st_train_aug.npy \
      --X_test     data/processed/X_st_test_feat.npy \
      --y_test     results/y_test.npy \
      --probs      results/hybxai_probs_test.npy \
      --out        results/figures/
"""

import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

SEED = 42

FEATURE_NAMES = [
    'age', 'gender', 'bmi', 'charlson_cci', 'admission_type',
    'lactate', 'creatinine', 'bilirubin', 'platelet', 'wbc',
    'hemoglobin', 'glucose', 'ph', 'pao2_fio2', 'bun',
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_renal', 'sofa_neuro',
    'shock_index', 'rox_index',
    'hr_mean', 'sbp_mean', 'spo2_mean',
    'hr_std',  'sbp_std',  'spo2_std',
    'hr_delta', 'sbp_delta', 'map_delta',
    'temp_max', 'resp_max', 'fio2_max',
    'news2_score', 'delta_sofa_6h', 'lactate_clearance', 'rox_extended',
    'alvarado_circ', 'sepsis_trajectory', 'pulse_pressure', 'mean_arterial_delta',
]

PHYSICS_NAMES = {
    'news2_score', 'delta_sofa_6h', 'lactate_clearance', 'rox_extended',
    'alvarado_circ', 'sepsis_trajectory', 'pulse_pressure', 'mean_arterial_delta',
}


def run_lime(model_path: str,
             X_train: np.ndarray,
             X_test: np.ndarray,
             y_test: np.ndarray,
             probs: np.ndarray,
             out_dir: str,
             n_patients: int = 3,
             n_features: int = 10,
             n_samples: int = 500) -> None:
    """
    Generate and plot LIME explanations for the n_patients highest-risk
    test patients.

    Parameters
    ----------
    model_path : path to xgboost_embedder.pkl
    X_train    : training static features (N_tr, 43) — used as LIME background
    X_test     : test static features (N_te, 43)
    y_test     : true labels (N_te,)
    probs      : predicted probabilities (N_te,)
    out_dir    : directory to save figure
    n_patients : number of high-risk patients to explain (default 3)
    n_features : number of LIME features per patient (default 10)
    n_samples  : perturbation samples per explanation (default 500)
    """
    os.makedirs(out_dir, exist_ok=True)

    embedder = joblib.load(model_path)
    clf = embedder.clf if hasattr(embedder, 'clf') else embedder

    # Build LIME explainer on a subset of training data
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train[:500],
        feature_names=FEATURE_NAMES,
        class_names=['Non-Sepsis', 'Sepsis'],
        mode='classification',
        random_state=SEED
    )

    # Select top-n highest-risk patients
    high_risk_idx = np.argsort(probs)[::-1][:n_patients]

    fig, axes = plt.subplots(1, n_patients, figsize=(7 * n_patients, 7))
    if n_patients == 1:
        axes = [axes]
    fig.suptitle('HybXAI-Sepsis v3 — LIME Patient-Level Explanations',
                 fontsize=14, fontweight='bold')

    for col, pat_idx in enumerate(high_risk_idx):
        prob_val = probs[pat_idx]
        true_lab = y_test[pat_idx]

        exp = lime_explainer.explain_instance(
            X_test[pat_idx],
            clf.predict_proba,
            num_features=n_features,
            num_samples=n_samples
        )

        feat_weights = exp.as_list()
        feat_names   = [fw[0] for fw in feat_weights]
        feat_vals    = [fw[1] for fw in feat_weights]
        colours      = ['#E74C3C' if v > 0 else '#3498DB' for v in feat_vals]

        ax = axes[col]
        ax.barh(range(len(feat_vals)), feat_vals, color=colours, alpha=0.85)
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels([f[:22] for f in feat_names], fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(
            f'Patient {col + 1}\nP(sepsis)={prob_val:.3f} | '
            f'True: {"Sepsis" if true_lab else "Non-Sepsis"}',
            fontsize=10
        )
        ax.set_xlabel('LIME Weight')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig9_lime_explanations.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # Save LIME weights as CSV
    import pandas as pd
    rows = []
    for col, pat_idx in enumerate(high_risk_idx):
        exp = lime_explainer.explain_instance(
            X_test[pat_idx], clf.predict_proba,
            num_features=n_features, num_samples=n_samples
        )
        for feat, weight in exp.as_list():
            rows.append({'patient': col + 1, 'patient_idx': int(pat_idx),
                         'feature': feat, 'lime_weight': weight,
                         'true_label': int(y_test[pat_idx]),
                         'pred_prob': float(probs[pat_idx])})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, '..', 'tables', 'lime_weights.csv'), index=False)
    print(f'LIME weights saved to results/tables/lime_weights.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='results/xgboost_embedder.pkl')
    parser.add_argument('--X_train', default='data/processed/X_st_train_aug.npy')
    parser.add_argument('--X_test',  default='data/processed/X_st_test_feat.npy')
    parser.add_argument('--y_test',  default='results/y_test.npy')
    parser.add_argument('--probs',   default='results/hybxai_probs_test.npy')
    parser.add_argument('--out',     default='results/figures/')
    parser.add_argument('--n_patients', type=int, default=3)
    args = parser.parse_args()

    X_train = np.load(args.X_train)
    X_test  = np.load(args.X_test)
    y_test  = np.load(args.y_test)
    probs   = np.load(args.probs)

    run_lime(args.model, X_train, X_test, y_test, probs, args.out, args.n_patients)


if __name__ == '__main__':
    main()
