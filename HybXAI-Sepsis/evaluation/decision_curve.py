"""
evaluation/decision_curve.py
==============================
Decision Curve Analysis (DCA) for clinical utility evaluation.
Reproduces Figure 10 of the manuscript.

Net benefit = TP/N − FP/N × pt/(1−pt)
where pt = threshold probability, N = total sample size.

Reference: Vickers & Elkin, Medical Decision Making 2006.

Usage
-----
  python evaluation/decision_curve.py \
      --probs  results/hybxai_probs_test.npy \
      --y      results/y_test.npy \
      --out    results/figures/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


PALETTE = {
    'MEWS (Rule-based)':           '#666666',
    'Logistic Regression':         '#95A5A6',
    'Random Forest':               '#3498DB',
    'LightGBM':                    '#9B59B6',
    'BiLSTM + Attention':          '#E67E22',
    'XGBoost-Only':                '#1ABC9C',
    'HybXAI-Sepsis v3 (Proposed)': '#E74C3C',
}


def compute_net_benefit(y_true: np.ndarray,
                        probs: np.ndarray,
                        thresholds: np.ndarray) -> np.ndarray:
    """
    Compute net benefit curve over a range of threshold probabilities.

    Parameters
    ----------
    y_true     : true binary labels (N,)
    probs      : predicted probabilities (N,)
    thresholds : array of decision thresholds to evaluate

    Returns
    -------
    net_benefit : np.ndarray, shape (len(thresholds),)
    """
    n = len(y_true)
    nb = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp   = int(((pred == 1) & (y_true == 1)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        nb.append(tp / n - fp / n * t / (1 - t + 1e-8))
    return np.array(nb)


def treat_all_benefit(y_true: np.ndarray,
                      thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of treating all patients."""
    prev = y_true.mean()
    return prev - (1 - prev) * thresholds / (1 - thresholds + 1e-8)


def plot_dca(probs_dict: dict,
             y_true: np.ndarray,
             out_dir: str,
             thresholds: np.ndarray = None,
             clinical_threshold: float = 0.10) -> pd.DataFrame:
    """
    Plot Decision Curve Analysis for multiple models.

    Parameters
    ----------
    probs_dict : {model_name: probability_array}
    y_true     : true binary labels
    out_dir    : output directory
    thresholds : evaluation thresholds (default 0.02–0.50, 100 steps)
    clinical_threshold : mark this threshold with a vertical line (default 0.10)

    Returns
    -------
    DataFrame of net benefits at clinical_threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.50, 100)

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    ta = treat_all_benefit(y_true, thresholds)
    results = {}

    for name, probs in probs_dict.items():
        nb = compute_net_benefit(y_true, probs, thresholds)
        results[name] = nb
        colour = PALETTE.get(name, '#333333')
        lw     = 2.5 if 'Proposed' in name else 1.5
        ax.plot(thresholds, nb, color=colour, lw=lw, label=name)

    ax.plot(thresholds, ta, 'g--', lw=1.5, alpha=0.6, label='Treat All')
    ax.axhline(0, color='k', lw=0.8, alpha=0.4, linestyle=':')
    ax.axvline(clinical_threshold, color='gray', ls='--', alpha=0.5,
               label=f'Clinical threshold ({clinical_threshold:.2f})')

    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(-0.05, 0.30)
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision Curve Analysis — Clinical Utility', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig10_decision_curve.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # Net benefit at clinical threshold
    ct_idx = int(np.argmin(np.abs(thresholds - clinical_threshold)))
    summary = [{
        'Model':          name,
        f'NB_at_{clinical_threshold}': round(float(results[name][ct_idx]), 4),
        'AUROC':          round(roc_auc_score(y_true, probs), 4),
    } for name, probs in probs_dict.items()]
    df = pd.DataFrame(summary).sort_values(f'NB_at_{clinical_threshold}', ascending=False)

    print(f'\nNet benefit at threshold = {clinical_threshold}:')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, '..', 'tables', 'dca_results.csv'), index=False)
    return df


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', required=True,
                        help='Path to .npy predicted probabilities (HybXAI-Sepsis v3)')
    parser.add_argument('--y',     required=True,
                        help='Path to .npy true labels')
    parser.add_argument('--out',   default='results/figures/')
    args = parser.parse_args()

    probs  = np.load(args.probs)
    y_true = np.load(args.y)

    plot_dca(
        {'HybXAI-Sepsis v3 (Proposed)': probs},
        y_true,
        args.out
    )


if __name__ == '__main__':
    main()
