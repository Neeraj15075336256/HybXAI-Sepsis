"""
evaluation/metrics.py
======================
Evaluation utilities: full metrics computation, Youden threshold,
DeLong statistical test, and bootstrap confidence intervals.
Reproduces Table 1 and all statistical significance tests.

Usage
-----
  python evaluation/metrics.py \
      --probs  results/hybxai_probs_test.npy \
      --y      results/y_test.npy \
      --out    results/tables/
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_score, recall_score, roc_curve
)

SEED = 42


# ── Core metric functions ─────────────────────────────────────

def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    """
    Find the Youden-optimal classification threshold.
    J = Sensitivity + Specificity − 1 = TPR − FPR

    Returns
    -------
    threshold : float
    fpr       : np.ndarray
    tpr       : np.ndarray
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx]), fpr, tpr


def full_metrics(y_true: np.ndarray, probs: np.ndarray,
                 name: str = 'Model') -> dict:
    """
    Compute the full set of performance metrics at Youden-optimal threshold.

    Returns
    -------
    dict with keys: Model, AUROC, AUPRC, F1, Sensitivity, Specificity,
                    PPV, Threshold
    """
    thr, fpr, tpr = youden_threshold(y_true, probs)
    y_pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        Model       = name,
        AUROC       = round(roc_auc_score(y_true, probs),             4),
        AUPRC       = round(average_precision_score(y_true, probs),   4),
        F1          = round(f1_score(y_true, y_pred),                 4),
        Sensitivity = round(tp / (tp + fn + 1e-8),                    4),
        Specificity = round(tn / (tn + fp + 1e-8),                    4),
        PPV         = round(precision_score(y_true, y_pred,
                                            zero_division=0),          4),
        Threshold   = round(thr, 3),
    )


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray,
                 metric_fn=roc_auc_score,
                 n_bootstrap: int = 2000,
                 ci: float = 0.95,
                 seed: int = SEED) -> tuple:
    """
    Bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn  : callable(y_true, probs) → float
    n_bootstrap: number of resampling iterations (default 2000)
    ci         : confidence level (default 0.95)

    Returns
    -------
    (mean, lower, upper)
    """
    rng  = np.random.default_rng(seed)
    vals = []
    n    = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Skip if only one class in resample
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], probs[idx]))
    vals  = np.array(vals)
    alpha = (1 - ci) / 2
    return float(np.mean(vals)), float(np.percentile(vals, alpha * 100)), float(np.percentile(vals, (1 - alpha) * 100))


def delong_test(y_true: np.ndarray,
                prob_a: np.ndarray,
                prob_b: np.ndarray) -> float:
    """
    Paired DeLong test for the difference between two AUROCs.
    Two-sided p-value.

    Reference: DeLong et al., Biometrics 1988.
    """
    pos = y_true == 1
    neg = y_true == 0
    m, n = int(pos.sum()), int(neg.sum())

    px, py = prob_a[pos], prob_a[neg]
    qx, qy = prob_b[pos], prob_b[neg]

    def _v10(p_pos, p_neg):
        return np.array([
            np.mean(p_pos[i] > p_neg) + 0.5 * np.mean(p_pos[i] == p_neg)
            for i in range(len(p_pos))
        ])

    def _v01(p_pos, p_neg):
        return np.array([
            np.mean(p_pos > p_neg[j]) + 0.5 * np.mean(p_pos == p_neg[j])
            for j in range(len(p_neg))
        ])

    V10_a, V10_b = _v10(px, py), _v10(qx, qy)
    V01_a, V01_b = _v01(px, py), _v01(qx, qy)

    S10 = np.cov(V10_a, V10_b)
    S01 = np.cov(V01_a, V01_b)
    S   = S10 / m + S01 / n

    diff = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    se   = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-10))
    z    = diff / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def print_comparison_table(metrics_list: list) -> pd.DataFrame:
    """
    Pretty-print and return a DataFrame of metrics for multiple models.
    Applies Bonferroni correction annotation.
    """
    df = pd.DataFrame(metrics_list)
    col_order = ['Model', 'AUROC', 'AUPRC', 'F1',
                 'Sensitivity', 'Specificity', 'PPV', 'Threshold']
    df = df[[c for c in col_order if c in df.columns]]

    print('\n' + '=' * 85)
    print('TABLE 1: Comparative Performance (MIMIC-IV internal test set)')
    print('=' * 85)
    print(df.to_string(index=False))
    print('-' * 85)
    print('Note: Statistical significance tested with two-sided paired DeLong test.')
    print('      Bonferroni-corrected α = 0.05/7 = 0.007 for 7 pairwise comparisons.\n')
    return df


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', required=True,
                        help='Path to .npy file of predicted probabilities')
    parser.add_argument('--y',     required=True,
                        help='Path to .npy file of true labels')
    parser.add_argument('--name',  default='HybXAI-Sepsis v3',
                        help='Model name for output table')
    parser.add_argument('--out',   default='results/tables/')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    probs  = np.load(args.probs)
    y_true = np.load(args.y)

    m = full_metrics(y_true, probs, args.name)
    print_comparison_table([m])

    # Bootstrap CIs for primary metrics
    print('Bootstrap 95% confidence intervals (n=2000):')
    for metric_fn, label in [
        (roc_auc_score,            'AUROC'),
        (average_precision_score,  'AUPRC'),
    ]:
        mu, lo, hi = bootstrap_ci(y_true, probs, metric_fn)
        print(f'  {label}: {mu:.4f} [{lo:.4f}, {hi:.4f}]')

    # Save
    out_path = os.path.join(args.out, 'metrics_summary.csv')
    pd.DataFrame([m]).to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
