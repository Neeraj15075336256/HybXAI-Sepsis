"""
evaluation/calibration.py
===========================
Post-hoc isotonic recalibration for cross-site probability correction.
Fits on a 10% eICU hold-out; evaluates on the remaining 90%.

Reference: Zadrozny & Elkan, "Transforming Classifier Scores into
Accurate Multiclass Probability Estimates", KDD 2002.

Usage
-----
  python evaluation/calibration.py \
      --probs_cal   results/eicu_probs_recal.npy \
      --y_cal       results/y_eicu_eval.npy \
      --out         results/tables/
"""

import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss


def fit_isotonic(raw_probs: np.ndarray,
                 y_true: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression recalibrator.

    Parameters
    ----------
    raw_probs : uncalibrated model output probabilities (N,)
    y_true    : true binary labels (N,)

    Returns
    -------
    Fitted IsotonicRegression object
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs, y_true)
    print(f'  Isotonic recalibrator fitted on {len(y_true):,} samples')
    return iso


def evaluate_calibration(raw_probs: np.ndarray,
                          cal_probs: np.ndarray,
                          y_true: np.ndarray,
                          out_dir: str,
                          label: str = 'eICU') -> dict:
    """
    Compare calibration before and after isotonic recalibration.
    Produces a calibration plot and computes Brier scores.

    Parameters
    ----------
    raw_probs : uncalibrated probabilities
    cal_probs : isotonic-recalibrated probabilities
    y_true    : true binary labels
    out_dir   : output directory for figure

    Returns
    -------
    dict with calibration metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    # Reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Calibration Analysis — {label} External Validation',
                 fontsize=13, fontweight='bold')

    n_bins = 10
    for ax, probs, title, colour in [
        (axes[0], raw_probs, 'Before Recalibration (raw)', '#E74C3C'),
        (axes[1], cal_probs, 'After Isotonic Recalibration', '#2ECC71'),
    ]:
        fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.plot(mean_pred, fraction_pos, 'o-', color=colour, lw=2,
                label=title)
        ax.fill_between(mean_pred, fraction_pos,
                        np.interp(mean_pred, [0, 1], [0, 1]),
                        alpha=0.15, color=colour)
        bs = brier_score_loss(y_true, probs)
        auroc = roc_auc_score(y_true, probs)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{title}\nBrier={bs:.4f}  AUROC={auroc:.4f}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'calibration_plot.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path}')

    metrics = {
        'label':               label,
        'brier_before':        round(brier_score_loss(y_true, raw_probs), 4),
        'brier_after':         round(brier_score_loss(y_true, cal_probs), 4),
        'auroc_before':        round(roc_auc_score(y_true, raw_probs),    4),
        'auroc_after':         round(roc_auc_score(y_true, cal_probs),    4),
        'brier_improvement':   round(
            brier_score_loss(y_true, raw_probs) - brier_score_loss(y_true, cal_probs), 4),
    }

    print(f'\n  Calibration results ({label}):')
    for k, v in metrics.items():
        print(f'    {k:<25}: {v}')

    return metrics


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs_cal', required=True,
                        help='Raw eICU probabilities for calibration set (10%)')
    parser.add_argument('--y_cal',     required=True,
                        help='True labels for calibration set')
    parser.add_argument('--probs_eval', default=None,
                        help='Raw eICU probabilities for evaluation set (90%)')
    parser.add_argument('--y_eval',    default=None,
                        help='True labels for evaluation set')
    parser.add_argument('--out',       default='results/')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw_cal = np.load(args.probs_cal)
    y_cal   = np.load(args.y_cal)

    # Fit recalibrator
    iso = fit_isotonic(raw_cal, y_cal)
    joblib.dump(iso, os.path.join(args.out, 'isotonic_recalibrator_new.pkl'))
    print(f'Recalibrator saved.')

    # Evaluate if evaluation set provided
    if args.probs_eval and args.y_eval:
        raw_eval = np.load(args.probs_eval)
        y_eval   = np.load(args.y_eval)
        cal_eval = iso.predict(raw_eval)
        evaluate_calibration(raw_eval, cal_eval, y_eval,
                             os.path.join(args.out, 'figures'), 'eICU')


if __name__ == '__main__':
    main()
