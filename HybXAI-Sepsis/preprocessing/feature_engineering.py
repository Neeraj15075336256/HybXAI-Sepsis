"""
preprocessing/feature_engineering.py
======================================
Appends the 8 physics-induced features to the 35-feature base static
matrix produced by build_cohort.py, applies normalisation, and augments
the training set with CC-SMOTE-Tomek resampling.

Physics-induced features (indices 35–42 of the final 43-dim vector):
  35  news2_score           — National Early Warning Score 2 (normalised /20)
  36  delta_sofa_6h         — Acute SOFA change in 6h window (clipped /12)
  37  lactate_clearance     — Relative lactate clearance (clipped ±2)
  38  rox_extended          — Extended ROX index (SpO₂/(FiO₂×RR))
  39  alvarado_circ         — Circulatory shock index variant (HR/SBP)
  40  sepsis_trajectory     — Composite organ dysfunction score
  41  pulse_pressure        — SBP − DBP (normalised /80)
  42  mean_arterial_delta   — MAP rate-of-change proxy

Usage
-----
  python preprocessing/feature_engineering.py \
      --data_dir data/processed/ \
      --out_dir  data/processed/
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

SEED = 42

# Feature indices in the base 35-dim static vector that correspond to:
IDX_HR   = 0   # age (placeholder; HR is in TS summaries at idx 23)
IDX_SBP  = 24  # sbp_mean (TS summary)
IDX_DBP  = 1   # gender (placeholder; DBP from TS not available in static — computed from MAP)
IDX_SPO2 = 25  # spo2_mean
IDX_RR   = 31  # resp_max
IDX_TEMP = 30  # temp_max
IDX_FIO2 = 32  # fio2_max
IDX_LACT = 5   # lactate
IDX_SOFA = slice(15, 21)  # sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_renal, sofa_neuro

# For bounded/unbounded normalisation split
BOUNDED_IDX   = list(range(5, 15))
UNBOUNDED_IDX = [i for i in range(43) if i not in BOUNDED_IDX]


def compute_physics_features(X_st: np.ndarray) -> np.ndarray:
    """
    Compute 8 physics-induced features from the 35-dim base static matrix.
    All features are scaled to roughly [-3, 3] for network stability.

    Parameters
    ----------
    X_st : np.ndarray, shape (N, 35)

    Returns
    -------
    np.ndarray, shape (N, 8)
    """
    N = X_st.shape[0]

    hr   = np.clip(X_st[:, 23], 30, 250)    # hr_mean
    sbp  = np.clip(X_st[:, 24], 50, 250)    # sbp_mean
    spo2 = np.clip(X_st[:, 25], 50, 100)    # spo2_mean
    rr   = np.clip(X_st[:, 31], 4,  60)     # resp_max
    temp = np.clip(X_st[:, 30], 30, 42)     # temp_max
    fio2 = np.clip(X_st[:, 32], 0.21, 1.0)  # fio2_max
    lact = np.clip(X_st[:, 5],  0,  20)     # lactate

    # SOFA subscores → total
    sofa = np.clip(X_st[:, 15:21], 0, 4).sum(axis=1)  # max 24
    # DBP proxy from MAP ≈ 2/3 SBP + 1/3 DBP  →  DBP ≈ 1.5*(MAP - SBP/3)
    map_ = np.clip(X_st[:, 3],  30, 160)    # map (base index 3)
    dbp  = np.clip(1.5 * (map_ - sbp / 3), 20, 150)

    # 1. NEWS2 (normalised /20)
    news2 = np.zeros(N)
    news2 += np.where(rr <= 8, 3, np.where(rr <= 11, 1, np.where(rr <= 20, 0, np.where(rr <= 24, 2, 3))))
    news2 += np.where(spo2 <= 91, 3, np.where(spo2 <= 93, 2, np.where(spo2 <= 95, 1, 0)))
    news2 += np.where(sbp <= 90, 3, np.where(sbp <= 100, 2, np.where(sbp <= 110, 1, np.where(sbp <= 219, 0, 3))))
    news2 += np.where(hr <= 40, 3, np.where(hr <= 50, 1, np.where(hr <= 90, 0, np.where(hr <= 110, 1, np.where(hr <= 130, 2, 3)))))
    news2 += np.where(temp <= 35.0, 3, np.where(temp <= 36.0, 1, np.where(temp <= 38.0, 0, np.where(temp <= 39.0, 1, 2))))
    news2 /= 20.0

    # 2. Delta-SOFA (normalised /12, clipped [-1, 1])
    # Without previous measurement, use total/24 as a severity proxy
    delta_sofa = np.clip(sofa / 12.0, 0, 2) - 1.0  # centred

    # 3. Lactate clearance proxy (no paired previous value: use absolute scaled)
    lact_clear = np.clip(-lact / 10.0, -2, 0)  # higher lactate = more negative clearance

    # 4. ROX extended (SpO₂ / (FiO₂×100) / RR), scaled
    rox_ext = np.clip((spo2 / (fio2 * 100.0 + 1e-6)) / (rr + 1e-6) / 10.0, -3, 3)

    # 5. Alvarado circulatory index (HR/SBP, shock proxy)
    alv_circ = np.clip(hr / (sbp + 1e-6), 0, 3)

    # 6. Composite sepsis trajectory
    sep_traj = np.clip(
        (lact / 10.0
         + np.maximum(0, 70 - map_) / 20.0
         + np.maximum(0, rr - 20)   / 15.0
         + np.maximum(0, 0.21 - fio2) / 0.2) / 4.0,
        0, 2
    )

    # 7. Pulse pressure (SBP − DBP, normalised /80)
    pp = np.clip((sbp - dbp) / 80.0, 0, 3)

    # 8. MAP rate-of-change proxy (deviation from target 65 mmHg)
    map_delta = np.clip((65.0 - map_) / 30.0, -2, 2)

    return np.column_stack([
        news2, delta_sofa, lact_clear, rox_ext,
        alv_circ, sep_traj, pp, map_delta
    ]).astype(np.float32)


def append_physics_features(X_st_35: np.ndarray) -> np.ndarray:
    """Append 8 physics-induced features → (N, 43)."""
    phys = compute_physics_features(X_st_35)
    return np.hstack([X_st_35, phys])


def normalise(X_tr: np.ndarray,
              X_va: np.ndarray,
              X_te: np.ndarray) -> tuple:
    """
    Normalise the 43-dim static vector:
      - Indices 5–14 (lab values, bounded): MinMaxScaler [0, 1]
      - All other indices (unbounded): StandardScaler
    Fitted on training set only.
    """
    scaler_mm  = MinMaxScaler()
    scaler_std = StandardScaler()
    scaler_mm.fit(X_tr[:, BOUNDED_IDX])
    scaler_std.fit(X_tr[:, UNBOUNDED_IDX])

    def _apply(X):
        Xn = X.copy()
        Xn[:, BOUNDED_IDX]   = scaler_mm.transform(X[:, BOUNDED_IDX])
        Xn[:, UNBOUNDED_IDX] = scaler_std.transform(X[:, UNBOUNDED_IDX])
        return Xn

    return _apply(X_tr), _apply(X_va), _apply(X_te), scaler_mm, scaler_std


def normalise_timeseries(X_ts: np.ndarray) -> np.ndarray:
    """
    Patient-level z-score normalisation + delta + coefficient-of-variation
    augmentation of the 8-channel vital-sign tensor.

    Input:  (N, T, 8)
    Output: (N, T, 24)  [original + delta + CV]
    """
    mu  = np.nanmean(X_ts, axis=1, keepdims=True)
    sig = np.nanstd(X_ts,  axis=1, keepdims=True) + 1e-8
    Xz  = (X_ts - mu) / sig

    delta = np.diff(Xz, axis=1, prepend=Xz[:, :1, :])
    cv    = np.broadcast_to(
                np.nanstd(X_ts, axis=1, keepdims=True)
                / (np.abs(np.nanmean(X_ts, axis=1, keepdims=True)) + 1e-8),
                X_ts.shape
            ).copy()

    return np.concatenate([np.nan_to_num(Xz), np.nan_to_num(delta), np.nan_to_num(cv)], axis=2).astype(np.float32)


def smote_tomek(X_ts_flat: np.ndarray,
                X_st: np.ndarray,
                y: np.ndarray) -> tuple:
    """
    Apply CC-SMOTE-Tomek to the concatenated (flattened TS + static) training array.
    Minority:majority ratio = 1:3.
    """
    X_comb = np.hstack([X_ts_flat.reshape(len(y), -1), X_st])
    smote  = SMOTE(sampling_strategy=0.33, k_neighbors=5, random_state=SEED)
    smt    = SMOTETomek(smote=smote, random_state=SEED)
    X_aug, y_aug = smt.fit_resample(X_comb, y)
    ts_dim = X_ts_flat.shape[1] * X_ts_flat.shape[2]
    X_ts_aug = X_aug[:, :ts_dim].reshape(-1, X_ts_flat.shape[1], X_ts_flat.shape[2])
    X_st_aug = X_aug[:, ts_dim:]
    print(f"  SMOTE-Tomek: {len(y)} → {len(y_aug)} samples "
          f"(septic: {y.sum()} → {y_aug.sum()})")
    return X_ts_aug, X_st_aug, y_aug


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed/')
    parser.add_argument('--out_dir',  default='data/processed/')
    args = parser.parse_args()

    d = args.data_dir
    o = args.out_dir
    os.makedirs(o, exist_ok=True)

    print("Loading base arrays ...")
    X_ts_tr = np.load(f'{d}/X_ts_train.npy')
    X_ts_va = np.load(f'{d}/X_ts_val.npy')
    X_ts_te = np.load(f'{d}/X_ts_test.npy')
    X_st_tr = np.load(f'{d}/X_st_train.npy')
    X_st_va = np.load(f'{d}/X_st_val.npy')
    X_st_te = np.load(f'{d}/X_st_test.npy')
    y_tr    = np.load(f'{d}/y_train.npy')
    y_va    = np.load(f'{d}/y_val.npy')
    y_te    = np.load(f'{d}/y_test.npy')

    # Append physics features
    print("Appending physics-induced features (35 → 43 dims) ...")
    X_st_tr = append_physics_features(X_st_tr)
    X_st_va = append_physics_features(X_st_va)
    X_st_te = append_physics_features(X_st_te)

    # Normalise static features
    print("Normalising static features ...")
    X_st_tr_n, X_st_va_n, X_st_te_n, sc_mm, sc_std = normalise(X_st_tr, X_st_va, X_st_te)

    # Normalise time-series (→ 24 channels per time step)
    print("Normalising time-series (8 → 24 channels) ...")
    X_ts_tr_f = normalise_timeseries(X_ts_tr)
    X_ts_va_f = normalise_timeseries(X_ts_va)
    X_ts_te_f = normalise_timeseries(X_ts_te)

    # SMOTE-Tomek augmentation on training set
    print("Applying CC-SMOTE-Tomek ...")
    X_ts_tr_aug, X_st_tr_aug, y_tr_aug = smote_tomek(X_ts_tr_f, X_st_tr_n, y_tr)

    # Save
    np.save(f'{o}/X_ts_train_aug.npy',  X_ts_tr_aug)
    np.save(f'{o}/X_st_train_aug.npy',  X_st_tr_aug)
    np.save(f'{o}/y_train_aug.npy',     y_tr_aug)
    np.save(f'{o}/X_ts_val_feat.npy',   X_ts_va_f)
    np.save(f'{o}/X_st_val_feat.npy',   X_st_va_n)
    np.save(f'{o}/X_ts_test_feat.npy',  X_ts_te_f)
    np.save(f'{o}/X_st_test_feat.npy',  X_st_te_n)
    joblib.dump(sc_mm,  f'{o}/scaler_minmax.pkl')
    joblib.dump(sc_std, f'{o}/scaler_standard.pkl')

    # eICU if present
    if os.path.exists(f'{d}/X_ts_eicu.npy'):
        print("Processing eICU features ...")
        X_ts_e = np.load(f'{d}/X_ts_eicu.npy')
        X_st_e = np.load(f'{d}/X_st_eicu.npy')
        X_st_e = append_physics_features(X_st_e)
        X_st_e_n = X_st_e.copy()
        X_st_e_n[:, BOUNDED_IDX]   = sc_mm.transform(X_st_e[:, BOUNDED_IDX])
        X_st_e_n[:, UNBOUNDED_IDX] = sc_std.transform(X_st_e[:, UNBOUNDED_IDX])
        X_ts_e_f = normalise_timeseries(X_ts_e)
        np.save(f'{o}/X_ts_eicu_feat.npy', X_ts_e_f)
        np.save(f'{o}/X_st_eicu_feat.npy', X_st_e_n)

    print("Feature engineering complete.")


if __name__ == '__main__':
    main()
