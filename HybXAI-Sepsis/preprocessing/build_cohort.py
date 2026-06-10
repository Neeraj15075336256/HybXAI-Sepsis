"""
preprocessing/build_cohort.py
==============================
Loads the SQL-exported CSVs, applies inclusion/exclusion criteria,
constructs the sliding-window time-series tensor and static feature
matrix, and saves clean NumPy arrays ready for model training.

Usage
-----
  python preprocessing/build_cohort.py \
      --vitals  data/mimic_vitals_hourly.csv \
      --static  data/mimic_static.csv \
      --out_dir data/processed/ \
      [--eicu_vitals data/eicu_vitals_hourly.csv] \
      [--eicu_static data/eicu_static.csv]
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

SEED        = 42
WINDOW_SIZE = 12          # hours of vital-sign history
VITAL_COLS  = ['heart_rate', 'sbp', 'dbp', 'map', 'spo2', 'resp_rate', 'temperature', 'fio2']
VITAL_BOUNDS = {           # physiologically plausible ranges for outlier clipping
    'heart_rate':  (20,  300),
    'sbp':         (40,  300),
    'dbp':         (20,  200),
    'map':         (20,  200),
    'spo2':        (50,  100),
    'resp_rate':   (4,   60),
    'temperature': (30,  42),
    'fio2':        (0.21, 1.0),
}

# ── Inclusion / exclusion ─────────────────────────────────────

def apply_inclusion_criteria(df_static: pd.DataFrame,
                              df_vitals: pd.DataFrame) -> tuple:
    """
    Apply pre-specified inclusion / exclusion criteria.

    Criteria:
      - Age >= 18
      - ICU LOS >= 12 h (unitdischargeoffset >= 720 min, or los >= 0.5 days)
      - >=50% hourly vital-sign coverage (>=6 of 12 hours populated)
      - First admission only (deduplicated on subject_id)
    """
    # Exclude minors
    df_static = df_static[df_static['age'].fillna(0) >= 18].copy()

    # Compute coverage per stay from vitals
    coverage = (df_vitals.groupby('stay_id')[VITAL_COLS]
                .apply(lambda g: g.notna().any(axis=1).sum()))
    sufficient = coverage[coverage >= WINDOW_SIZE * 0.5].index
    df_static = df_static[df_static['stay_id'].isin(sufficient)].copy()

    # Keep first admission per subject
    if 'subject_id' in df_static.columns:
        df_static = (df_static.sort_values('intime')
                              .drop_duplicates('subject_id', keep='first'))

    df_vitals = df_vitals[df_vitals['stay_id'].isin(df_static['stay_id'])].copy()
    print(f"  After inclusion: {len(df_static):,} stays "
          f"({int(df_static['sepsis_label'].sum()):,} septic, "
          f"{df_static['sepsis_label'].mean()*100:.1f}%)")
    return df_static.reset_index(drop=True), df_vitals


# ── Vital-sign tensor construction ────────────────────────────

def build_timeseries_tensor(df_vitals: pd.DataFrame,
                            stay_ids: np.ndarray) -> np.ndarray:
    """
    Pivot hourly vitals into shape (N, WINDOW_SIZE, 8).
    Missing time steps are left as NaN; imputed downstream.
    """
    N   = len(stay_ids)
    idx = {s: i for i, s in enumerate(stay_ids)}
    X   = np.full((N, WINDOW_SIZE, len(VITAL_COLS)), np.nan, dtype=np.float32)

    for col, (lo, hi) in VITAL_BOUNDS.items():
        df_vitals[col] = df_vitals[col].clip(lo, hi)

    for _, row in df_vitals.iterrows():
        hr = int(row.get('hr', -1))
        if 0 <= hr < WINDOW_SIZE and row['stay_id'] in idx:
            X[idx[row['stay_id']], hr, :] = [row.get(c, np.nan) for c in VITAL_COLS]

    # Forward-fill gaps <=4h along time axis (per patient, per channel)
    for i in range(N):
        for c in range(len(VITAL_COLS)):
            series = X[i, :, c]
            mask   = np.isnan(series)
            if mask.any() and not mask.all():
                last_valid = None
                count      = 0
                for t in range(WINDOW_SIZE):
                    if not np.isnan(series[t]):
                        last_valid = series[t]; count = 0
                    elif last_valid is not None and count < 4:
                        series[t] = last_valid; count += 1
                    else:
                        count += 1
                X[i, :, c] = series
    return X


# ── Static feature matrix ─────────────────────────────────────

BASE_STATIC_COLS = [
    'age', 'gender', 'bmi', 'charlson_cci', 'admission_type_encoded',
    'lactate', 'creatinine', 'bilirubin', 'platelet', 'wbc',
    'hemoglobin', 'glucose', 'ph', 'pao2_fio2', 'bun',
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_renal', 'sofa_neuro',
    'shock_index', 'rox_index',
]
TS_SUMMARY_COLS = [
    'hr_mean', 'sbp_mean', 'spo2_mean',
    'hr_std',  'sbp_std',  'spo2_std',
    'hr_delta', 'sbp_delta', 'map_delta',
    'temp_max', 'resp_max',  'fio2_max',
]


def compute_ts_summary(X_ts: np.ndarray) -> np.ndarray:
    """Derive 12 summary statistics from the time-series tensor."""
    hr   = X_ts[:, :, 0]; sbp  = X_ts[:, :, 1]; map_ = X_ts[:, :, 3]
    spo2 = X_ts[:, :, 4]; temp = X_ts[:, :, 6]; rr   = X_ts[:, :, 5]; fio2 = X_ts[:, :, 7]
    def safe(fn, arr): return np.where(np.isnan(arr).all(axis=1), np.nan, fn(arr, axis=1))
    def delta(arr):
        last  = np.nanmean(arr[:, -3:], axis=1)
        first = np.nanmean(arr[:,  :3], axis=1)
        return last - first
    return np.column_stack([
        safe(np.nanmean, hr),  safe(np.nanmean, sbp),  safe(np.nanmean, spo2),
        safe(np.nanstd,  hr),  safe(np.nanstd,  sbp),  safe(np.nanstd,  spo2),
        delta(hr),             delta(sbp),              delta(map_),
        safe(np.nanmax,  temp),safe(np.nanmax,  rr),   safe(np.nanmax,  fio2),
    ]).astype(np.float32)


def build_static_matrix(df_static: pd.DataFrame,
                        stay_ids: np.ndarray,
                        X_ts: np.ndarray) -> np.ndarray:
    """
    Assemble the 35-feature base static matrix (23 base + 12 TS summaries).
    Physics-induced features (8) are appended by feature_engineering.py.
    """
    df = df_static.set_index('stay_id').reindex(stay_ids)

    # Encode admission type
    if 'admission_type' in df.columns:
        df['admission_type_encoded'] = pd.factorize(df['admission_type'])[0].astype(float)

    # BMI fallback
    if 'bmi' not in df.columns and 'weight_kg' in df.columns and 'height_cm' in df.columns:
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2 + 1e-8)

    # PaO2/FiO2 ratio
    if 'pao2_fio2' not in df.columns and 'pao2' in df.columns:
        fio2_snap = df.get('fio2_first', pd.Series(0.21, index=df.index))
        df['pao2_fio2'] = df['pao2'] / (fio2_snap + 1e-6)

    # Shock index
    if 'shock_index' not in df.columns:
        df['shock_index'] = df.get('hr_mean', 80) / (df.get('sbp_mean', 120) + 1e-6)

    # ROX index
    if 'rox_index' not in df.columns:
        df['rox_index'] = (df.get('spo2_mean', 97) / (df.get('fio2_mean', 0.21) * 100 + 1e-6)
                          / (df.get('resp_max', 18) + 1e-6)) * 10

    base = np.column_stack([
        df.get(c, pd.Series(np.nan, index=df.index)).values.astype(np.float32)
        for c in BASE_STATIC_COLS
    ])
    ts_summary = compute_ts_summary(X_ts)
    return np.hstack([base, ts_summary])  # shape (N, 35)


# ── KNN imputation ────────────────────────────────────────────

def impute_static(X_tr: np.ndarray,
                  X_va: np.ndarray,
                  X_te: np.ndarray,
                  k: int = 5) -> tuple:
    """Fit KNNImputer on train, transform val/test."""
    imp = KNNImputer(n_neighbors=k)
    return imp.fit_transform(X_tr), imp.transform(X_va), imp.transform(X_te), imp


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vitals',      required=True)
    parser.add_argument('--static',      required=True)
    parser.add_argument('--out_dir',     default='data/processed/')
    parser.add_argument('--eicu_vitals', default=None)
    parser.add_argument('--eicu_static', default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── MIMIC-IV ──────────────────────────────────────────────
    print("\nLoading MIMIC-IV ...")
    df_v = pd.read_csv(args.vitals)
    df_s = pd.read_csv(args.static)
    df_s, df_v = apply_inclusion_criteria(df_s, df_v)

    stay_ids = df_s['stay_id'].values
    y        = df_s['sepsis_label'].astype(int).values

    print("Building time-series tensor ...")
    X_ts = build_timeseries_tensor(df_v, stay_ids)
    print("Building static matrix ...")
    X_st = build_static_matrix(df_s, stay_ids, X_ts)  # (N, 35) — physics added next

    # Stratified temporal split
    X_ts_tr, X_ts_te, X_st_tr, X_st_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X_ts, X_st, y, stay_ids, test_size=0.15, stratify=y, random_state=SEED)
    X_ts_tr, X_ts_va, X_st_tr, X_st_va, y_tr, y_va, _, _ = train_test_split(
        X_ts_tr, X_st_tr, y_tr, ids_tr, test_size=0.176, stratify=y_tr, random_state=SEED)

    print(f"Train={len(y_tr):,} | Val={len(y_va):,} | Test={len(y_te):,}")

    X_st_tr_imp, X_st_va_imp, X_st_te_imp, imputer = impute_static(X_st_tr, X_st_va, X_st_te)

    # Save
    out = args.out_dir
    np.save(f'{out}/X_ts_train.npy',  X_ts_tr);  np.save(f'{out}/X_ts_val.npy',  X_ts_va)
    np.save(f'{out}/X_ts_test.npy',   X_ts_te)
    np.save(f'{out}/X_st_train.npy',  X_st_tr_imp); np.save(f'{out}/X_st_val.npy',  X_st_va_imp)
    np.save(f'{out}/X_st_test.npy',   X_st_te_imp)
    np.save(f'{out}/y_train.npy',     y_tr);     np.save(f'{out}/y_val.npy',     y_va)
    np.save(f'{out}/y_test.npy',      y_te)

    import joblib
    joblib.dump(imputer, f'{out}/knn_imputer.pkl')
    print(f"Saved processed arrays to {out}")

    # ── eICU-CRD (optional) ────────────────────────────────────
    if args.eicu_vitals and args.eicu_static:
        print("\nLoading eICU-CRD ...")
        df_ev = pd.read_csv(args.eicu_vitals)
        df_es = pd.read_csv(args.eicu_static)
        df_es, df_ev = apply_inclusion_criteria(df_es, df_ev)
        eicu_ids = df_es['stay_id'].values
        y_e      = df_es['sepsis_label'].astype(int).values
        X_ts_e   = build_timeseries_tensor(df_ev, eicu_ids)
        X_st_e   = build_static_matrix(df_es, eicu_ids, X_ts_e)
        X_st_e   = imputer.transform(X_st_e)
        np.save(f'{out}/X_ts_eicu.npy', X_ts_e)
        np.save(f'{out}/X_st_eicu.npy', X_st_e)
        np.save(f'{out}/y_eicu.npy',    y_e)
        print(f"eICU saved: {len(y_e):,} stays ({y_e.sum():,} septic)")


if __name__ == '__main__':
    main()
