# Data Description

> ⚠️ **No patient data is stored in this repository.** All three datasets require independent
> access through PhysioNet (https://physionet.org). This file describes each dataset,
> the cohort inclusion criteria, and how to reproduce the exact data extracts used in the paper.

---

## Datasets Used

### 1. MIMIC-IV v2.2 (Primary / Internal Validation)
| Property | Value |
|----------|-------|
| Source | Beth Israel Deaconess Medical Center, Boston MA, USA |
| Period | 2008 – 2019 |
| Access | https://physionet.org/content/mimiciv/2.2/ |
| Licence | PhysioNet Credentialed Health Data Licence 1.5.0 |
| Cohort size (after inclusion) | **14,622 ICU stays** |
| Sepsis-positive | 2,850 (19.5%) |
| Split | 70 / 15 / 15 train / val / test (stratified, temporally ordered) |
| Test set used in this repo | **2,194 stays** (`results/y_test.npy`, `results/hybxai_probs_test.npy`) |

### 2. eICU-CRD (External Validation — US Multi-site)
| Property | Value |
|----------|-------|
| Source | 208 US ICUs (Philips eICU programme) |
| Period | 2014 – 2015 |
| Access | https://physionet.org/content/eicu-crd/ |
| Licence | PhysioNet Credentialed Health Data Licence 1.5.0 |
| Cohort size (after inclusion) | **28,746 ICU stays** |
| Sepsis-positive | 5,490 (19.1%) |
| Used for calibration (10%) | 2,874 stays → isotonic recalibration |
| Used for evaluation (90%) | **25,872 stays** (`results/y_eicu_eval.npy`, `results/eicu_probs_recal.npy`) |

### 3. AmsterdamUMCdb-style Simulation (External Validation — European)
| Property | Value |
|----------|-------|
| Source | Synthetic cohort generated from published AmsterdamUMCdb summary statistics |
| Reference | Thoral et al., Crit Care Med 2021; Johnson et al., Sci Data 2023 |
| No PhysioNet access required | Generated in notebook Cell 3 |
| Cohort size | **8,410 ICU stays** |
| Sepsis-positive | ~19% |

---

## Inclusion / Exclusion Criteria

**Included:**
- Adult ICU admissions (age ≥ 18 years)
- ICU stay length ≥ 12 hours
- ≥ 50% hourly vital-sign coverage over the first 24-hour observation window
- First ICU admission per hospitalisation only

**Excluded:**
- Age < 18 years
- Re-admissions within 48 hours of a prior ICU stay
- Stays with < 4 of 8 vital-sign channels populated at any time step
- Stays where Sepsis-3 label could not be algorithmically assigned (missing culture or antibiotic data)

---

## Sepsis-3 Label Definition

Labels were derived algorithmically following Seymour et al. (JAMA 2016):

1. **Suspected infection** = antibiotic administration AND blood culture order within 24 hours of each other
2. **Organ dysfunction** = acute increase in total SOFA score ≥ 2 points from baseline within 24 hours of suspicion time
3. **Label timestamp** = `suspicion_time` = earlier of antibiotic start or culture order time
4. **Leakage prevention**: `suspicion_time` is never used as a model input feature

---

## Feature Vector

### Time-series branch (TCN) — 8 channels × 12 hours
| Channel | MIMIC-IV itemids | eICU field |
|---------|-----------------|------------|
| Heart rate | 220045 | `heartrate` |
| Systolic BP | 220179, 220050 | `systemicsystolic` |
| Diastolic BP | 220180, 220051 | `systemicdiastolic` |
| MAP | 220052, 220181, 225312 | `systemicmean` |
| SpO₂ | 220277 | `sao2` |
| Respiratory rate | 220210, 224690 | `respiration` |
| Temperature (°C) | 223762, 220049 | `temperature` |
| FiO₂ | 223835 | nursecharting: `fio2` |

### Static branch (XGBoost) — 43 features
- **Indices 0–22**: 23 base clinical features (demographics, labs, SOFA subscores, shock index, ROX index)
- **Indices 23–34**: 12 time-series summary statistics (mean, std, delta, max per selected channels)
- **Indices 35–42**: 8 physics-induced features (NEWS2, Δ-SOFA₆h, lactate clearance, ROX extended, Alvarado circulatory, sepsis trajectory, pulse pressure, MAP rate-of-change)

---

## How to Obtain the Data

### MIMIC-IV
```bash
# 1. Complete CITI training: https://about.citiprogram.org/
# 2. Register at PhysioNet and apply for MIMIC-IV access
# 3. Download:
wget -r -N -c -np \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimiciv/2.2/
```

### eICU-CRD
```bash
wget -r -N -c -np \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/eicu-crd/2.0/
```

### Running the SQL extractions
```bash
# After loading MIMIC-IV into PostgreSQL or BigQuery:
psql -d mimic -f sql/mimic_iv_extraction.sql -o data/mimic_vitals_hourly.csv
psql -d mimic -f sql/mimic_static_extraction.sql -o data/mimic_static.csv

psql -d eicu  -f sql/eicu_preprocessing.sql -o data/eicu_vitals_hourly.csv
psql -d eicu  -f sql/eicu_static_extraction.sql -o data/eicu_static.csv
```

See `sql/` directory for full extraction queries.
