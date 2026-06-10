-- ============================================================
--  HybXAI-Sepsis v3 — eICU-CRD Cohort Preprocessing
--  Source: eICU Collaborative Research Database (PhysioNet)
--  Reference preprocessing scripts:
--    https://github.com/MIT-LCP/eicu-code
--
--  This script adapts the eICU schema to match the feature
--  vector format used in MIMIC-IV extraction, enabling
--  cross-institutional validation (208 US hospitals).
--
--  Outputs:
--    eicu_vitals_hourly.csv  — time-series (8 channels × 24h)
--    eicu_static.csv         — static feature vector (23 base)
--
--  Domain shift vs MIMIC-IV (documented in paper Section 6.7):
--    - Different lab reference ranges (some sites report
--      creatinine in μmol/L instead of mg/dL)
--    - Higher SpO2 and MAP measurement variability across 208 sites
--    - Label definition heterogeneity in antibiotic timing
--    - Multi-site documentation latency (attenuated TCN signals)
-- ============================================================

-- ── Step 1: Identify ICU stays with suspected infection ───────
-- eICU uses different table structure: vitalPeriodic, nurseCharting
-- Sepsis-3 proxied by ICD diagnosis codes + antibiotic + culture

WITH sepsis_admits AS (
    SELECT DISTINCT
        p.patientunitstayid     AS stay_id,
        p.patienthealthsystemstayid AS hadm_id,
        p.hospitalid,
        p.unitadmittime24,
        p.unitdischargeoffset,
        CASE
            WHEN LOWER(d.diagnosisstring) LIKE '%sepsis%'
              OR LOWER(d.diagnosisstring) LIKE '%septic shock%'
            THEN 1 ELSE 0
        END                     AS sepsis_label
    FROM eicu_crd.patient p
    LEFT JOIN eicu_crd.diagnosis d
        ON p.patientunitstayid = d.patientunitstayid
    WHERE p.unitdischargeoffset >= 720  -- at least 12h stay (in minutes)
),

-- Aggregate sepsis label per stay (1 if any sepsis diagnosis)
sepsis_labels AS (
    SELECT
        stay_id,
        hadm_id,
        hospitalid,
        unitadmittime24,
        unitdischargeoffset,
        MAX(sepsis_label) AS sepsis_label
    FROM sepsis_admits
    GROUP BY 1,2,3,4,5
),

-- ── Step 2: Hourly vital-sign time-series ─────────────────────
-- vitalPeriodic records every 5 min; aggregate to hourly means

vitals_hourly AS (
    SELECT
        vp.patientunitstayid                              AS stay_id,
        FLOOR(vp.observationoffset / 60.0)               AS hr,
        AVG(vp.heartrate)                                AS heart_rate,
        AVG(vp.systemicsystolic)                         AS sbp,
        AVG(vp.systemicdiastolic)                        AS dbp,
        AVG(vp.systemicmean)                             AS map,
        AVG(vp.sao2)                                     AS spo2,
        AVG(vp.respiration)                              AS resp_rate,
        AVG(vp.temperature)                              AS temperature,
        -- FiO2 from nurseCharting (stored as text)
        AVG(CAST(NULLIF(REGEXP_REPLACE(
            nc.cellvaluenumeric, '[^0-9.]', ''), '') AS FLOAT) / 100.0
        )                                                AS fio2
    FROM eicu_crd.vitalperiodic vp
    -- Join FiO2 from nurse charting
    LEFT JOIN eicu_crd.nursecharting nc
        ON  nc.patientunitstayid = vp.patientunitstayid
        AND LOWER(nc.celllabel) IN ('fio2','fi02','fraction of inspired o2')
        AND ABS(nc.nursingchartoffset - vp.observationoffset) <= 30
    WHERE vp.observationoffset BETWEEN 0 AND 1439  -- first 24h (minutes)
    AND (
        vp.heartrate       BETWEEN 20 AND 300
     OR vp.systemicsystolic BETWEEN 40 AND 300
     OR vp.sao2            BETWEEN 50 AND 100
    )
    GROUP BY 1, 2
),

-- ── Step 3: First-24h laboratory values ───────────────────────
-- NOTE: Check your eICU site's creatinine units.
-- Some sites report creatinine in μmol/L (divide by 88.4 to get mg/dL).
-- This conversion is applied in Python preprocessing (Cell 4 of notebook).

labs_first24 AS (
    SELECT
        l.patientunitstayid    AS stay_id,
        AVG(CASE WHEN LOWER(l.labname) IN ('lactate','lactic acid')
            THEN l.labresult END)                          AS lactate,
        AVG(CASE WHEN LOWER(l.labname) = 'creatinine'
            THEN l.labresult END)                          AS creatinine,
        AVG(CASE WHEN LOWER(l.labname) IN ('total bilirubin','bilirubin')
            THEN l.labresult END)                          AS bilirubin,
        AVG(CASE WHEN LOWER(l.labname) = 'platelets x 1000'
            THEN l.labresult END)                          AS platelet,
        AVG(CASE WHEN LOWER(l.labname) = '-wbc x 1000'
            THEN l.labresult END)                          AS wbc,
        AVG(CASE WHEN LOWER(l.labname) IN ('hgb','hemoglobin')
            THEN l.labresult END)                          AS hemoglobin,
        AVG(CASE WHEN LOWER(l.labname) = 'glucose'
            THEN l.labresult END)                          AS glucose,
        AVG(CASE WHEN LOWER(l.labname) = 'ph'
            THEN l.labresult END)                          AS ph,
        AVG(CASE WHEN LOWER(l.labname) = 'pao2'
            THEN l.labresult END)                          AS pao2,
        AVG(CASE WHEN LOWER(l.labname) IN ('bun','urea nitrogen')
            THEN l.labresult END)                          AS bun
    FROM eicu_crd.lab l
    INNER JOIN sepsis_labels sl ON l.patientunitstayid = sl.stay_id
    WHERE l.labresultoffset BETWEEN 0 AND 1440  -- first 24h
    AND l.labresult IS NOT NULL
    AND l.labresult > 0
    GROUP BY 1
),

-- ── Step 4: Demographics ──────────────────────────────────────

demographics AS (
    SELECT
        p.patientunitstayid    AS stay_id,
        CASE
            WHEN p.age = '> 89' THEN 90
            WHEN p.age ~ '^[0-9]+$' THEN CAST(p.age AS INTEGER)
            ELSE NULL
        END                    AS age,
        CASE p.gender WHEN 'Male' THEN 1 WHEN 'Female' THEN 0 ELSE NULL END AS gender,
        p.admissionweight      AS weight_kg,
        p.admissionheight      AS height_cm,
        -- BMI computed in Python
        p.unittype             AS admission_type
    FROM eicu_crd.patient p
),

-- ── Step 5: APACHE-derived SOFA proxies ───────────────────────
-- eICU does not have a native SOFA table; use APACHE APS III
-- component scores as proxies, then remap in Python

apache_sofa_proxy AS (
    SELECT
        a.patientunitstayid    AS stay_id,
        a.intubated            AS sofa_resp_proxy,
        a.wbc                  AS sofa_coag_proxy,
        a.bilirubin            AS sofa_liver_proxy,
        a.heartrate            AS sofa_cardio_proxy,
        a.creatinine           AS sofa_renal_proxy,
        a.gcs                  AS sofa_neuro_proxy
    FROM eicu_crd.apacheapsvar a
)

-- ── Final time-series export ──────────────────────────────────
-- Save as: eicu_vitals_hourly.csv

SELECT
    sl.stay_id,
    sl.hadm_id,
    sl.hospitalid,
    sl.sepsis_label,
    vh.hr,
    COALESCE(vh.heart_rate,  NULL) AS heart_rate,
    COALESCE(vh.sbp,         NULL) AS sbp,
    COALESCE(vh.dbp,         NULL) AS dbp,
    COALESCE(vh.map,         NULL) AS map,
    COALESCE(vh.spo2,        NULL) AS spo2,
    COALESCE(vh.resp_rate,   NULL) AS resp_rate,
    COALESCE(vh.temperature, NULL) AS temperature,
    COALESCE(vh.fio2,        0.21) AS fio2
FROM sepsis_labels sl
INNER JOIN vitals_hourly vh ON sl.stay_id = vh.stay_id
WHERE vh.hr BETWEEN 0 AND 23
ORDER BY sl.stay_id, vh.hr;

-- ── Static features export ────────────────────────────────────
-- Save as: eicu_static.csv
-- Run separately by replacing final SELECT:
/*
SELECT
    sl.stay_id,
    sl.hadm_id,
    sl.hospitalid,
    sl.sepsis_label,
    d.age,
    d.gender,
    d.weight_kg,
    d.height_cm,
    d.admission_type,
    l.lactate,
    l.creatinine,   -- NOTE: check units (mg/dL vs μmol/L per hospitalid)
    l.bilirubin,
    l.platelet,
    l.wbc,
    l.hemoglobin,
    l.glucose,
    l.ph,
    l.pao2,
    l.bun,
    asp.sofa_resp_proxy,
    asp.sofa_coag_proxy,
    asp.sofa_liver_proxy,
    asp.sofa_cardio_proxy,
    asp.sofa_renal_proxy,
    asp.sofa_neuro_proxy
FROM sepsis_labels sl
LEFT JOIN demographics d       ON sl.stay_id = d.stay_id
LEFT JOIN labs_first24 l       ON sl.stay_id = l.stay_id
LEFT JOIN apache_sofa_proxy asp ON sl.stay_id = asp.stay_id
ORDER BY sl.stay_id;
*/

-- ── Post-export Python preprocessing notes ────────────────────
-- After exporting, run the following in Python (or Cell 4 of notebook):
--
-- 1. Creatinine unit correction (eICU-specific):
--    df.loc[df['creatinine'] > 20, 'creatinine'] /= 88.4
--    (Sites reporting in μmol/L have values typically > 20;
--     MIMIC-IV creatinine is always in mg/dL with max ~15)
--
-- 2. Temperature conversion (some eICU sites report in Fahrenheit):
--    df.loc[df['temperature'] > 45, 'temperature'] = \
--        (df.loc[df['temperature'] > 45, 'temperature'] - 32) * 5/9
--
-- 3. FiO2 clamping (some sites report as percentage not fraction):
--    df.loc[df['fio2'] > 1.0, 'fio2'] /= 100.0
--    df['fio2'] = df['fio2'].clip(0.21, 1.0)
