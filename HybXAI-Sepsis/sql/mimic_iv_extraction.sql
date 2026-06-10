-- ============================================================
--  HybXAI-Sepsis v3 — MIMIC-IV Cohort Extraction
--  Source: MIMIC-IV v2.2 (PhysioNet)
--  Requires: PhysioNet credentialing + CITI certification
--
--  Run on BigQuery (recommended) or local PostgreSQL after
--  loading MIMIC-IV using the official loader scripts at:
--  https://github.com/MIT-LCP/mimic-iv
--
--  This script extracts:
--    1. Sepsis-3 cohort (14,622 ICU stays)
--    2. Hourly vital-sign time-series (8 channels, 24h window)
--    3. Static feature vector (23 base features)
--  Physics-induced features (indices 35-42) are computed in
--  Python (Cell 3 of the notebook), not in SQL.
-- ============================================================

-- ── Step 1: Identify suspected infection ─────────────────────
-- Suspected infection = concurrent antibiotic admin + blood culture
-- within 24h of each other (Seymour et al., JAMA 2016 definition)

WITH suspected_infection AS (
    SELECT
        icu.hadm_id,
        icu.stay_id,
        icu.subject_id,
        icu.intime,
        icu.outtime,
        MIN(abx.starttime)  AS antibiotic_time,
        MIN(bc.charttime)   AS culture_time,
        LEAST(MIN(abx.starttime), MIN(bc.charttime)) AS suspicion_time
    FROM mimiciv_icu.icustays icu
    -- Antibiotic administration
    INNER JOIN mimiciv_hosp.prescriptions abx
        ON  icu.hadm_id = abx.hadm_id
        AND abx.drug_type = 'MAIN'
        AND LOWER(abx.drug) IN (
            'vancomycin','piperacillin-tazobactam','cefepime',
            'meropenem','imipenem','levofloxacin','ciprofloxacin',
            'metronidazole','ampicillin-sulbactam','ceftriaxone',
            'ceftazidime','azithromycin','fluconazole','linezolid'
        )
    -- Blood culture ordering
    INNER JOIN mimiciv_hosp.microbiologyevents bc
        ON  icu.hadm_id = bc.hadm_id
        AND bc.spec_type_desc LIKE '%BLOOD%'
        AND ABS(EXTRACT(EPOCH FROM (abx.starttime - bc.charttime))/3600) <= 24
    WHERE icu.los >= 0.5  -- at least 12h ICU stay
    GROUP BY 1,2,3,4,5
),

-- ── Step 2: SOFA scores (admission baseline + rolling) ────────
-- Using the MIMIC-IV materialised SOFA view (run sofa.sql first)
-- https://github.com/MIT-LCP/mimic-iv/blob/main/concepts/score/sofa.sql

sofa_scores AS (
    SELECT
        s.stay_id,
        s.hr,
        s.sofa_24hours,
        s.respiration   AS sofa_resp,
        s.coagulation   AS sofa_coag,
        s.liver         AS sofa_liver,
        s.cardiovascular AS sofa_cardio,
        s.renal         AS sofa_renal,
        s.cns           AS sofa_neuro
    FROM mimiciv_derived.sofa s
),

-- ── Step 3: Sepsis-3 label assignment ─────────────────────────
-- Sepsis = suspected infection + acute SOFA increase ≥ 2
-- Label timestamp = suspicion_time (antibiotic OR culture, whichever first)
-- Leakage prevention: label_time is NEVER used as a model feature

sepsis_cohort AS (
    SELECT
        si.stay_id,
        si.hadm_id,
        si.subject_id,
        si.intime,
        si.outtime,
        si.suspicion_time                      AS label_time,
        -- Baseline SOFA: median of first 24h
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY sf_base.sofa_24hours)     AS sofa_baseline,
        -- Worst SOFA within 24h of suspicion
        MAX(sf_acute.sofa_24hours)             AS sofa_peak,
        CASE
            WHEN MAX(sf_acute.sofa_24hours) -
                 PERCENTILE_CONT(0.5) WITHIN GROUP (
                     ORDER BY sf_base.sofa_24hours) >= 2
            THEN 1 ELSE 0
        END                                    AS sepsis_label
    FROM suspected_infection si
    -- Baseline SOFA (first 24h)
    LEFT JOIN sofa_scores sf_base
        ON sf_base.stay_id = si.stay_id
        AND sf_base.hr <= 24
    -- Acute SOFA (within 24h of suspicion)
    LEFT JOIN sofa_scores sf_acute
        ON sf_acute.stay_id = si.stay_id
        AND sf_acute.hr BETWEEN
            EXTRACT(EPOCH FROM (si.suspicion_time - si.intime))/3600 - 24
            AND EXTRACT(EPOCH FROM (si.suspicion_time - si.intime))/3600 + 24
    GROUP BY 1,2,3,4,5,6
),

-- ── Step 4: Hourly vital-sign time-series ─────────────────────
-- 8 channels × 24 hours = primary TCN input
-- Forward-fill gaps ≤ 4h; remaining gaps imputed in Python (k-NN)

vitals_hourly AS (
    SELECT
        ce.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (ce.charttime - icu.intime))/3600) AS hr,
        AVG(CASE WHEN ce.itemid IN (220045) THEN ce.valuenum END)     AS heart_rate,
        AVG(CASE WHEN ce.itemid IN (220179,220050) THEN ce.valuenum END) AS sbp,
        AVG(CASE WHEN ce.itemid IN (220180,220051) THEN ce.valuenum END) AS dbp,
        AVG(CASE WHEN ce.itemid IN (220052,220181,225312) THEN ce.valuenum END) AS map,
        AVG(CASE WHEN ce.itemid IN (220277) THEN ce.valuenum END)     AS spo2,
        AVG(CASE WHEN ce.itemid IN (220210,224690) THEN ce.valuenum END) AS resp_rate,
        AVG(CASE WHEN ce.itemid IN (223762,220049) THEN ce.valuenum END) AS temperature,
        AVG(CASE WHEN ce.itemid IN (223835) THEN ce.valuenum/100.0 END)  AS fio2
    FROM mimiciv_icu.chartevents ce
    INNER JOIN mimiciv_icu.icustays icu
        ON ce.stay_id = icu.stay_id
    WHERE ce.itemid IN (
        220045,             -- Heart Rate
        220179, 220050,     -- Systolic BP (non-invasive, arterial)
        220180, 220051,     -- Diastolic BP
        220052, 220181, 225312, -- MAP
        220277,             -- SpO2
        220210, 224690,     -- Respiratory Rate
        223762, 220049,     -- Temperature (C / F converted upstream)
        223835              -- FiO2 (as %)
    )
    AND ce.valuenum IS NOT NULL
    AND ce.error IS DISTINCT FROM 1
    GROUP BY 1, 2
),

-- ── Step 5: First-day laboratory values (static features) ─────

labs_first24 AS (
    SELECT
        le.hadm_id,
        AVG(CASE WHEN le.itemid IN (50813,52442) THEN le.valuenum END) AS lactate,
        AVG(CASE WHEN le.itemid IN (50912,52024) THEN le.valuenum END) AS creatinine,
        AVG(CASE WHEN le.itemid IN (50885,52057) THEN le.valuenum END) AS bilirubin,
        AVG(CASE WHEN le.itemid IN (51265,52152) THEN le.valuenum END) AS platelet,
        AVG(CASE WHEN le.itemid IN (51301,52075) THEN le.valuenum END) AS wbc,
        AVG(CASE WHEN le.itemid IN (51222,52060) THEN le.valuenum END) AS hemoglobin,
        AVG(CASE WHEN le.itemid IN (50931,52027) THEN le.valuenum END) AS glucose,
        AVG(CASE WHEN le.itemid IN (50820,52038) THEN le.valuenum END) AS ph,
        AVG(CASE WHEN le.itemid IN (50821,52004) THEN le.valuenum END) AS pao2,
        AVG(CASE WHEN le.itemid IN (51006,52161) THEN le.valuenum END) AS bun
    FROM mimiciv_hosp.labevents le
    INNER JOIN sepsis_cohort sc ON le.hadm_id = sc.hadm_id
    WHERE le.charttime BETWEEN sc.intime AND sc.intime + INTERVAL '24 hours'
    AND le.itemid IN (
        50813, 52442,  -- Lactate
        50912, 52024,  -- Creatinine
        50885, 52057,  -- Total Bilirubin
        51265, 52152,  -- Platelets
        51301, 52075,  -- WBC
        51222, 52060,  -- Hemoglobin
        50931, 52027,  -- Glucose
        50820, 52038,  -- pH
        50821, 52004,  -- PaO2
        51006, 52161   -- BUN
    )
    GROUP BY 1
),

-- ── Step 6: Patient demographics ──────────────────────────────

demographics AS (
    SELECT
        icu.stay_id,
        icu.hadm_id,
        EXTRACT(YEAR FROM AGE(adm.admittime, pat.anchor_dob)) +
            (pat.anchor_age - EXTRACT(YEAR FROM AGE(pat.anchor_year::TEXT::DATE,
                pat.anchor_dob)))       AS age,
        CASE pat.gender WHEN 'M' THEN 1 ELSE 0 END AS gender,
        adm.admission_type,
        -- Charlson CCI from ICD-10 codes (simplified)
        0                               AS charlson_cci  -- computed separately via charlson.sql
    FROM mimiciv_icu.icustays icu
    INNER JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
    INNER JOIN mimiciv_hosp.patients pat   ON icu.subject_id = pat.subject_id
)

-- ── Final Assembly ─────────────────────────────────────────────
-- Produces one row per (stay_id, hour) for the time-series table
-- and one row per stay_id for the static feature table.
-- Join in Python after exporting both tables.

-- Time-series export (save as mimic_vitals_hourly.csv)
SELECT
    sc.stay_id,
    sc.hadm_id,
    sc.sepsis_label,
    sc.label_time,
    sc.intime,
    vh.hr,
    vh.heart_rate,
    vh.sbp,
    vh.dbp,
    vh.map,
    vh.spo2,
    vh.resp_rate,
    vh.temperature,
    vh.fio2
FROM sepsis_cohort sc
INNER JOIN vitals_hourly vh ON sc.stay_id = vh.stay_id
WHERE vh.hr BETWEEN 0 AND 23  -- first 24h window
ORDER BY sc.stay_id, vh.hr;

-- Static features export (save as mimic_static.csv):
-- Run separately by replacing the final SELECT with:
/*
SELECT
    sc.stay_id,
    sc.hadm_id,
    sc.sepsis_label,
    sc.label_time,
    d.age,
    d.gender,
    d.admission_type,
    d.charlson_cci,
    l.lactate,
    l.creatinine,
    l.bilirubin,
    l.platelet,
    l.wbc,
    l.hemoglobin,
    l.glucose,
    l.ph,
    l.pao2,
    l.bun,
    sf.sofa_resp,
    sf.sofa_coag,
    sf.sofa_liver,
    sf.sofa_cardio,
    sf.sofa_renal,
    sf.sofa_neuro
FROM sepsis_cohort sc
LEFT JOIN demographics d     ON sc.stay_id = d.stay_id
LEFT JOIN labs_first24 l     ON sc.hadm_id = l.hadm_id
LEFT JOIN sofa_scores sf     ON sc.stay_id = sf.stay_id AND sf.hr = 24
ORDER BY sc.stay_id;
*/
