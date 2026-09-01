-- Diabetes Risk Prediction Project - SQL Exploratory Analysis
-- These queries were run directly against the patient database
-- to understand risk patterns BEFORE any machine learning modeling.
-- 1. Diabetes rate by age group
SELECT
    CASE
        WHEN age < 35 THEN 'Under 35'
        WHEN age BETWEEN 35 AND 55 THEN '35-55'
        ELSE 'Over 55'
    END AS age_group,
    COUNT(*) AS total_patients,
    SUM(diabetes) AS diabetes_cases,
    ROUND(AVG(CAST(diabetes AS FLOAT)) * 100, 1) AS diabetes_rate_pct
FROM patients
GROUP BY
    CASE
        WHEN age < 35 THEN 'Under 35'
        WHEN age BETWEEN 35 AND 55 THEN '35-55'
        ELSE 'Over 55'
    END
ORDER BY diabetes_rate_pct DESC;

-- 2. Average clinical measurements: diabetic vs non-diabetic patients
SELECT
    CASE WHEN diabetes = 1 THEN 'Diabetic' ELSE 'Non-Diabetic' END AS status,
    ROUND(AVG(glucose), 1) AS avg_glucose,
    ROUND(AVG(bmi), 1) AS avg_bmi,
    ROUND(AVG(age), 1) AS avg_age
FROM patients
GROUP BY diabetes;

-- 3. Window Function: rank patients by glucose level WITHIN each activity level
--    (identifies highest-risk patients in each activity category)
SELECT
    patient_id, physical_activity, glucose, diabetes,
    RANK() OVER (PARTITION BY physical_activity ORDER BY glucose DESC) AS glucose_rank
FROM patients;

-- 4. Diabetes rate by family history
SELECT
    family_history,
    COUNT(*) AS total_patients,
    ROUND(AVG(CAST(diabetes AS FLOAT)) * 100, 1) AS diabetes_rate_pct
FROM patients
GROUP BY family_history;

-- 5. BMI category breakdown with diabetes rate
SELECT
    CASE
        WHEN bmi < 25 THEN 'Normal'
        WHEN bmi BETWEEN 25 AND 30 THEN 'Overweight'
        ELSE 'Obese'
    END AS bmi_category,
    COUNT(*) AS total_patients,
    ROUND(AVG(CAST(diabetes AS FLOAT)) * 100, 1) AS diabetes_rate_pct
FROM patients
GROUP BY
    CASE
        WHEN bmi < 25 THEN 'Normal'
        WHEN bmi BETWEEN 25 AND 30 THEN 'Overweight'
        ELSE 'Obese'
    END;

-- 6. Top 10 highest-risk patients (by glucose level)
SELECT TOP 10 patient_id, age, glucose, bmi, diabetes
FROM patients
ORDER BY glucose DESC;