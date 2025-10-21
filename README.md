# ICU Survival Analysis

## Overview
This repository provides a three-part, notebook-driven walkthrough of intensive care unit (ICU) survival analysis using the publicly available PhysioNet/Computing in Cardiology Challenge 2012 dataset. The workflow starts with cohort curation and Kaplan–Meier (KM) exploration, progresses to multivariable Cox proportional hazards (CPH) modeling, and culminates in decision tree and random forest survival classifiers that operate at clinically actionable horizons (7, 30, and 60 days). Together, the notebooks show how to quantify in-hospital mortality risk, interpret model outputs, and translate analytical findings into operational guidance for multidisciplinary ICU teams.【F:01_ICU_Survival_analysis_KM.ipynb†L84-L109】【F:03_ICU_Survival_analysis_DT_RF.ipynb†L1-L36】

## Dataset
All analyses rely on the PhysioNet Challenge 2012 Set A cohort (4,000 de-identified ICU stays) and its engineered descriptors: record identifiers, severity scores (SAPS-I, SOFA), length of stay, survival time, and in-hospital death indicators. The KM notebook derives analysis-ready `duration` and `event` variables that align with the challenge definitions to support consistent censoring logic across notebooks.【F:01_ICU_Survival_analysis_KM.ipynb†L84-L109】

## Environment setup
A Conda environment specification is provided for reproducibility. Create and activate the environment with:

```bash
git clone https://github.com/<your-org>/ICU_Survival_analysis.git
cd ICU_Survival_analysis
conda env create -f environment.yml
conda activate icu-survival
```

The environment installs Python 3.11, pandas, numpy, matplotlib, seaborn, scikit-learn, lifelines, scikit-survival, and JupyterLab—covering all dependencies used across the notebooks.【F:environment.yml†L1-L15】

## Workflow at a glance

| Notebook | Primary focus | Representative outputs |
| --- | --- | --- |
| `01_ICU_Survival_analysis_KM.ipynb` | Data validation, censoring checks, Kaplan–Meier, cumulative incidence, Nelson–Aalen, and subgroup log-rank tests | Survival probabilities at 7/30/90 days, median survival, 90-day RMST, SAPS-I stratified comparisons |
| `02_ICU_Survival_analysis_CPH.ipynb` | Leakage-aware feature curation, train/validation/test splits, univariable and multivariable CPH modeling, proportional hazards diagnostics, time-dependent metrics, baseline comparisons | Age-only HR and C-index diagnostics, multivariable HR interpretation (SOFA, ICU type), positive Brier skill scores, calibration assessment |
| `03_ICU_Survival_analysis_DT_RF.ipynb` | Shared preprocessing pipeline, decision tree and random forest classifiers at fixed horizons, threshold analysis, decision curves, fairness checks, final model showdown | Decision tree AUROC band (≈0.68–0.76), random forest AUROC 0.82 at 7 days, subgroup AUROC gaps, cross-model calibration comparison |

## Step-by-step analysis summary

### 1. Cohort QC and Kaplan–Meier exploration (`01_ICU_Survival_analysis_KM.ipynb`)
* Confirms an in-hospital death rate of 13.85%, ensuring the time-to-event targets and censoring logic are consistent before modeling.【F:01_ICU_Survival_analysis_KM.ipynb†L3632-L3634】
* Reports KM survival probabilities of 0.942, 0.713, and 0.341 at 7, 30, and 90 days, respectively, providing decision-ready risk summaries for early, intermediate, and prolonged ICU stays.【F:01_ICU_Survival_analysis_KM.ipynb†L2791-L2794】
* Calculates a median survival time of 58 days and a 90-day restricted mean survival time (RMST) of 56.5 days overall, with SAPS-I tertiles spanning 49.5–61.2 days to quantify the expected time patients spend in-hospital under competing discharge.【F:01_ICU_Survival_analysis_KM.ipynb†L2825-L2832】【F:01_ICU_Survival_analysis_KM.ipynb†L3730-L3733】
* Uses log-rank testing and narrative guidance to highlight that sex differences are not statistically significant (p ≈ 0.25) while SAPS-I severity drives clinically meaningful divergence, orienting which subgroups merit escalation planning.【F:01_ICU_Survival_analysis_KM.ipynb†L3656-L3657】【F:01_ICU_Survival_analysis_KM.ipynb†L2764-L2778】

### 2. Cox proportional hazards modeling (`02_ICU_Survival_analysis_CPH.ipynb`)
* Demonstrates leakage-aware preprocessing (train-only fitting of pipelines) and motivates multivariable modeling by showing that age alone has an HR of 0.992 with a weak C-index of 0.518—evidence of confounding and limited prognostic value in isolation.【F:02_ICU_Survival_analysis_CPH.ipynb†L1250-L1277】
* Fits a penalized multivariable CPH model with moderate validation concordance (0.659) and interpretable hazard ratios, emphasizing SOFA as the dominant risk driver (HR 1.088) and ICU type effects such as the protective CSRU indicator (HR 0.420).【F:02_ICU_Survival_analysis_CPH.ipynb†L1645-L1699】
* Evaluates time-dependent discrimination and calibration using cumulative dynamic AUC, Brier score, and competing-risk-aware cumulative incidence, positioning the model as a moderate yet actionable screener across 7, 30, and 60-day horizons.【F:02_ICU_Survival_analysis_CPH.ipynb†L2360-L2392】【F:02_ICU_Survival_analysis_CPH.ipynb†L2600-L2658】
* Benchmarks against a Kaplan–Meier baseline and documents positive Brier Skill Scores at all horizons, confirming the multivariable model adds signal beyond cohort-level averages before applying isotonic calibration when warranted.【F:02_ICU_Survival_analysis_CPH.ipynb†L2529-L2634】

### 3. Decision tree and random forest horizon models (`03_ICU_Survival_analysis_DT_RF.ipynb`)
* Builds a shared preprocessing pipeline with stratified splits reused across models to ensure fair comparisons and leakage-free evaluation of 7/30/60-day mortality classification tasks.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L1-L58】【F:03_ICU_Survival_analysis_DT_RF.ipynb†L93-L132】
* Trains interpretable decision trees whose test AUROC ranges from roughly 0.68–0.76 at 7 days, highlighting the transparency-versus-accuracy trade-offs that make trees valuable for bedside rule generation but less competitive for precise risk scoring.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L1451-L1461】
* Deploys random forests that improve short-term discrimination to an AUROC of 0.82 at 7 days while still requiring workload-aware threshold selection and calibration to support operational adoption.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L2245-L2254】【F:03_ICU_Survival_analysis_DT_RF.ipynb†L2736-L2744】
* Audits fairness and subgroup stability, noting gender-specific AUROC gaps (0.871 vs. 0.778) and reinforcing the need to monitor differential performance before deployment.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L3092-L3115】

## Cross-model comparison and clinical positioning
The final showdown contrasts calibrated KM-derived survival estimates, CPH, decision tree, and random forest outputs. Random forests dominate 7-day discrimination, all models converge around 30-day performance, and decision trees catch up by 60 days, while CPH and random forests remain the most reliable in low-risk calibration bands. The practical takeaway: use random forests for maximal predictive accuracy, Cox models for interpretable effect estimates, and decision trees when transparent if-then logic outweighs marginal accuracy gains.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L3503-L3533】

## Running the notebooks
1. Launch JupyterLab (or Jupyter Notebook) inside the activated environment: `jupyter lab`.
2. Execute the notebooks in order (`01` → `02` → `03`) to reproduce the entire analysis pipeline. Update dataset paths if you relocate the CSV and adjust plotting settings through `utils.py` as needed.
3. Expect the KM notebook to finish within minutes on a modern laptop; tree and forest training may take longer depending on available CPU cores.

## Extending the analysis
* Swap in institutional cohorts that match the PhysioNet schema to perform external validation and local calibration.
* Expand feature engineering (e.g., time-windowed labs, ventilator settings) to boost discrimination in the CPH and ensemble models.
* Iterate on fairness checks (additional demographics, comorbidities) and decision-curve thresholds to align alerts with staffing capacity.【F:03_ICU_Survival_analysis_DT_RF.ipynb†L2968-L3129】

## Troubleshooting
* Install `scikit-survival` via Conda Forge to avoid compilation issues on Linux and macOS.
* Use pandas `usecols` to reduce memory pressure when exploring the 23 MB CSV on resource-limited machines.
* If running headless, export Matplotlib figures directly to disk or enable interactive backends compatible with your environment.

## References and license
Please cite PhysioNet, Harrell’s *Regression Modeling Strategies*, and Ishwaran et al. on Random Survival Forests when using this repository. Code is released under the MIT License; dataset reuse follows PhysioNet’s credentialed data use agreement (see [`LICENSE`](LICENSE)).
