# ICU Survival Analysis

A clear, end to end walkthrough of intensive care unit survival analysis across three Jupyter notebooks. The pipeline progresses from cohort curation and Kaplan-Meier exploration to multivariable Cox proportional hazards modeling and finally decision tree and random forest models evaluated at clinically actionable horizons

## Repository structure

- 01_ICU_Survival_analysis_KM.ipynb
- 02_ICU_Survival_analysis_CPH.ipynb
- 03_ICU_Survival_analysis_DT_RF.ipynb

## Dataset

- Source dataset is PhysioNet Computing in Cardiology Challenge 2012, Set A, with 4 000 de-identified ICU stays
- Engineered variables include `duration` in days, `event` indicator for in-hospital death, severity scores such as SAPS-I and SOFA, ICU service type, and basic descriptors
- Censoring aligns with the challenge definitions so that survival times and event flags are consistent across notebooks

## Environment and reproducibility

- Python 3.11 recommended
- Key libraries include pandas, numpy, matplotlib, scikit-learn, lifelines, scikit-survival, jupyterlab
- Create an isolated environment with Conda or uv, then run `jupyter lab`
- Set dataset paths at the top of each notebook if your files live outside the repo

## Workflow at a glance

| Notebook | Primary focus | Representative outputs |
| --- | --- | --- |
| 01_ICU_Survival_analysis_KM.ipynb | Data validation, censoring checks, Kaplan-Meier curves, cumulative incidence, Nelson-Aalen, subgroup log-rank tests | Survival probabilities at 7, 30, 90 days, median survival, 90-day restricted mean survival time, SAPS-I stratified comparisons |
| 02_ICU_Survival_analysis_CPH.ipynb | Leakage-aware preprocessing, univariable and multivariable Cox proportional hazards, proportional hazards diagnostics, time-dependent metrics, baseline comparisons | Hazard ratios for age, SOFA, ICU type, concordance index, Brier and calibration curves |
| 03_ICU_Survival_analysis_DT_RF.ipynb | Shared preprocessing, decision tree and random forest classifiers at fixed horizons, threshold analysis, decision curves, subgroup checks, final comparison | AUROC at 7, 30, 60-day horizons, operating-point guidance, subgroup stability and fairness checks |

## End to end process

### 1) Cohort QC and Kaplan-Meier exploration

- Confirms in-hospital death rate of 13.85 percent before modeling
- Kaplan-Meier survival probabilities at 7, 30, 90 days are 0.942, 0.713, 0.341
- Median survival time is 58 days and 90-day restricted mean survival time is 56.5 days overall
- SAPS-I severity tertiles show expected separation in survival time, roughly 49.5 to 61.2 days across bands
- Log-rank testing shows no statistically significant difference by sex with p ≈ 0.25, while SAPS-I differences are significant

Practical use
- Treat the KM curves as a cohort baseline for risk communication and for sanity-checking later model outputs
- Use subgroup curves to identify strata that may need separate thresholds or escalation paths

### 2) Cox proportional hazards modeling

- Train only on the training split to avoid leakage, keep validation and test for honest assessment
- Age alone has a hazard ratio of 0.992 and weak concordance of 0.518, which motivates a multivariable approach
- A penalized multivariable Cox model reaches validation concordance near 0.659 with interpretable hazard ratios
- SOFA is the dominant risk driver with HR ≈ 1.088 per unit, and the CSRU ICU type indicator appears protective with HR ≈ 0.420
- Time-dependent area under the curve and Brier score show the model is a moderate but actionable screener at 7, 30, 60 days
- Calibration against cohort baselines improves with isotonic calibration when needed

Practical use
- Use Cox when effect sizes and explanatory narratives are required
- Favor this model to inform pathway design and feature prioritization, even if a tree ensemble wins on raw discrimination

### 3) Decision tree and random forest horizon models

- Reuse the same stratified splits and preprocessing for leakage-free, apples-to-apples comparisons
- Decision trees deliver AUROC around 0.68 to 0.76 at 7 days, trading a little accuracy for high transparency
- Random forests improve short-term discrimination to AUROC ≈ 0.82 at 7 days
- Subgroup review shows gender AUROC gap of roughly 0.871 vs 0.778, which merits monitoring before deployment
- Choose thresholds using workload-aware trade-offs, then check calibration and decision curves at the chosen operating point

Practical use
- Use random forests for maximal predictive accuracy and alerts, backed by calibration checks
- Use decision trees when simple if-then rules are preferred at the bedside

## Model comparison and selection

- For 7-day risk, random forests dominate discrimination while keeping acceptable calibration after post-processing
- By 30 days, models converge and Cox plus random forest give the most reliable low-risk calibration
- By 60 days, decision trees catch up on discrimination, narrowing the gap while staying easier to explain

## How to reproduce

1. Launch JupyterLab inside the environment
2. Run the notebooks in order 01, then 02, then 03
3. Update the dataset path at the top of 01 if your files are elsewhere
4. Expect 01 to run in minutes on a modern laptop, while 03 may take longer depending on CPU cores
5. Save figures to disk if running headless

## Outputs to expect

- KM plots with 7, 30, 90 day survival estimates and subgroup curves
- Cox model summary with hazard ratios, concordance, calibration and Brier curves
- Decision tree diagram with rules at the selected depth
- Random forest performance table with AUROC, precision, recall, thresholds by horizon
- Cross-model comparison figure and brief written guidance for use in clinical workflows

## Extending the analysis

- Swap in a local cohort that matches the PhysioNet schema for external validation and local calibration
- Engineer additional features such as time-windowed labs and ventilator settings to improve discrimination
- Expand fairness and stability checks across additional demographics and comorbidities
- Use decision curves and cost assumptions to pick thresholds that fit staffing capacity

## Troubleshooting

- Install scikit-survival from conda-forge to avoid compilation issues
- Use pandas `usecols` to lower memory if you only need a subset of columns
- If interactive backends are not available, export Matplotlib figures directly to disk

## Ethical use and limitations

- Models are trained on a historical research cohort and are not a substitute for clinical judgment
- Performance can drift with local case-mix, care protocols, and data coding, so perform local validation and calibration
- Review subgroup performance before deploying any alert that could affect resource allocation

## License and citation

- Code under the MIT License
- Dataset usage follows PhysioNet credentialed access terms
- When using this repository, cite the PhysioNet Challenge 2012 dataset, Harrell’s Regression Modeling Strategies, and work on Random Survival Forests by Ishwaran and colleagues
