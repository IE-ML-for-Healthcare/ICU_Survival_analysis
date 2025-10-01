# ICU Survival Analysis

## Introduction and Purpose
This repository accompanies a two-part, notebook-driven tutorial on intensive care unit (ICU) survival analysis using the [PhysioNet/Computing in Cardiology Challenge 2012 dataset](https://physionet.org/content/challenge-2012/1.0.0/). The project illustrates how to quantify in-hospital mortality risk for critically ill patients and how to communicate survival insights to multidisciplinary ICU teams. By working through the notebooks you will:

- Understand the impact of right censoring on survival estimates and why Kaplan–Meier methods remain essential in the ICU setting.
- Translate aggregate survival curves into clinically interpretable metrics at 7, 30, and 90 days after admission.
- Build and critique individual-risk models with Cox proportional hazards and Random Survival Forests (RSF) to support triage, escalation, and resource-planning decisions.

The primary audience includes clinical data scientists, intensivists exploring data-driven quality initiatives, and graduate students learning applied survival analysis in healthcare.

## Motivation and Clinical Context
Critically ill patients exhibit high early mortality and complex discharge dynamics. Traditional binary classification (alive/dead) ignores how risk evolves over time and fails to acknowledge the dominance of censoring (patients discharged alive). Survival analysis bridges this gap by modeling the temporal dimension of risk. The notebooks highlight:

- **Front-loaded mortality**: most ICU deaths happen within the first month, which is central for staffing, family communication, and planning follow-up services.
- **Competing outcomes**: discharge alive versus in-hospital death and how Kaplan–Meier, cumulative hazard, and cumulative incidence views complement each other.
- **Actionable horizons**: communicating 30- and 90-day restricted mean survival times (RMST) to clinicians, families, and administrators.
- **Model governance**: validating proportional hazards assumptions and complementing interpretable Cox models with flexible RSF models when non-linearities dominate.

## Repository Structure
```
ICU_Survival_analysis/
├── 01_ICU_Survival_analysis_KM.ipynb   # Cohort curation, censoring checks, Kaplan–Meier and cumulative incidence views
├── 02_ICU_Survival_analysis_CPH.ipynb  # Cox PH workflow, proportional hazards diagnostics, Random Survival Forest comparison
├── PhysionetChallenge2012-set-a.csv.gz # Aggregated outcomes and predictors for 4,000 ICU stays (PhysioNet Challenge 2012, Set A)
├── utils.py                            # Helper functions shared across notebooks (data cleaning, plotting utilities)
├── environment.yml                     # Conda environment specification for reproducible execution
└── README.md                           # Project overview and usage guide (this file)
```

## Data Description
The project uses the publicly available PhysioNet/Computing in Cardiology Challenge 2012 training set A, a cohort of 4,000 adult ICU stays with 120 engineered variables summarizing demographics, vital signs, laboratory values, severity scores, and ventilator usage.【04e1d7†L1-L14】 Key outcome variables derived in the notebooks include:

- **Length_of_stay**: days from ICU admission to hospital discharge or in-hospital death (time origin for Kaplan–Meier analysis).
- **Survival**: days to death when it occurs within follow-up; coded as −1 when death is unobserved.
- **In-hospital_death**: binary indicator of in-hospital mortality.
- **Duration (T)**: analysis-ready time variable equal to `Length_of_stay` for in-hospital outcomes.
- **Event (E)**: event indicator inferred from the dataset rules and aligned with `In-hospital_death`.

Each notebook performs label consistency checks to ensure survival targets honor the published dataset definitions.【F:01_ICU_Survival_analysis_KM.ipynb†L82-L126】【F:02_ICU_Survival_analysis_CPH.ipynb†L1-L34】 No raw protected health information (PHI) is included; all variables are de-identified and publicly distributed by PhysioNet.

> **Note:** The compressed CSV is ~23 MB. Loading all columns into memory requires ~1.5 GB RAM. Consider selective column loading or dtype optimization when scaling to larger cohorts.【04e1d7†L1-L14】

## Notebook Roadmap
### 1. Kaplan–Meier Survival Analysis (`01_ICU_Survival_analysis_KM.ipynb`)
Focus: cohort QC, censoring validation, non-parametric survival estimates.

Highlights:
- Explore censoring structure, severity score distributions, and missingness before modeling.
- Estimate overall Kaplan–Meier survival curves with at-risk tables, cumulative hazard, and cumulative incidence plots.
- Compute clinically relevant metrics such as median survival and RMST at 90 days.
- Compare subgroups (e.g., sex, SAPS-I tertiles, age bands) with log-rank tests and assess proportional hazards visually via log–log plots to motivate Cox modeling.【F:01_ICU_Survival_analysis_KM.ipynb†L71-L229】

### 2. Cox PH & Random Survival Forests (`02_ICU_Survival_analysis_CPH.ipynb`)
Focus: interpretable and flexible survival modeling.

Highlights:
- Fit Cox proportional hazards models, report hazard ratios with 95% confidence intervals, and perform Schoenfeld residual diagnostics.
- Train Random Survival Forest models to capture non-linearities and interactions beyond the Cox framework.
- Evaluate models with concordance index, time-dependent AUC (7, 30, 90 days), and integrated Brier score; calibrate 30-day survival probabilities and stratify patients into risk quintiles.【F:02_ICU_Survival_analysis_CPH.ipynb†L1-L34】

## Installation and Environment Setup
The recommended setup uses Conda or Mamba with the provided environment file.

### Prerequisites
- Conda (Anaconda, Miniconda) or Mamba ≥ 23.1
- Git

### Create the Environment
```bash
# Clone the repository
git clone https://github.com/<your-org>/ICU_Survival_analysis.git
cd ICU_Survival_analysis

# Create and activate the conda environment
conda env create -f environment.yml
conda activate icu-survival
```

The `environment.yml` file installs Python 3.11 and key scientific libraries required for the notebooks, including pandas, numpy, matplotlib, seaborn, scikit-learn, lifelines, scikit-survival, and JupyterLab.【F:environment.yml†L1-L12】

> **Optional:** If you do not use Conda, create a virtual environment with Python 3.11 and install the dependencies listed in `environment.yml` via `pip`. The `scikit-survival` package may require system-level libraries (e.g., `libgomp` on Linux). Consult the [project documentation](https://scikit-survival.readthedocs.io/en/stable/install.html) for platform-specific instructions.

## Quickstart Workflow
1. **Launch JupyterLab**
   ```bash
   jupyter lab
   ```
2. **Open the notebooks in order**
   - Start with `01_ICU_Survival_analysis_KM.ipynb` to prepare the cohort, confirm censoring logic, and understand global survival patterns.
   - Proceed to `02_ICU_Survival_analysis_CPH.ipynb` for covariate-adjusted modeling and risk stratification.
3. **Configure notebook settings**
   - Update file paths if you relocate the dataset.
   - Adjust plotting aesthetics (e.g., figure size) through the helper functions in `utils.py` as needed.
4. **Run all cells** to reproduce the analysis. Expect the Kaplan–Meier notebook to execute within a few minutes on a modern laptop; the Random Survival Forest section may take longer depending on CPU cores.

## Reproducing Results & Extending the Analysis
- **Sensitivity analyses**: Modify censoring definitions (e.g., 60-day RMST) or subgroup stratifications to match local ICU policies.
- **Feature engineering**: Incorporate additional physiologic variables or temporal aggregations to enhance RSF performance.
- **External validation**: Replace the dataset with your institutional EHR extracts, ensuring identical variable schema and careful de-identification.

## Troubleshooting
- **Missing `scikit-survival`**: Ensure you install from Conda Forge or use wheels compatible with your platform; compiling from source requires Cython and a C compiler.
- **Memory constraints**: Load a subset of columns using pandas `usecols` when exploring on resource-limited machines.
- **Plot rendering issues**: If running in a headless environment, use JupyterLab with `ipympl` or export plots directly to files via Matplotlib.

## Citation and Attribution
If you use this repository in academic work, please cite:

- Goldberger AL, Amaral LAN, Glass L, et al. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation. 2000;101(23):e215–e220.
- Harrell FE Jr. *Regression Modeling Strategies.* Springer; 2015.
- Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. *Random Survival Forests.* Annals of Applied Statistics. 2008;2(3):841–860.

## License
This project is distributed under the MIT License (see [`LICENSE`](LICENSE)). Dataset licensing follows the PhysioNet credentialed data use agreement; consult the PhysioNet repository for reuse requirements.

