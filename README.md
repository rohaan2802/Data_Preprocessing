# Data Preprocessing

Data cleaning, imputation, and feature-scaling notebooks for ML-for-robotics coursework, including a real **diabetes** CSV with missing values.

## Overview

**Assignment 2 — ML for Robotics (Spring 2026)** focused on practical preprocessing:

- Missing-value handling (mean / median fill, `SimpleImputer`)
- Feature scaling with `StandardScaler`
- Metric selection notes for regression vs classification
- End-to-end cleaning of an unclean medical dataset

## Repository Contents

| File | Description |
|------|-------------|
| `Data_Preprocessing.ipynb` | Customer-data demos: fillna, SimpleImputer, scaling, small regression comparison |
| `ML_For_Robo.ipynb` | Full pipeline on `diabetes_unclean.csv` |
| `diabetes_unclean.csv` | Unclean diabetes-related tabular data (~1,000 rows) |
| `Asg-2-ML for Robotics-Spring 2026-NU.pdf` | Assignment brief |
| `i222391_i222327.pdf` | Submission / report PDF |

## Dataset: `diabetes_unclean.csv`

| Column | Role |
|--------|------|
| `ID`, `No_Pation` | Identifiers |
| `Gender`, `AGE` | Demographics |
| `Urea`, `Cr`, `HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI` | Clinical / lab features |
| `CLASS` | Label / category |

`ML_For_Robo.ipynb` counts missingness, builds mean- and median-imputed copies, drops ID-like / categorical columns, then applies `StandardScaler`.

## Notebook Highlights

### `Data_Preprocessing.ipynb`

- Toy customer tables with intentional `NaN`s  
- Cleaning via column means and `SimpleImputer`  
- Scaling plus discussion of **MAE / R²** (regression) vs precision-style metrics (classification)

### `ML_For_Robo.ipynb`

1. Load and `describe()` the diabetes CSV  
2. Count missing values  
3. Fill with **mean** and **median**  
4. Drop non-scaled columns (`ID`, `No_Pation`, `Gender`, `AGE`, `CLASS`)  
5. Apply `StandardScaler` and round for display  
6. Tabular before/after visualization helpers  

## Tech Stack

Python 3 · Jupyter · pandas · NumPy · Matplotlib · scikit-learn (`SimpleImputer`, `StandardScaler`, `train_test_split`, `LinearRegression`, metrics)

## Getting Started

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook ML_For_Robo.ipynb
```

Keep `diabetes_unclean.csv` beside `ML_For_Robo.ipynb`.

## Project Structure

```
Data_Preprocessing/
├── Data_Preprocessing.ipynb
├── ML_For_Robo.ipynb
├── diabetes_unclean.csv
├── Asg-2-ML for Robotics-Spring 2026-NU.pdf
└── i222391_i222327.pdf
```

## Author

**Mohammad Rohaan** — i222327 · [rohaan2802](https://github.com/rohaan2802)
