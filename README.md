# Data Preprocessing — Cleaning, Imputation, and Scaling (ML for Robotics A#02)

Two Jupyter notebooks for **ML for Robotics, Assignment 2 (Spring 2026)**. `Data_Preprocessing.ipynb` walks through **mean fill**, `SimpleImputer`, `StandardScaler`, and a tiny **linear regression** that compares MAE / R² before and after cleaning. `ML_For_Robo.ipynb` applies the same ideas to **`diabetes_unclean.csv`** (1,009 rows of Iraqi-hospital-style lab fields). No classifier is trained on `CLASS`; the diabetes notebook stops at missing-value counts, mean vs median fill, z-scoring, and table plots.

**Author:** Mohammad Rohaan · **22I-2327** · [rohaan2802](https://github.com/rohaan2802)  
Group PDF on GitHub: `i222391_i222327.pdf` (joint submission with roll **i222391**). Assignment brief: `Asg-2-ML for Robotics-Spring 2026-NU.pdf` (scanned/custom-font PDF; requirements below follow the **implemented notebooks**, not OCR of that file).

## Table of contents

- [Problem statement / academic context](#problem-statement--academic-context)
- [Features](#features)
- [Architecture / design](#architecture--design)
- [Algorithms and data structures](#algorithms-and-data-structures)
- [File-by-file reference](#file-by-file-reference)
- [Data formats / schemas](#data-formats--schemas)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites and install](#prerequisites-and-install)
- [How to build and run](#how-to-build-and-run)
- [Usage walkthrough](#usage-walkthrough)
- [Configuration / constants](#configuration--constants)
- [Results / metrics](#results--metrics)
- [Controls / CLI](#controls--cli)
- [Known limitations / bugs](#known-limitations--bugs)
- [How to extend](#how-to-extend)
- [Author](#author)

## Problem statement / academic context

Raw tables for learning are messy: **NaNs**, mixed types, IDs that must not be z-scored, and **skewed lab values**. The assignment is to show a reproducible chain:

1. Inspect shape / `describe` / missingness.
2. Impute (pandas `fillna` mean or median, or sklearn `SimpleImputer(strategy='mean')`).
3. Drop identifiers and categoricals before scaling.
4. `StandardScaler` (mean 0, variance 1).
5. Visualize first rows as matplotlib **tables** (before / after clean / after scale).
6. In the toy customer notebook, argue **which metric belongs to which task** (MAE and R² for regression; precision/recall for classification) and fit `LinearRegression` on `SpendingScore`.

## Features

**`Data_Preprocessing.ipynb` (5 cells)**

| Cell | Role |
|------|------|
| 0 | Markdown: “Preprocess customer data with cleaning and feature scaling.” |
| 1 | Five labeled customers; `fillna(mean)`; `StandardScaler`; three colored tables (`#FFCCCC` / `#CCFFCC` / `#CCE5FF`). |
| 2 | Six customers; `SimpleImputer` → scale → `train_test_split(..., test_size=0.3, random_state=42)`. |
| 3 | Markdown heading “Comparison”. |
| 4 | n=50 synthetic customers, two planted NaNs, OLS on `Age`+`Income` → `SpendingScore`, MAE/R² at three stages, styled results table. |

**`ML_For_Robo.ipynb` (15 cells)**

- `read_csv("diabetes_unclean.csv")`, `head()`, `describe()`.
- `isnull().sum()` per column and **total missing = 15**.
- `df_Mean = fillna(mean)`, `df_Median = fillna(median)`; both print **0** remaining missing.
- Drop `ID`, `No_Pation`, `Gender`, `AGE`, `CLASS` **before** scaling (AGE is numeric but excluded).
- `StandardScaler.fit_transform` → `round(4)`.
- `plot_table`: first **10** rows for before-clean / after-clean / after-scale.

## Architecture / design

```
Data_Preprocessing.ipynb
  in-memory customer dicts
       → impute (mean / SimpleImputer)
       → StandardScaler
       → (cell 4) LinearRegression + MAE, R²

ML_For_Robo.ipynb
  diabetes_unclean.csv (1009 × 14)
       → missingness audit
       → mean copy and median copy
       → drop ID / patient no. / Gender / AGE / CLASS
       → StandardScaler on 9 lab+BMI columns
       → matplotlib tables (not histograms)
```

The two notebooks **do not share variables**. The diabetes pipeline never calls `SimpleImputer` or a sklearn estimator.

## Algorithms and data structures

- **Mean / median fill:** pandas `DataFrame.fillna` with `numeric_only=True` (strings such as `Gender` and `CLASS` are unchanged).
- **`SimpleImputer(strategy='mean')`:** cell 2 only; `fit_transform` on the whole 6×3 frame (no train-only fit).
- **`StandardScaler`:** \( z = (x - \mu)/\sigma \) with population-style sklearn defaults. Cell 4 scales **features only** after mean-fill; OLS with intercept is invariant to that scale, which is why MAE/R² match the “after cleaning” row.
- **`LinearRegression`:** OLS, `test_size=0.2`, `random_state=42`, `np.random.seed(42)`.
- **Leakage notes (as coded):** cell 1–2 scale the **full** toy frame; cell 4 also `fit_transform`s all cleaned `X` **before** the split. Only the energy-assignment notebook (separate repo) fits the scaler on train alone.

## File-by-file reference

| File | Description |
|------|-------------|
| `Data_Preprocessing.ipynb` | Toy customers + metric comparison (executed Colab outputs). |
| `ML_For_Robo.ipynb` | Diabetes CSV pipeline (executed `describe` / missing / scaled head). |
| `diabetes_unclean.csv` | 1,009 data rows + header. |
| `Asg-2-ML for Robotics-Spring 2026-NU.pdf` | Course brief (repo root). |
| `i222391_i222327.pdf` | Group write-up (rolls **i222391** and **i222327**). |

## Data formats / schemas

Exact CSV header:

```text
ID,No_Pation,Gender,AGE,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI,CLASS
```

First executed `head()` row: `ID=502`, `No_Pation=17975`, `Gender=F`, `AGE=50`, `Urea=4.7`, `Cr=46`, `HbA1c=4.9`, `Chol=4.2`, `TG=0.9`, `HDL=2.4`, `LDL=1.4`, `VLDL=0.5`, `BMI=24.0`, `CLASS=N`.

**Missing cells (notebook `isnull().sum()`):** AGE 1, Urea 1, Cr 2, HbA1c 3, Chol 2, TG 2, HDL 1, LDL 2, VLDL 1; ID / No_Pation / Gender / BMI / CLASS = 0. **Total = 15.**

**`df.describe()` (numeric, executed):**

| | count | mean | std | min | 50% | max |
|--|-------|------|-----|-----|-----|-----|
| AGE | 1008 | 53.620 | 8.741 | 25 | 55 | 79 |
| Urea | 1008 | 5.131 | 2.931 | 0.5 | 4.6 | 38.9 |
| Cr | 1007 | 68.973 | 59.813 | 6 | 60 | **800** |
| HbA1c | 1006 | 8.284 | 2.534 | 0.9 | 8.0 | 16 |
| Chol | 1007 | 4.864 | 1.297 | 0 | 4.8 | 10.3 |
| TG | 1007 | 2.349 | 1.397 | 0.3 | 2.0 | 13.8 |
| HDL | 1008 | 1.204 | 0.658 | 0.2 | 1.1 | 9.9 |
| LDL | 1007 | 2.610 | 1.116 | 0.3 | 2.5 | 9.9 |
| VLDL | 1008 | 1.851 | 3.650 | 0.1 | 0.9 | 35 |
| BMI | 1009 | 29.590 | 4.946 | 19 | 30 | 47.75 |

`ID` mean 339.16 (1–800); `No_Pation` is a large patient-file number (max ~7.54×10⁷), not a lab analyte.

**Categorical value counts (from the CSV):** `Gender` = M 570, F 437, **f 2** (inconsistent case). `CLASS` = `Y` 840, `N` 102, `P` 53, plus trailing-space labels **`Y ` 13** and **`N ` 1**. The notebook never maps N/P/Y to English names and never strips those spaces.

After the drop, scaled columns are: `Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI`.

**Toy cell 1 data** (index `Customer 1`–`5`): Age `{25, 30, NaN, 45, 35}`, Income `{50000, NaN, 60000, 80000, 75000}`, SpendingScore `{60, 70, 80, NaN, 50}`.

**Toy cell 2:** six rows with `None` in Age row 4, Income row 2, SpendingScore row 3; split **~70/30**.

## Tech stack

- Python 3 / Jupyter (outputs include Colab `Styler.applymap` deprecation warning)
- pandas, numpy, matplotlib
- sklearn: `SimpleImputer`, `StandardScaler`, `train_test_split`, `LinearRegression`, `mean_absolute_error`, `r2_score`

## Project structure

```
Data_Preprocessing/
├── Data_Preprocessing.ipynb
├── ML_For_Robo.ipynb
├── diabetes_unclean.csv
├── Asg-2-ML for Robotics-Spring 2026-NU.pdf
└── i222391_i222327.pdf
```

## Prerequisites and install

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## How to build and run

```bash
cd Data_Preprocessing
jupyter notebook Data_Preprocessing.ipynb
jupyter notebook ML_For_Robo.ipynb
```

Keep `diabetes_unclean.csv` in the **same directory** as `ML_For_Robo.ipynb`. Run cells top to bottom. No extra datasets or Excel files.

## Usage walkthrough

1. Open the customer notebook; run cell 1 and read the three colored tables (NaNs become column means; scaled cells are order-1 numbers).
2. Cell 2 prints original / cleaned / scaled frames and the 0.3 test split.
3. Cell 4 prints the metric explanation, MAE/R² formulas, then the styled comparison table (numbers in [Results](#results--metrics)).
4. Open `ML_For_Robo.ipynb`. Confirm 1,009 rows and 15 missing values.
5. Compare mean vs median copies (both reach 0 missing on numerics). Prefer **median** in a viva if asked about `Cr` (max 800 vs median 60) or `VLDL` (max 35 vs median 0.9).
6. After the drop, check `df_scaled.head()` (first printed z-row for Urea…BMI: `-0.1472, -0.3847, -1.3384, -0.5125, -1.0382, 1.8187, -1.0859, -0.3704, -1.1307`).
7. Inspect the three `plot_table` figures (first 10 rows only).

## Configuration / constants

| Item | Value |
|------|-------|
| Diabetes path | `"diabetes_unclean.csv"` |
| Diabetes rows | 1009 |
| Drop before scale | `['ID','No_Pation','Gender','AGE','CLASS']` |
| Scaled precision | 4 decimal places |
| Table plot | `figsize=(12, 4)`, first 10 rows, fontsize 10 |
| Customer cell 1 figure | `figsize=(16, 5)`, 1×3 tables, fontsize 9 |
| Comparison n | 50; missing at `data.loc[5,'Income']`, `data.loc[10,'SpendingScore']` |
| Comparison split | `test_size=0.2`, `random_state=42` |
| Imputer split (cell 2) | `test_size=0.3`, `random_state=42` |

## Results / metrics

**Diabetes missingness (executed):** 15 cells; mean and median fills both report **0** remaining missing.

**Customer comparison notebook (executed HTML table in cell 4):**

| Stage | MAE | R² |
|-------|-----|-----|
| Before Cleaning | **29.779585** | **−0.017902** |
| After Cleaning | **31.184247** | **−0.094195** |
| After Scaling | **31.184247** | **−0.094195** |

“Before cleaning” fills **features and the target** with **0** for the two NaNs (`y_raw = SpendingScore.fillna(0)`). Mean-fill of the target changes the 80/20 test draw enough that MAE **rises** and R² becomes more negative. After scaling, MAE and R² are **identical** to the cleaned row (OLS + intercept). All three R² values are **below 0**: Age and Income do not explain SpendingScore on this 50-row draw. The pedagogical point in the explanation block still stands: use **MAE / R²** for regression, **precision / recall** when false positives or false negatives are costly.

## Controls / CLI

Notebooks only. No argparse, no saved `.pkl`.

## Known limitations / bugs

- Assignment PDF text is **not extractable** here (embedded fonts/scan); this README tracks the notebooks and CSV.
- Diabetes `CLASS` and `Gender` are **not cleaned** (trailing spaces; mixed `F`/`f`).
- `AGE` is dropped rather than imputed-and-scaled, so age never enters a model.
- No train/test split or sklearn model on the medical table.
- Cell 4 `Styler.applymap` is deprecated (`Styler.map`).
- Scaling the full matrix before `train_test_split` in the toy notebooks leaks test statistics into \(\mu,\sigma\).
- Mean vs median copies are created; **only `df_Mean` is scaled**.
- Group PDF implies a partner (i222391); this snapshot’s notebooks are Rohaan’s local copies.

## How to extend

- Strip `CLASS`/`Gender`, then encode and train a three-way classifier (the CSV supports it; this repo does not).
- Fit the scaler on **train** only; compare median vs mean with a downstream model.
- Replace table plots with histograms / boxplots of `Cr` and `VLDL` to justify median imputation.
- Persist `df_scaled` to CSV for later assignments.

## Author

**Mohammad Rohaan** — 22I-2327 · [rohaan2802](https://github.com/rohaan2802)
