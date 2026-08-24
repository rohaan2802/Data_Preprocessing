# Data Preprocessing (ML for Robotics A#02)

Assignment on **cleaning, imputation, and scaling** before learning: a small **customer** notebook and a full pipeline on an **unclean diabetes** CSV (~1009 rows).

**Rolls referenced:** i222327 (and i222391 on a joint PDF) · Spring 2026  
[rohaan2802](https://github.com/rohaan2802)

---

## Table of contents

1. [Files](#files)
2. [`Data_Preprocessing.ipynb`](#data_preprocessingipynb)
3. [`ML_For_Robo.ipynb` + diabetes CSV](#ml_for_roboipynb--diabetes-csv)
4. [Metric discussion](#metric-discussion)
5. [How to run](#how-to-run)

---

## Files

| File | Role |
|------|------|
| `Data_Preprocessing.ipynb` | Toy customers: `fillna` / `SimpleImputer` / scaler / tiny regression |
| `ML_For_Robo.ipynb` | Real-ish medical table, missingness, mean vs median, scale, table plots |
| `diabetes_unclean.csv` | Source table |
| `Asg-2-ML for Robotics-Spring 2026-NU.pdf` | Brief |
| `i222391_i222327.pdf` | Submission / report |

---

## `Data_Preprocessing.ipynb`

**Cell 1 — mean fill**

Customer rows with `Age`, `Income`, `SpendingScore` and planted `NaN`s. `df.fillna(df.mean(numeric_only=True))`, then `StandardScaler`.

**Cell 2 — sklearn imputer**

Six customers; `SimpleImputer` + `train_test_split` + scaler (classic sklearn preprocessing chain).

**Cell 3 — “Comparison”**

Fits a small `LinearRegression` and discusses:

- Regression → **MAE** (and R² as variance explained).  
- Classification → **precision**-style metrics (text block in the notebook).

This is the viva hook: **wrong metric = wrong model choice**.

---

## `ML_For_Robo.ipynb` + diabetes CSV

**Header:** `ID,No_Pation,Gender,AGE,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI,CLASS`  
**Rows:** 1009 (file reports `nrows: 1009`). `CLASS` example values include `N`.

Pipeline:

1. `read_csv("diabetes_unclean.csv")`, `head()`, `describe()`.  
2. `isnull().sum()` and total missing count.  
3. `df_Mean = fillna(mean)`, `df_Median = fillna(median)`; reprint missing totals (should be 0 on numeric cols).  
4. Drop `ID`, `No_Pation`, `Gender`, `AGE`, `CLASS` before scaling (IDs + categoricals not z-scored).  
5. `StandardScaler.fit_transform` → `round(4)`.  
6. `plot_table` helper: matplotlib tables for **before cleaning / after cleaning / after scaling** (first 10 rows).

Mean vs median: median is safer when lab values are **skewed or outlier-heavy** (typical for `Cr`, lipids). The notebook keeps both copies so you can justify a choice in the PDF.

---

## Metric discussion

From the customer notebook explanation block:

| Task | Prefer |
|------|--------|
| Regression | MAE (interpretable units), R² (fit quality) |
| Classification | Precision / related rates — not MAE |

---

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook ML_For_Robo.ipynb
```

Keep `diabetes_unclean.csv` in the same directory as that notebook.

---

## Author

**Mohammad Rohaan** — i222327 · [rohaan2802](https://github.com/rohaan2802)
