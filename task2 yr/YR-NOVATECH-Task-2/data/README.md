# Dataset Documentation — YR NOVATECH Task 2

## Primary Dataset: UCI Adult Income

| Property | Details |
|---|---|
| **Dataset Name** | Adult Income (also called "Census Income") |
| **Source URL** | https://archive.ics.uci.edu/ml/datasets/adult |
| **Direct Data URL** | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data |
| **License** | Public domain. UCI ML Repository datasets are made available for research and educational use. Verify any commercial use restrictions with the repository. |
| **Original Rows** | 48,842 (training + test combined) |
| **Original Columns** | 15 |
| **Verified** | Row count verified by the application at download time. |

---

## Description

This dataset was donated to the UCI ML Repository in 1996 by Ronny Kohavi and Barry Becker.
It contains census data used to predict whether a person's annual income exceeds $50,000.

It is an ideal preprocessing demonstration dataset because it:
- Has a realistic mix of numerical and categorical features.
- Contains missing values (coded as ` ?` in the original file).
- Has a binary prediction target (`income`).
- Is large enough (>1,000 rows) to demonstrate meaningful statistics.

---

## Column Definitions

| Column | Type | Description |
|---|---|---|
| `age` | Numerical (int) | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Self-emp, Gov, etc.) |
| `fnlwgt` | Numerical (int) | Final sampling weight (census statistic) |
| `education` | Categorical | Highest education level attained |
| `education_num` | Numerical (int) | Education level encoded as integer |
| `marital_status` | Categorical | Marital status |
| `occupation` | Categorical | Occupation category |
| `relationship` | Categorical | Relationship to household head |
| `race` | Categorical | Race category |
| `sex` | Categorical | Sex (Male / Female) |
| `capital_gain` | Numerical (int) | Capital gain from investments |
| `capital_loss` | Numerical (int) | Capital loss from investments |
| `hours_per_week` | Numerical (int) | Hours worked per week |
| `native_country` | Categorical | Country of origin |
| `income` | Categorical (target) | `<=50K` or `>50K` |

---

## Collection Method

This data was extracted from the 1994 US Census database by Barry Becker.
It is available from the UCI Machine Learning Repository.

---

## Known Limitations

- Missing values are coded as ` ?` (a space followed by a question mark) in the original file. The application handles this at load time using `na_values=" ?"`.
- The `fnlwgt` column represents a sampling weight and may not be meaningful for typical ML tasks; consider dropping it for predictive modelling.
- The `education` and `education_num` columns are highly correlated; consider dropping one.
- The dataset reflects 1994 US census data and does not represent current demographics.
- Class imbalance: approximately 76% of records are `<=50K`.

---

## Data Quality Observations (from initial inspection)

- Missing values are present in `workclass`, `occupation`, and `native_country`.
- No structural parse errors are expected.
- `capital_gain` and `capital_loss` contain many zero values — this is expected (not missing data).
- All numerical columns have non-negative values by design.

---

## Manual Download Instructions

If the automatic download fails:

1. Open a browser and go to:
   ```
   https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
   ```
2. The page will display raw CSV data. Right-click → **Save As** → save as `adult.csv`.
3. Place `adult.csv` in the `data/` folder of this project.
4. In the application, go to **📥 Data Collection → Upload Your File** and upload `adult.csv`.

The application will add the correct column headers automatically if you select the UCI Adult format from the upload tab, or you can add headers manually before uploading.

---

## Note on Dataset Inclusion in Git

The actual dataset file is excluded from version control by `.gitignore` (`data/*.csv`).
This keeps the repository lightweight and avoids potential licensing questions.
The dataset can be downloaded at runtime by the application or manually as described above.
