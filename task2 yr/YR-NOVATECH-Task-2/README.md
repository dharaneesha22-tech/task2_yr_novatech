# 📊 Data Collection & Preprocessing Dashboard

**An End-to-End Data Quality, Cleaning and Feature Engineering Platform**

| | |
|---|---|
| **Organisation** | YR NOVATECH |
| **Internship Task** | Task 2 — Data Collection & Preprocessing |
| **Framework** | Streamlit |
| **Language** | Python 3.11 |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | GitHub |

---

## Project Overview

This platform provides a complete, interactive workflow for data professionals and students to:

- Collect real-world tabular data from a public source or CSV upload
- Assess data quality automatically
- Clean data with configurable, reversible operations
- Engineer new features without writing code
- Explore data through interactive visualisations
- Compare datasets before and after preprocessing
- Export cleaned data and documentation in multiple formats

The primary demonstration dataset is the **UCI Adult Income dataset** (~48,842 rows, 15 columns), downloaded directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult).

---

## Objectives

1. Demonstrate real-world data collection from a legitimate public source.
2. Build an interactive, professional-quality Streamlit dashboard.
3. Implement configurable, non-destructive data cleaning operations.
4. Provide reproducible feature engineering with validation.
5. Generate comprehensive data quality reports.
6. Produce downloadable cleaned datasets and preprocessing logs.
7. Deploy the application to Streamlit Community Cloud.

---

## Features

### Data Collection
- Upload any CSV or Excel file (up to 50 MB)
- One-click load of UCI Adult Income dataset from UCI ML Repository
- Automatic column type detection
- File validation with clear error messages

### Data Quality Assessment
- Missing value counts and percentages per column
- Duplicate row detection
- Constant column detection
- High-cardinality column detection
- Numerical summary statistics
- Categorical frequency tables
- Automated suspicious-value flags (with human-review note)

### Data Cleaning
- Numerical imputation (mean / median / constant)
- Categorical imputation (mode / constant)
- Drop rows or columns with missing values
- Remove exact duplicate rows
- Convert columns to numeric with error reporting
- Parse datetime columns
- Text normalisation (trim, whitespace, casing)
- Drop or rename columns

### Outlier Analysis
- IQR-based outlier detection with configurable multiplier
- Summary table of Q1, Q3, IQR, bounds, and outlier counts
- Box plots per column
- Cap (winsorise) or remove outliers

### Feature Engineering
- Arithmetic features (add, subtract, multiply, divide) — no eval()
- Date feature extraction (year, month, day, day of week, hour, quarter)
- Categorical binning with custom boundaries and labels
- Interaction features (A × B)
- Column renaming
- Full validation with duplicate name protection

### Exploratory Data Analysis
- Histograms with colour grouping
- Bar charts and pie charts for categoricals
- Box plots with optional grouping
- Scatter plots (sampled for performance)
- Correlation heatmap
- Grouped summary charts

### Before-and-After Comparison
- Side-by-side metrics: rows, columns, missing, duplicates, completeness
- Comparison bar charts
- Distribution overlays
- Column type change tracking
- Preprocessing log export

### Export
- Cleaned dataset (CSV)
- Preprocessing log (CSV)
- Data quality report (CSV)
- Data dictionary (CSV)
- Pipeline summary (JSON)
- Project summary (Markdown)

---

## Technology Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| Streamlit | 1.35.0 | Web application |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical operations |
| Plotly | 5.22.0 | Interactive charts |
| Matplotlib | 3.9.0 | Static charts (utility) |
| Seaborn | 0.13.2 | Statistical visualisation |
| Scikit-learn | 1.5.0 | Preprocessing utilities |
| Pytest | 8.2.2 | Automated testing |
| OpenPyXL | 3.1.4 | Excel file support |

---

## Project Structure

```
YR-NOVATECH-Task-2/
├── app.py                    ← Streamlit entry point (11 pages)
├── preprocessing.py          ← Cleaning & feature engineering engine
├── data_quality.py           ← Quality assessment module
├── visualizations.py         ← Plotly chart builders
├── utils.py                  ← Shared helpers, file I/O, session state
├── requirements.txt          ← Pinned runtime dependencies
├── README.md                 ← This file
├── .gitignore                ← Git exclusions
├── .streamlit/
│   └── config.toml           ← Light theme + server settings
├── data/
│   └── README.md             ← Dataset documentation
├── tests/
│   ├── test_preprocessing.py ← 50+ unit tests for preprocessing
│   ├── test_data_quality.py  ← Tests for quality module
│   └── test_utils.py         ← Tests for utility functions
└── screenshots/
    └── README.md             ← Screenshot guide
```

---

## Dataset

**UCI Adult Income**

| Property | Value |
|---|---|
| Source | https://archive.ics.uci.edu/ml/datasets/adult |
| Direct URL | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data |
| Rows | ~48,842 |
| Columns | 15 |
| License | Public domain / research use |

### Column Schema

| Column | Type | Description |
|---|---|---|
| age | int | Age |
| workclass | str | Employment type |
| fnlwgt | int | Census sampling weight |
| education | str | Education level |
| education_num | int | Education level (numeric) |
| marital_status | str | Marital status |
| occupation | str | Occupation |
| relationship | str | Relationship type |
| race | str | Race |
| sex | str | Sex |
| capital_gain | int | Capital gain |
| capital_loss | int | Capital loss |
| hours_per_week | int | Hours worked per week |
| native_country | str | Country of origin |
| income | str | Target: <=50K or >50K |

---

## Installation (Windows)

### Step 1 — Prerequisites

Make sure Python 3.11 is installed. Check with:
```
python --version
```

### Step 2 — Clone or Download the Repository

If you have Git:
```
git clone https://github.com/YOUR_USERNAME/YR-NOVATECH-Task-2.git
cd YR-NOVATECH-Task-2
```

Or download the ZIP from GitHub and extract it, then open a terminal in the project folder.

### Step 3 — Create a Virtual Environment

```
python -m venv .venv
```

### Step 4 — Activate the Virtual Environment

```
.venv\Scripts\activate
```

Your prompt should now show `(.venv)`.

### Step 5 — Install Dependencies

```
python -m pip install -r requirements.txt
```

This installs all required packages with pinned versions.

---

## Running the Application Locally

```
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

---

## Testing

Run all unit tests with:

```
python -m pytest -q
```

To see verbose output:
```
python -m pytest -v
```

To run a specific test file:
```
python -m pytest tests/test_preprocessing.py -v
```

Tests use small synthetic DataFrames only. They do not require a network connection or the actual UCI dataset.

---

## How to Use the Dashboard

### Upload a Dataset
1. Go to **📥 Data Collection** in the sidebar.
2. Click **Upload Your File** and select a CSV or Excel file.
   — OR —
3. Click **Download & Load UCI Adult Income Dataset** to load the built-in dataset.

### Inspect Quality
1. Go to **📋 Data Quality Report** to see missing values, duplicates, and statistics.

### Clean Data
1. Go to **🧹 Data Cleaning**.
2. Use the tabs to handle missing values, remove duplicates, fix data types, and normalise text.
3. Each operation requires an explicit click — nothing changes automatically.

### Analyse Outliers
1. Go to **📦 Outlier Analysis**.
2. Choose columns and a multiplier.
3. Review box plots, then choose to cap or remove outliers.

### Engineer Features
1. Go to **⚙️ Feature Engineering**.
2. Create arithmetic features, extract date parts, bin numerical columns, or make interaction features.

### Explore Data
1. Go to **📊 Exploratory Data Analysis**.
2. Use the tabs to create histograms, bar charts, scatter plots, and correlation heatmaps.

### Compare Before and After
1. Go to **🔄 Before-and-After Comparison** to see the impact of all your changes.

### Download Outputs
1. Go to **💾 Export & Documentation**.
2. Download the cleaned dataset, preprocessing log, quality report, or pipeline JSON.

---

## GitHub Upload

### One-time setup

```
git init
git add .
git commit -m "Initial commit — YR NOVATECH Task 2"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YR-NOVATECH-Task-2.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### For subsequent updates

```
git add .
git commit -m "Your commit message"
git push
```

---

## Streamlit Community Cloud Deployment

1. **Push your code to GitHub** (see above).
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Sign in with your GitHub account.
4. Click **New app**.
5. Fill in:
   - **Repository:** `YOUR_USERNAME/YR-NOVATECH-Task-2`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **Deploy**.
7. Wait 2–5 minutes for the build to complete.
8. Your app will be live at `https://YOUR_USERNAME-yr-novatech-task-2-app-XXXX.streamlit.app`.

**Notes:**
- No API keys or secrets are required.
- The app can start without a bundled dataset; users upload their own CSV or load the public dataset.
- If the UCI dataset download fails in the cloud, use the file upload option.

---

## Known Limitations

- The UCI Adult Income dataset download requires internet access. If it fails, the upload fallback works.
- Excel file support requires `openpyxl` (included in requirements).
- Very large files (>50 MB) are rejected to protect performance.
- The correlation heatmap works only for numerical columns.
- Date extraction requires the column to be explicitly parsed as datetime first.
- Streamlit's `st.download_button` reloads the page on click — this is expected Streamlit behaviour.
- The `infer_datetime_format` parameter used in date parsing may show a deprecation warning in newer pandas versions; it remains functional.

---

## Screenshots

See `screenshots/README.md` for a guide on which screenshots to capture.
Add captured screenshots to this folder and reference them here.

---

## Credits

- **Dataset:** UCI Machine Learning Repository — Adult Income dataset
  (Kohavi, R. and Becker, B., 1996)
- **Framework:** [Streamlit](https://streamlit.io)
- **Charts:** [Plotly](https://plotly.com/python/)
- **Organisation:** YR NOVATECH
- **Developed by:** [Your Name] — Internship Task 2

---

## License

This project code is provided for educational and internship demonstration purposes.
The UCI Adult Income dataset is used under its public domain / research use terms.
Please verify commercial use restrictions with the UCI ML Repository before any commercial application.
