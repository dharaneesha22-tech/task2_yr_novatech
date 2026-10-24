"""
app.py — Main Streamlit entry point for the YR NOVATECH
Data Collection & Preprocessing Dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Local modules
import data_quality as dq
import preprocessing as pp
import visualizations as viz
import utils

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Preprocessing Dashboard | YR NOVATECH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
utils.init_session_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Home Dashboard",
    "📥 Data Collection",
    "🔍 Data Preview",
    "📋 Data Quality Report",
    "🧹 Data Cleaning",
    "📦 Outlier Analysis",
    "⚙️ Feature Engineering",
    "📊 Exploratory Data Analysis",
    "🔄 Before-and-After Comparison",
    "💾 Export & Documentation",
    "🔎 Data Profiling",
    "🔍 Search & Filter",
    "💡 Smart Cleaning Suggestions",
    "ℹ️ About the Project",
]

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=64,
    )
    st.markdown("## 📊 YR NOVATECH")
    st.markdown("**Task 2 — Data Preprocessing**")
    st.markdown("---")
    selected_page = st.radio("Navigate to", PAGES, label_visibility="collapsed")
    st.markdown("---")

    # Dataset status indicator
    if st.session_state.original_df is not None:
        df_info = st.session_state.working_df
        st.success(
            f"✅ Dataset loaded\n\n"
            f"**{df_info.shape[0]:,}** rows × **{df_info.shape[1]}** cols"
        )
        # Quick column stats expander in sidebar
        with st.expander("📐 Quick Column Stats"):
            col_sel = st.selectbox(
                "Pick a column",
                df_info.columns.tolist(),
                key="sidebar_col_stat",
                label_visibility="collapsed",
            )
            s = df_info[col_sel]
            st.caption(f"**Type:** {s.dtype}")
            st.caption(f"**Non-null:** {s.notnull().sum():,}")
            st.caption(f"**Missing:** {s.isnull().sum():,}")
            st.caption(f"**Unique:** {s.nunique():,}")
            if pd.api.types.is_numeric_dtype(s):
                st.caption(f"**Mean:** {s.mean():.3f}")
                st.caption(f"**Std:** {s.std():.3f}")
                st.caption(f"**Min:** {s.min()}")
                st.caption(f"**Max:** {s.max()}")
            else:
                top = s.value_counts(dropna=False).index[0] if s.notnull().any() else "N/A"
                st.caption(f"**Top value:** {top}")

        if st.button("🗑️ Reset Dataset", use_container_width=True):
            utils.reset_session()
            st.rerun()
    else:
        st.info("No dataset loaded. Go to **Data Collection** to upload one.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def require_dataset(label: str = "this page") -> bool:
    """Show a prompt and return False if no dataset is loaded."""
    if st.session_state.original_df is None:
        st.warning(
            f"⚠️ No dataset is loaded. Please go to **📥 Data Collection** "
            f"to upload or download a dataset before using {label}."
        )
        return False
    return True


def get_working_df() -> pd.DataFrame:
    return st.session_state.working_df.copy()


def set_working_df(df: pd.DataFrame) -> None:
    st.session_state.working_df = df


# ===========================================================================
# PAGE 1 — Home Dashboard
# ===========================================================================

def page_home() -> None:
    st.title("📊 Data Collection & Preprocessing Dashboard")
    st.markdown(
        "#### An End-to-End Data Quality, Cleaning and Feature Engineering Platform"
    )
    st.markdown(
        "> **Organisation:** YR NOVATECH &nbsp;|&nbsp; "
        "**Internship Task:** Task 2 — Data Collection & Preprocessing"
    )
    st.markdown("---")

    if st.session_state.original_df is None:
        # Welcome screen
        st.markdown(
            """
            ### 👋 Welcome!

            This platform helps you collect, clean, analyse, and export tabular data
            for machine-learning and business analytics workflows.

            **Get started in three steps:**
            1. Go to **📥 Data Collection** in the sidebar.
            2. Upload your own CSV file **or** load the built-in sample dataset.
            3. Explore the pages to clean, visualise, and export your data.
            """
        )
        # Workflow diagram
        workflow_steps = [
            "Upload Dataset",
            "Inspect Quality",
            "Clean Data",
            "Engineer Features",
            "Visualise",
            "Export",
        ]
        completed = [False] * len(workflow_steps)
        st.plotly_chart(
            viz.workflow_progress_chart(workflow_steps, completed),
            use_container_width=True,
        )
        return

    orig = st.session_state.original_df
    work = st.session_state.working_df

    # --- Metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", f"{orig.shape[0]:,}")
    c2.metric("Columns", f"{orig.shape[1]}")
    c3.metric(
        "Missing Cells",
        f"{int(work.isnull().sum().sum()):,}",
        delta=f"{int(orig.isnull().sum().sum()) - int(work.isnull().sum().sum()):+,} vs original",
        delta_color="inverse",
    )
    c4.metric(
        "Duplicate Rows",
        f"{pp.count_duplicates(work):,}",
        delta=f"{pp.count_duplicates(orig) - pp.count_duplicates(work):+,} vs original",
        delta_color="inverse",
    )

    c5, c6, c7, c8 = st.columns(4)
    num_c = len(utils.numerical_cols(work))
    cat_c = len(utils.categorical_cols(work))
    c5.metric("Numerical Columns", f"{num_c}")
    c6.metric("Categorical Columns", f"{cat_c}")
    c7.metric("Data Completeness", f"{utils.completeness_pct(work):.1f}%")
    c8.metric("Preprocessing Ops", f"{st.session_state.operations_count}")

    st.markdown("---")

    # Workflow progress
    ops = st.session_state.operations_count
    workflow_steps = [
        "Upload Dataset",
        "Inspect Quality",
        "Clean Data",
        "Engineer Features",
        "Visualise",
        "Export",
    ]
    completed_flags = [
        True,  # uploaded
        True,  # quality checked (dataset is loaded implies they can inspect)
        ops > 0,
        ops > 0,
        ops >= 0,
        False,
    ]
    st.subheader("Workflow Progress")
    st.plotly_chart(
        viz.workflow_progress_chart(workflow_steps, completed_flags),
        use_container_width=True,
    )

    st.markdown("---")

    # Quick overview charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Missing Values by Column")
        st.plotly_chart(viz.missing_values_bar(work), use_container_width=True)

    with col_r:
        num_cols_list = utils.numerical_cols(work)
        if num_cols_list:
            first_num = num_cols_list[0]
            st.subheader(f"Distribution — {first_num}")
            st.plotly_chart(viz.histogram(work, first_num), use_container_width=True)
        else:
            st.info("No numerical columns available for a quick chart.")


# ===========================================================================
# PAGE 2 — Data Collection
# ===========================================================================

def page_data_collection() -> None:
    st.title("📥 Data Collection")
    st.markdown(
        "Upload your own CSV / Excel file, or load the built-in UCI Adult Income "
        "dataset (≥ 48,000 rows) directly from the UCI ML Repository."
    )
    st.markdown("---")

    tab_upload, tab_public = st.tabs(["📤 Upload Your File", "🌐 Load Public Dataset"])

    # --- Tab 1: Upload ---
    with tab_upload:
        st.markdown(
            f"Accepted formats: **CSV**, **XLSX**. Maximum size: **{utils.MAX_UPLOAD_MB} MB**."
        )
        uploaded = st.file_uploader(
            "Drop your file here or click Browse",
            type=["csv", "xlsx", "xls"],
            help="Your file is processed locally and never sent to any server.",
        )
        if uploaded is not None:
            with st.spinner("Validating and loading file…"):
                df, msg = utils.load_uploaded_file(uploaded)
            if df is None:
                st.error(f"❌ {msg}")
            else:
                st.success(f"✅ {msg}")
                st.session_state.original_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.dataset_name = uploaded.name
                st.session_state.dataset_source = "User upload"
                st.session_state.preprocessing_log = []
                st.session_state.operations_count = 0
                st.session_state.upload_validated = True
                st.rerun()

    # --- Tab 2: Public dataset ---
    with tab_public:
        st.markdown(
            """
            **UCI Adult Income Dataset**

            | Property | Detail |
            |----------|--------|
            | Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
            | Rows | ~48,842 |
            | Columns | 15 |
            | License | Public domain / research use |
            | Features | age, workclass, education, occupation, income, … |

            This dataset is used to predict whether a person earns more than $50K/year.
            It contains a rich mix of numerical and categorical features with realistic
            missing values — ideal for preprocessing demonstrations.
            """
        )
        if st.button("⬇️ Download & Load UCI Adult Income Dataset", use_container_width=True):
            with st.spinner("Downloading from UCI ML Repository…"):
                df, msg = utils.load_public_dataset()
            if df is None:
                st.error(f"❌ {msg}")
                st.markdown(
                    """
                    **Manual download instructions:**
                    1. Visit https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
                    2. Save the file as `adult.csv` in this project folder.
                    3. Upload it using the **Upload Your File** tab above.
                    """
                )
            else:
                st.success(f"✅ {msg}")
                st.session_state.original_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.dataset_name = "UCI Adult Income"
                st.session_state.dataset_source = utils.DATASET_URL
                st.session_state.preprocessing_log = []
                st.session_state.operations_count = 0
                st.session_state.upload_validated = True
                st.rerun()

    # Current dataset info
    if st.session_state.original_df is not None:
        st.markdown("---")
        st.subheader("Loaded Dataset")
        orig = st.session_state.original_df
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Rows", f"{orig.shape[0]:,}")
        ic2.metric("Columns", f"{orig.shape[1]}")
        ic3.metric("Memory", utils.df_memory_usage(orig))
        ic4.metric("Source", st.session_state.dataset_source[:30] + "…" if len(st.session_state.dataset_source) > 30 else st.session_state.dataset_source)


# ===========================================================================
# PAGE 3 — Data Preview
# ===========================================================================

def page_data_preview() -> None:
    st.title("🔍 Data Preview")
    if not require_dataset("Data Preview"):
        return

    work = get_working_df()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Sample Rows", "🗂️ Column Info", "📖 Data Dictionary", "🔢 Statistics"
    ])

    with tab1:
        n_rows = st.slider("Rows to display", 5, min(100, len(work)), 10)
        view = st.radio("View", ["Head", "Tail", "Random sample"], horizontal=True)
        if view == "Head":
            st.dataframe(work.head(n_rows), use_container_width=True)
        elif view == "Tail":
            st.dataframe(work.tail(n_rows), use_container_width=True)
        else:
            st.dataframe(work.sample(min(n_rows, len(work)), random_state=42), use_container_width=True)

    with tab2:
        info_df = pd.DataFrame({
            "Column": work.columns,
            "Data Type": work.dtypes.astype(str).values,
            "Non-Null Count": work.notnull().sum().values,
            "Null Count": work.isnull().sum().values,
            "Null %": (work.isnull().sum() / len(work) * 100).round(2).values,
            "Unique Values": work.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with tab3:
        dict_df = utils.build_data_dictionary(work)
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with tab4:
        num_cols = utils.numerical_cols(work)
        if num_cols:
            st.dataframe(work[num_cols].describe().T.round(4), use_container_width=True)
        else:
            st.info("No numerical columns for statistics.")


# ===========================================================================
# PAGE 4 — Data Quality Report
# ===========================================================================

def page_quality_report() -> None:
    st.title("📋 Data Quality Report")
    if not require_dataset("the Data Quality Report"):
        return

    work = get_working_df()
    report = dq.generate_quality_report(work)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{report['rows']:,}")
    c2.metric("Columns", f"{report['columns']}")
    c3.metric("Missing Cells", f"{report['missing_total']:,}")
    c4.metric("Duplicate Rows", f"{report['duplicate_rows']:,}")
    c5.metric("Completeness", f"{report['completeness_pct']}%")

    st.markdown("---")

    tab_miss, tab_cols, tab_num, tab_cat, tab_flags = st.tabs([
        "Missing Values", "Column Quality", "Numerical Summary",
        "Categorical Summary", "⚠️ Flags"
    ])

    with tab_miss:
        miss_df = dq.missingness_table(work)
        if miss_df.empty or miss_df["Missing Count"].sum() == 0:
            st.success("✅ No missing values detected.")
        else:
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            st.plotly_chart(viz.missing_values_bar(work), use_container_width=True)

    with tab_cols:
        col_q = dq.column_quality_table(work)
        st.dataframe(col_q, use_container_width=True, hide_index=True)

        if report["constant_cols"]:
            st.warning(f"⚠️ Constant columns (only one unique value): {report['constant_cols']}")
        if report["high_cardinality_cols"]:
            st.warning(
                f"⚠️ High-cardinality categorical columns (>50% unique): "
                f"{report['high_cardinality_cols']}"
            )

    with tab_num:
        if not report["numerical_summary"].empty:
            st.dataframe(report["numerical_summary"], use_container_width=True)
        else:
            st.info("No numerical columns.")

    with tab_cat:
        cat_cols = report["categorical_cols"]
        if cat_cols:
            chosen = st.selectbox("Select categorical column", cat_cols)
            if chosen in report["categorical_summary"]:
                st.dataframe(report["categorical_summary"][chosen], use_container_width=True, hide_index=True)
                st.plotly_chart(viz.bar_chart(work, chosen), use_container_width=True)
        else:
            st.info("No categorical columns.")

    with tab_flags:
        flags = report["suspicious_flags"]
        st.caption(
            "⚠️ Automated detection may produce false positives. "
            "Always verify flagged values manually before acting."
        )
        if flags:
            for col, col_flags in flags.items():
                with st.expander(f"Column: {col}"):
                    for f in col_flags:
                        st.warning(f)
        else:
            st.success("✅ No suspicious values automatically detected.")


# ===========================================================================
# PAGE 5 — Data Cleaning
# ===========================================================================

def page_data_cleaning() -> None:
    st.title("🧹 Data Cleaning")
    if not require_dataset("Data Cleaning"):
        return

    work = get_working_df()

    st.markdown(
        "Configure and apply cleaning operations below. "
        "Each operation is **optional** — nothing changes until you click Apply."
    )

    tab_missing, tab_dupes, tab_types, tab_text, tab_cols = st.tabs([
        "Missing Values", "Duplicates", "Data Types", "Text Normalisation", "Column Operations"
    ])

    # ---- A. Missing Values ----
    with tab_missing:
        st.subheader("A. Handle Missing Values")
        miss_df = dq.missingness_table(work)
        if miss_df["Missing Count"].sum() == 0:
            st.success("✅ No missing values in the current working dataset.")
        else:
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            num_cols = utils.numerical_cols(work)
            cat_cols = utils.categorical_cols(work)

            with st.expander("Numerical Imputation", expanded=True):
                num_miss_cols = [c for c in num_cols if work[c].isnull().sum() > 0]
                if not num_miss_cols:
                    st.info("No numerical columns with missing values.")
                else:
                    sel_num = st.multiselect("Columns to impute", num_miss_cols, default=num_miss_cols)
                    num_strategy = st.radio("Strategy", ["mean", "median", "constant"], horizontal=True)
                    num_const = None
                    if num_strategy == "constant":
                        num_const = st.number_input("Fill value", value=0.0)
                    if st.button("Apply Numerical Imputation", key="btn_num_imp"):
                        if sel_num:
                            before = len(work)
                            new_df, stats = pp.impute_numerical(work, sel_num, num_strategy, num_const)
                            filled = sum(v["filled"] for v in stats.values())
                            utils.log_operation(
                                "Numerical Imputation", sel_num,
                                {"strategy": num_strategy, "fill_value": num_const},
                                before, len(new_df),
                                f"Filled {filled} missing value(s) using {num_strategy}.",
                            )
                            set_working_df(new_df)
                            st.success(f"✅ Filled {filled} missing value(s).")
                            st.rerun()

            with st.expander("Categorical Imputation", expanded=True):
                cat_miss_cols = [c for c in cat_cols if work[c].isnull().sum() > 0]
                if not cat_miss_cols:
                    st.info("No categorical columns with missing values.")
                else:
                    sel_cat = st.multiselect("Columns to impute", cat_miss_cols, default=cat_miss_cols)
                    cat_strategy = st.radio("Strategy ", ["mode", "constant"], horizontal=True)
                    cat_const = None
                    if cat_strategy == "constant":
                        cat_const = st.text_input("Fill value", value="Unknown")
                    if st.button("Apply Categorical Imputation", key="btn_cat_imp"):
                        if sel_cat:
                            before = len(work)
                            new_df, stats = pp.impute_categorical(work, sel_cat, cat_strategy, cat_const or "Unknown")
                            filled = sum(v["filled"] for v in stats.values())
                            utils.log_operation(
                                "Categorical Imputation", sel_cat,
                                {"strategy": cat_strategy, "fill_value": cat_const},
                                before, len(new_df),
                                f"Filled {filled} missing value(s) using {cat_strategy}.",
                            )
                            set_working_df(new_df)
                            st.success(f"✅ Filled {filled} missing value(s).")
                            st.rerun()

            with st.expander("Drop Rows with Missing Values"):
                drop_miss_cols = st.multiselect(
                    "Drop rows that have NaN in these columns (leave blank = all columns)",
                    work.columns.tolist()
                )
                if st.button("Drop Rows", key="btn_drop_rows"):
                    before = len(work)
                    new_df, dropped = pp.drop_rows_with_missing(work, drop_miss_cols if drop_miss_cols else None)
                    utils.log_operation(
                        "Drop Rows with Missing", drop_miss_cols or list(work.columns),
                        {}, before, len(new_df),
                        f"Dropped {dropped} row(s) with missing values.",
                    )
                    set_working_df(new_df)
                    st.success(f"✅ Dropped {dropped} row(s).")
                    st.rerun()

    # ---- B. Duplicates ----
    with tab_dupes:
        st.subheader("B. Duplicate Rows")
        dup_count = pp.count_duplicates(work)
        if dup_count == 0:
            st.success("✅ No duplicate rows found.")
        else:
            st.warning(f"⚠️ Found **{dup_count:,}** duplicate row(s) ({dup_count/len(work)*100:.2f}% of data).")
            st.dataframe(
                work[work.duplicated()].head(10),
                use_container_width=True,
                help="Showing first 10 duplicate rows.",
            )
            if st.button("🗑️ Remove Duplicate Rows", key="btn_rm_dups"):
                before = len(work)
                new_df, removed = pp.remove_duplicates(work)
                utils.log_operation(
                    "Remove Duplicates", [],
                    {}, before, len(new_df),
                    f"Removed {removed} exact duplicate row(s).",
                )
                set_working_df(new_df)
                st.success(f"✅ Removed {removed} duplicate row(s).")
                st.rerun()

    # ---- C. Data Types ----
    with tab_types:
        st.subheader("C. Data Type Correction")
        obj_cols = utils.categorical_cols(work)

        with st.expander("Convert to Numeric"):
            to_numeric = st.multiselect("Select columns to convert to numeric", obj_cols)
            if st.button("Convert to Numeric", key="btn_to_num"):
                if to_numeric:
                    before = len(work)
                    new_df, results = pp.convert_to_numeric(work, to_numeric)
                    total_coerced = sum(v["coerced_nulls"] for v in results.values())
                    utils.log_operation(
                        "Convert to Numeric", to_numeric,
                        {}, before, len(new_df),
                        f"Converted {len(to_numeric)} column(s). {total_coerced} value(s) became NaN.",
                    )
                    set_working_df(new_df)
                    st.success(f"✅ Converted. {total_coerced} values could not be parsed (now NaN).")
                    st.rerun()

        with st.expander("Parse as Date / Time"):
            date_candidate = st.selectbox("Column to parse as date", ["(none)"] + work.columns.tolist())
            date_fmt = st.text_input("Date format (optional, e.g. %Y-%m-%d)", value="")
            if st.button("Parse Date Column", key="btn_parse_date"):
                if date_candidate != "(none)":
                    before = len(work)
                    new_df, errors, msg_dt = pp.parse_datetime_column(
                        work, date_candidate, fmt=date_fmt or None
                    )
                    utils.log_operation(
                        "Parse Datetime", [date_candidate],
                        {"format": date_fmt or "inferred"}, before, len(new_df), msg_dt,
                    )
                    set_working_df(new_df)
                    st.info(msg_dt)
                    st.rerun()

    # ---- D. Text Normalisation ----
    with tab_text:
        st.subheader("D. Text Normalisation")
        str_cols = utils.categorical_cols(work)
        if not str_cols:
            st.info("No string/object columns in current dataset.")
        else:
            sel_txt = st.multiselect("Columns to normalise", str_cols, default=str_cols)
            do_trim = st.checkbox("Trim leading/trailing whitespace", value=True)
            do_norm = st.checkbox("Collapse repeated whitespace", value=False)
            do_case = st.checkbox("Standardise casing", value=False)
            case_opt = "lower"
            if do_case:
                case_opt = st.radio("Case style", ["lower", "upper", "title"], horizontal=True)

            if st.button("Apply Text Normalisation", key="btn_text"):
                if sel_txt:
                    before = len(work)
                    new_df = work.copy()
                    ops_applied: list[str] = []
                    if do_trim:
                        new_df = pp.trim_whitespace(new_df, sel_txt)
                        ops_applied.append("trim")
                    if do_norm:
                        new_df = pp.normalize_whitespace(new_df, sel_txt)
                        ops_applied.append("normalize whitespace")
                    if do_case:
                        new_df = pp.standardize_casing(new_df, sel_txt, case_opt)
                        ops_applied.append(f"case={case_opt}")
                    utils.log_operation(
                        "Text Normalisation", sel_txt,
                        {"operations": ops_applied}, before, len(new_df),
                        f"Applied {ops_applied} to {len(sel_txt)} column(s).",
                    )
                    set_working_df(new_df)
                    st.success(f"✅ Text normalisation applied: {ops_applied}.")
                    st.rerun()

    # ---- E. Column Operations ----
    with tab_cols:
        st.subheader("E. Column Operations")
        with st.expander("Drop Columns", expanded=True):
            to_drop = st.multiselect("Select columns to drop", work.columns.tolist())
            if to_drop:
                st.warning(f"⚠️ This will permanently remove: {to_drop}")
            if st.button("Drop Selected Columns", key="btn_drop_cols"):
                if to_drop:
                    before = len(work)
                    new_df, dropped = pp.drop_columns(work, to_drop)
                    utils.log_operation(
                        "Drop Columns", dropped,
                        {}, before, len(new_df),
                        f"Dropped columns: {dropped}.",
                    )
                    set_working_df(new_df)
                    st.success(f"✅ Dropped: {dropped}.")
                    st.rerun()

        with st.expander("Rename a Column"):
            old_name = st.selectbox("Column to rename", work.columns.tolist(), key="rename_old")
            new_name = st.text_input("New name", key="rename_new")
            if st.button("Rename Column", key="btn_rename"):
                if new_name:
                    before = len(work)
                    new_df, msg_r = pp.rename_column(work, old_name, new_name)
                    if "already exists" in msg_r or "not found" in msg_r or "cannot be empty" in msg_r:
                        st.error(msg_r)
                    else:
                        utils.log_operation(
                            "Rename Column", [old_name],
                            {"new_name": new_name}, before, len(new_df), msg_r,
                        )
                        set_working_df(new_df)
                        st.success(f"✅ {msg_r}")
                        st.rerun()


# ===========================================================================
# PAGE 6 — Outlier Analysis
# ===========================================================================

def page_outlier_analysis() -> None:
    st.title("📦 Outlier Analysis")
    if not require_dataset("Outlier Analysis"):
        return

    work = get_working_df()
    num_cols = utils.numerical_cols(work)

    if not num_cols:
        st.warning("No numerical columns available for outlier analysis.")
        return

    st.markdown(
        "Outlier detection uses the **IQR method**. An outlier is any value "
        "below `Q1 − k×IQR` or above `Q3 + k×IQR` where k is the multiplier."
    )
    st.info(
        "⚠️ Not every statistical outlier is an error. Review results carefully before removing values."
    )

    multiplier = st.slider("IQR multiplier (k)", min_value=1.0, max_value=3.0, value=1.5, step=0.25)
    sel_cols = st.multiselect("Columns to analyse", num_cols, default=num_cols[:min(5, len(num_cols))])

    if not sel_cols:
        st.info("Select at least one column.")
        return

    outlier_info = pp.detect_outliers_iqr(work, sel_cols, multiplier)

    # Summary table
    summary_rows = []
    for col, info in outlier_info.items():
        summary_rows.append({
            "Column": col,
            "Q1": round(info["q1"], 4),
            "Q3": round(info["q3"], 4),
            "IQR": round(info["iqr"], 4),
            "Lower Bound": round(info["lower"], 4),
            "Upper Bound": round(info["upper"], 4),
            "Outlier Count": info["outlier_count"],
            "Outlier %": info["outlier_pct"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Box plots
    st.subheader("Box Plots")
    cols_per_row = 2
    for i in range(0, len(sel_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(sel_cols[i : i + cols_per_row]):
            with row_cols[j]:
                st.plotly_chart(viz.box_plot(work, col, title=f"Box Plot — {col}"), use_container_width=True)

    # Handling options
    st.markdown("---")
    st.subheader("Handle Outliers")
    handle_method = st.radio(
        "Method",
        ["Keep (no change)", "Cap (Winsorise)", "Remove rows"],
        horizontal=True,
    )
    if handle_method != "Keep (no change)":
        method_key = "cap" if "Cap" in handle_method else "remove"
        affected = sum(v["outlier_count"] for v in outlier_info.values())
        st.warning(
            f"This will affect up to **{affected:,}** value(s) / row(s) "
            f"across {len(sel_cols)} column(s)."
        )
        if st.button(f"Apply: {handle_method}", key="btn_outlier_handle"):
            before = len(work)
            new_df, stats = pp.handle_outliers(work, sel_cols, method_key, multiplier)
            summary_str = "; ".join(
                f"{c}: {list(s.values())}" for c, s in stats.items()
            )
            utils.log_operation(
                "Outlier Handling", sel_cols,
                {"method": method_key, "multiplier": multiplier},
                before, len(new_df),
                f"{handle_method} applied. {summary_str}",
            )
            set_working_df(new_df)
            st.success(f"✅ Outlier handling applied. Rows: {before:,} → {len(new_df):,}.")
            st.rerun()


# ===========================================================================
# PAGE 7 — Feature Engineering
# ===========================================================================

def page_feature_engineering() -> None:
    st.title("⚙️ Feature Engineering")
    if not require_dataset("Feature Engineering"):
        return

    work = get_working_df()
    num_cols = utils.numerical_cols(work)
    date_cols_list = utils.date_cols(work)

    tab_arith, tab_date, tab_bins, tab_interact = st.tabs([
        "Arithmetic Feature", "Date Extraction", "Binning", "Interaction Feature"
    ])

    # ---- Arithmetic ----
    with tab_arith:
        st.subheader("Create Arithmetic Feature")
        if len(num_cols) < 2:
            st.warning("At least 2 numerical columns are required.")
        else:
            col_a = st.selectbox("Column A", num_cols, key="fe_col_a")
            op_label = st.selectbox("Operation", list(pp.ALLOWED_ARITHMETIC_OPS.keys()))
            op_sym = pp.ALLOWED_ARITHMETIC_OPS[op_label]
            col_b = st.selectbox("Column B", [c for c in num_cols if c != col_a], key="fe_col_b")
            new_feat_name = st.text_input("New feature name", value=f"{col_a}_{op_sym}_{col_b}")
            drop_src = st.checkbox("Drop source columns after creation", value=False)
            if st.button("Create Arithmetic Feature", key="btn_arith"):
                if not new_feat_name.strip():
                    st.error("Feature name cannot be empty.")
                else:
                    before = len(work)
                    new_df, msg_fe = pp.create_arithmetic_feature(
                        work, col_a, col_b, op_sym, new_feat_name.strip(), drop_src
                    )
                    if "already exists" in msg_fe or "not found" in msg_fe or "Unsupported" in msg_fe:
                        st.error(msg_fe)
                    else:
                        utils.log_operation(
                            "Arithmetic Feature", [col_a, col_b],
                            {"operation": op_sym, "new_col": new_feat_name},
                            before, len(new_df), msg_fe,
                        )
                        set_working_df(new_df)
                        st.success(f"✅ {msg_fe}")
                        st.rerun()

    # ---- Date extraction ----
    with tab_date:
        st.subheader("Extract Date Features")
        if not date_cols_list:
            st.info(
                "No datetime columns detected. Go to **Data Cleaning → Data Types** "
                "to parse a column as a date first."
            )
        else:
            date_col = st.selectbox("Datetime column", date_cols_list)
            feat_options = ["year", "month", "day", "dayofweek", "hour", "quarter"]
            sel_feats = st.multiselect("Features to extract", feat_options, default=["year", "month", "day"])
            drop_src_dt = st.checkbox("Drop source column after extraction", value=False)
            if st.button("Extract Date Features", key="btn_date_feat"):
                if sel_feats:
                    before = len(work)
                    new_df, msg_dt = pp.extract_datetime_features(work, date_col, sel_feats, drop_src_dt)
                    utils.log_operation(
                        "Date Feature Extraction", [date_col],
                        {"features": sel_feats}, before, len(new_df), msg_dt,
                    )
                    set_working_df(new_df)
                    st.success(f"✅ {msg_dt}")
                    st.rerun()

    # ---- Binning ----
    with tab_bins:
        st.subheader("Bin a Numerical Column")
        if not num_cols:
            st.warning("No numerical columns available.")
        else:
            bin_col = st.selectbox("Column to bin", num_cols, key="bin_col")
            bin_mode = st.radio("Bin mode", ["Equal-width bins (count)", "Custom edges"], horizontal=True)
            new_bin_col = st.text_input("New binned column name", value=f"{bin_col}_bin")
            drop_src_bin = st.checkbox("Drop source column after binning", value=False)

            if bin_mode == "Equal-width bins (count)":
                n_bins = st.slider("Number of bins", 2, 20, 4)
                labels_input = st.text_input(
                    "Labels (comma-separated, optional)",
                    value="",
                    help="Leave blank for automatic labels.",
                )
                labels_list = [l.strip() for l in labels_input.split(",") if l.strip()] or None
                bins_val = n_bins
            else:
                edges_input = st.text_input(
                    "Bin edges (comma-separated numbers)",
                    value="0, 25, 50, 75, 100",
                )
                try:
                    edges = [float(e.strip()) for e in edges_input.split(",") if e.strip()]
                    bins_val = edges
                except ValueError:
                    st.error("Invalid bin edges. Enter numbers separated by commas.")
                    return
                n_intervals = len(bins_val) - 1 if isinstance(bins_val, list) else bins_val
                labels_input = st.text_input(
                    f"Labels ({n_intervals} intervals, comma-separated, optional)",
                    value="",
                )
                labels_list = [l.strip() for l in labels_input.split(",") if l.strip()] or None

            if st.button("Create Bins", key="btn_bins"):
                before = len(work)
                new_df, msg_bin = pp.create_bins(work, bin_col, bins_val, labels_list, new_bin_col.strip(), drop_src_bin)
                if "already exists" in msg_bin or "not found" in msg_bin or "failed" in msg_bin.lower():
                    st.error(msg_bin)
                else:
                    utils.log_operation(
                        "Binning", [bin_col],
                        {"bins": str(bins_val), "labels": labels_list},
                        before, len(new_df), msg_bin,
                    )
                    set_working_df(new_df)
                    st.success(f"✅ {msg_bin}")
                    st.rerun()

    # ---- Interaction ----
    with tab_interact:
        st.subheader("Create Interaction Feature (A × B)")
        if len(num_cols) < 2:
            st.warning("At least 2 numerical columns are required.")
        else:
            int_a = st.selectbox("Column A", num_cols, key="int_a")
            int_b = st.selectbox("Column B", [c for c in num_cols if c != int_a], key="int_b")
            int_name = st.text_input("Interaction feature name", value=f"{int_a}_x_{int_b}")
            drop_src_int = st.checkbox("Drop source columns", value=False, key="drop_src_int")
            if st.button("Create Interaction", key="btn_interact"):
                before = len(work)
                new_df, msg_int = pp.create_interaction_feature(work, int_a, int_b, int_name.strip(), drop_src_int)
                if "already exists" in msg_int or "not found" in msg_int:
                    st.error(msg_int)
                else:
                    utils.log_operation(
                        "Interaction Feature", [int_a, int_b],
                        {"new_col": int_name}, before, len(new_df), msg_int,
                    )
                    set_working_df(new_df)
                    st.success(f"✅ {msg_int}")
                    st.rerun()


# ===========================================================================
# PAGE 8 — Exploratory Data Analysis
# ===========================================================================

def page_eda() -> None:
    st.title("📊 Exploratory Data Analysis")
    if not require_dataset("Exploratory Data Analysis"):
        return

    work = get_working_df()
    num_cols = utils.numerical_cols(work)
    cat_cols = utils.categorical_cols(work)

    tab_hist, tab_bar, tab_box, tab_scatter, tab_corr, tab_group, tab_vc, tab_miss_heat = st.tabs([
        "Histogram", "Bar Chart", "Box Plot", "Scatter Plot",
        "Correlation Heatmap", "Grouped Summary", "Value Counts", "Missing Heatmap"
    ])

    with tab_hist:
        st.subheader("Numerical Histogram")
        if not num_cols:
            st.warning("No numerical columns.")
        else:
            h_col = st.selectbox("Column", num_cols, key="hist_col")
            h_color = st.selectbox("Colour by (optional)", ["(none)"] + cat_cols, key="hist_color")
            h_bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
            st.plotly_chart(
                viz.histogram(work, h_col, h_color if h_color != "(none)" else None, h_bins),
                use_container_width=True,
            )

    with tab_bar:
        st.subheader("Categorical Bar Chart")
        if not cat_cols:
            st.warning("No categorical columns.")
        else:
            b_col = st.selectbox("Column", cat_cols, key="bar_col")
            top_n = st.slider("Top N categories", 5, 30, 15, key="bar_top_n")
            st.plotly_chart(viz.bar_chart(work, b_col, top_n), use_container_width=True)
            st.plotly_chart(viz.pie_chart(work, b_col, top_n), use_container_width=True)

    with tab_box:
        st.subheader("Box Plot")
        if not num_cols:
            st.warning("No numerical columns.")
        else:
            bp_col = st.selectbox("Numerical column", num_cols, key="box_col")
            bp_group = st.selectbox("Group by (optional)", ["(none)"] + cat_cols, key="box_group")
            st.plotly_chart(
                viz.box_plot(work, bp_col, bp_group if bp_group != "(none)" else None),
                use_container_width=True,
            )

    with tab_scatter:
        st.subheader("Scatter Plot")
        if len(num_cols) < 2:
            st.warning("At least 2 numerical columns are needed for a scatter plot.")
        else:
            sc_x = st.selectbox("X axis", num_cols, key="sc_x")
            sc_y = st.selectbox("Y axis", [c for c in num_cols if c != sc_x], key="sc_y")
            sc_color = st.selectbox("Colour by", ["(none)"] + cat_cols, key="sc_color")
            sc_sample = st.slider("Max sample size", 500, 5000, 2000, key="sc_sample")
            st.plotly_chart(
                viz.scatter_plot(work, sc_x, sc_y, sc_color if sc_color != "(none)" else None, sc_sample),
                use_container_width=True,
            )

    with tab_corr:
        st.subheader("Correlation Heatmap")
        st.caption("Pearson correlation for numerical columns only.")
        st.plotly_chart(viz.correlation_heatmap(work), use_container_width=True)

    with tab_group:
        st.subheader("Grouped Summary")
        if not cat_cols or not num_cols:
            st.warning("Need both categorical and numerical columns.")
        else:
            g_x = st.selectbox("Group by (X axis)", cat_cols, key="grp_x")
            g_y = st.selectbox("Numerical column (Y axis)", num_cols, key="grp_y")
            g_agg = st.radio("Aggregation", ["mean", "median", "sum", "count"], horizontal=True)
            g_color = st.selectbox("Colour by (optional)", ["(none)"] + [c for c in cat_cols if c != g_x], key="grp_color")
            st.plotly_chart(
                viz.grouped_bar(work, g_x, g_y, g_color if g_color != "(none)" else None, g_agg),
                use_container_width=True,
            )

    with tab_vc:
        st.subheader("Value Counts Drill-Down")
        vc_col = st.selectbox("Select column", work.columns.tolist(), key="vc_col")
        vc_top = st.slider("Top N values", 5, 50, 20, key="vc_top")
        vc_drop_na = st.checkbox("Exclude missing values", value=False, key="vc_na")
        vc_series = work[vc_col].value_counts(dropna=vc_drop_na).head(vc_top)
        vc_df = vc_series.reset_index()
        vc_df.columns = ["Value", "Count"]
        vc_df["Percentage"] = (vc_df["Count"] / len(work) * 100).round(2)
        vc_df["Cumulative %"] = vc_df["Percentage"].cumsum().round(2)

        col_tbl, col_chart = st.columns([1, 2])
        with col_tbl:
            st.dataframe(vc_df, use_container_width=True, hide_index=True)
        with col_chart:
            import plotly.express as px
            fig_vc = px.bar(
                vc_df, x="Count", y="Value", orientation="h",
                title=f"Value Counts — {vc_col}",
                template="plotly_white",
                color="Percentage",
                color_continuous_scale="Blues",
            )
            fig_vc.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_vc, use_container_width=True)

    with tab_miss_heat:
        st.subheader("Missing Value Pattern Heatmap")
        st.caption("Red = missing. Up to 200 rows sampled for performance.")
        if work.isnull().sum().sum() == 0:
            st.success("✅ No missing values in the current dataset.")
        else:
            st.plotly_chart(viz.missing_values_heatmap(work, max_rows=200), use_container_width=True)


# ===========================================================================
# PAGE 9 — Before-and-After Comparison
# ===========================================================================

def page_comparison() -> None:
    st.title("🔄 Before-and-After Comparison")
    if not require_dataset("the Comparison page"):
        return

    orig = st.session_state.original_df
    work = get_working_df()

    # --- Key metrics side-by-side ---
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🗂️ Original")
        st.metric("Rows", f"{orig.shape[0]:,}")
        st.metric("Columns", f"{orig.shape[1]}")
        st.metric("Missing Cells", f"{int(orig.isnull().sum().sum()):,}")
        st.metric("Duplicate Rows", f"{pp.count_duplicates(orig):,}")
        st.metric("Completeness", f"{utils.completeness_pct(orig):.2f}%")
    with col2:
        st.markdown("#### ✅ Cleaned")
        st.metric("Rows", f"{work.shape[0]:,}", delta=f"{work.shape[0] - orig.shape[0]:+,}")
        st.metric("Columns", f"{work.shape[1]}", delta=f"{work.shape[1] - orig.shape[1]:+,}")
        st.metric("Missing Cells", f"{int(work.isnull().sum().sum()):,}",
                  delta=f"{int(work.isnull().sum().sum()) - int(orig.isnull().sum().sum()):+,}",
                  delta_color="inverse")
        st.metric("Duplicate Rows", f"{pp.count_duplicates(work):,}",
                  delta=f"{pp.count_duplicates(work) - pp.count_duplicates(orig):+,}",
                  delta_color="inverse")
        st.metric("Completeness", f"{utils.completeness_pct(work):.2f}%")

    st.markdown("---")

    if st.session_state.operations_count == 0:
        st.info("ℹ️ No preprocessing operations have been applied yet. The dataset is unchanged.")
        return

    # Comparison bar chart
    labels = ["Rows", "Missing Cells", "Duplicate Rows"]
    before_vals = [orig.shape[0], int(orig.isnull().sum().sum()), pp.count_duplicates(orig)]
    after_vals = [work.shape[0], int(work.isnull().sum().sum()), pp.count_duplicates(work)]
    st.plotly_chart(
        viz.comparison_bar(labels, before_vals, after_vals, "Before vs After — Key Metrics"),
        use_container_width=True,
    )

    # Per-column missing comparison
    st.subheader("Missing Values: Before vs After")
    orig_miss = orig.isnull().sum().rename("Original Missing")
    work_miss = work.isnull().sum().rename("Cleaned Missing")
    miss_compare = pd.concat([orig_miss, work_miss], axis=1).fillna(0).astype(int)
    miss_compare = miss_compare[(miss_compare["Original Missing"] > 0) | (miss_compare["Cleaned Missing"] > 0)]
    if miss_compare.empty:
        st.success("✅ No missing values in either the original or cleaned dataset.")
    else:
        st.dataframe(miss_compare, use_container_width=True)

    # Distribution comparison for numerical columns
    common_num = [c for c in utils.numerical_cols(orig) if c in utils.numerical_cols(work)]
    if common_num:
        st.subheader("Distribution Comparison")
        dist_col = st.selectbox("Select column", common_num)
        st.plotly_chart(
            viz.distribution_comparison(orig, work, dist_col),
            use_container_width=True,
        )

    # Column type changes
    st.subheader("Column Data Types")
    orig_types = orig.dtypes.rename("Original Type").astype(str)
    work_types = work.dtypes.rename("Cleaned Type").astype(str)
    type_df = pd.concat([orig_types, work_types], axis=1)
    type_df["Changed"] = type_df["Original Type"] != type_df["Cleaned Type"]
    st.dataframe(type_df, use_container_width=True)

    # Preprocessing log
    st.subheader("Preprocessing Log")
    log_df = utils.get_log_df()
    if log_df.empty:
        st.info("No operations logged.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 10 — Export & Documentation
# ===========================================================================

def page_export() -> None:
    st.title("💾 Export & Documentation")
    if not require_dataset("Export"):
        return

    work = get_working_df()
    orig = st.session_state.original_df
    log_df = utils.get_log_df()

    st.subheader("Final Dataset Summary")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Rows", f"{work.shape[0]:,}")
    ec2.metric("Columns", f"{work.shape[1]}")
    ec3.metric("Operations Applied", f"{st.session_state.operations_count}")
    ec4.metric("Completeness", f"{utils.completeness_pct(work):.1f}%")

    st.markdown("---")
    st.subheader("Download Files")

    d1, d2 = st.columns(2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with d1:
        st.markdown("**Cleaned Dataset (CSV)**")
        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=utils.df_to_csv_bytes(work),
            file_name=f"cleaned_dataset_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Preprocessing Log (CSV)**")
        st.download_button(
            "⬇️ Download Preprocessing Log",
            data=utils.df_to_csv_bytes(log_df) if not log_df.empty else b"No operations logged.",
            file_name=f"preprocessing_log_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Data Dictionary (CSV)**")
        dict_df = utils.build_data_dictionary(work)
        st.download_button(
            "⬇️ Download Data Dictionary",
            data=utils.df_to_csv_bytes(dict_df),
            file_name=f"data_dictionary_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.markdown("**Data Quality Report (CSV)**")
        report = dq.generate_quality_report(work)
        miss_df = dq.missingness_table(work)
        st.download_button(
            "⬇️ Download Quality Report",
            data=utils.df_to_csv_bytes(miss_df),
            file_name=f"quality_report_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Pipeline Summary (JSON)**")
        pipeline_json = utils.build_pipeline_json(work, st.session_state.preprocessing_log)
        st.download_button(
            "⬇️ Download Pipeline JSON",
            data=pipeline_json.encode("utf-8"),
            file_name=f"pipeline_summary_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("**Cleaned Dataset (Excel)**")
        try:
            excel_bytes = utils.df_to_excel_bytes(work)
            st.download_button(
                "⬇️ Download Cleaned Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"cleaned_dataset_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.info("Excel export requires openpyxl. Install with: pip install openpyxl")

        st.markdown("**Full Report Workbook (Excel)**")
        try:
            workbook_bytes = utils.df_to_excel_multi_bytes({
                "Cleaned Data": work,
                "Preprocessing Log": log_df if not log_df.empty else pd.DataFrame({"note": ["No operations logged"]}),
                "Data Dictionary": utils.build_data_dictionary(work),
                "Quality Report": dq.missingness_table(work),
            })
            st.download_button(
                "⬇️ Download Full Report Workbook (.xlsx)",
                data=workbook_bytes,
                file_name=f"full_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"Multi-sheet Excel export unavailable: {exc}")

        st.markdown("**Project Summary (Markdown)**")
        summary_md = f"""# YR NOVATECH — Task 2 Export Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- Source: {st.session_state.dataset_source}
- Original shape: {orig.shape[0]:,} rows × {orig.shape[1]} columns
- Final shape: {work.shape[0]:,} rows × {work.shape[1]} columns
- Operations applied: {st.session_state.operations_count}
- Completeness after cleaning: {utils.completeness_pct(work):.2f}%

## Preprocessing Steps
{log_df.to_markdown(index=False) if not log_df.empty else 'No operations applied.'}
"""
        st.download_button(
            "⬇️ Download Project Summary (MD)",
            data=summary_md.encode("utf-8"),
            file_name=f"project_summary_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Preview: Cleaned Dataset (first 20 rows)")
    st.dataframe(work.head(20), use_container_width=True)

    if not log_df.empty:
        st.subheader("Preprocessing Log Preview")
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 11 — About
# ===========================================================================

def page_about() -> None:
    st.title("ℹ️ About the Project")
    st.markdown(
        """
        ## Data Collection & Preprocessing Dashboard

        | | |
        |---|---|
        | **Organisation** | YR NOVATECH |
        | **Internship Task** | Task 2 — Data Collection & Preprocessing |
        | **Framework** | Streamlit |
        | **Language** | Python 3.11 |
        | **Deployment** | Streamlit Community Cloud |

        ---

        ### Features
        - Upload CSV / Excel or load the built-in UCI Adult Income dataset
        - Comprehensive data quality assessment
        - Configurable missing-value imputation (mean / median / mode / constant)
        - Duplicate detection and removal
        - Data type correction and date parsing
        - Text normalisation
        - IQR-based outlier detection with cap / remove options
        - Arithmetic, date, binning, and interaction feature engineering
        - Interactive EDA (histograms, scatter, correlation heatmap, grouped summaries)
        - Before-and-after comparison with visual diff
        - Downloadable CSV, JSON, and Markdown exports
        - Preprocessing log for reproducibility

        ---

        ### Dataset
        **UCI Adult Income**
        - Source: https://archive.ics.uci.edu/ml/datasets/adult
        - Rows: ~48,842 | Columns: 15
        - Mix of numerical and categorical features
        - Contains realistic missing values (coded as "?")

        ---

        ### Technology Stack
        | Library | Purpose |
        |---------|---------|
        | Streamlit | Web application framework |
        | Pandas | Data manipulation |
        | NumPy | Numerical operations |
        | Plotly | Interactive visualisations |
        | Scikit-learn | Preprocessing utilities |
        | Pytest | Automated testing |

        ---

        ### Deployment
        1. Push this repository to GitHub.
        2. Go to [share.streamlit.io](https://share.streamlit.io).
        3. Connect your GitHub account.
        4. Select this repository, branch `main`, and `app.py`.
        5. Click **Deploy**.

        ---

        *Developed for YR NOVATECH Internship Task 2.*
        """
    )


# ===========================================================================
# PAGE 12 — Data Profiling (full column-by-column deep dive)
# ===========================================================================

def page_data_profiling() -> None:
    st.title("🔎 Data Profiling")
    st.markdown("Deep per-column statistics, distribution shapes, and value inventory.")
    if not require_dataset("Data Profiling"):
        return

    work = get_working_df()

    # Summary bar at top
    num_cols = utils.numerical_cols(work)
    cat_cols = utils.categorical_cols(work)
    date_cols_list = utils.date_cols(work)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Columns", work.shape[1])
    m2.metric("Numerical", len(num_cols))
    m3.metric("Categorical", len(cat_cols))
    m4.metric("Datetime", len(date_cols_list))
    m5.metric("Rows", f"{work.shape[0]:,}")

    st.markdown("---")

    # Missing-value heatmap (pattern)
    st.subheader("Missing Value Pattern")
    st.caption("Red cells = missing. Up to 200 rows sampled for display.")
    if work.isnull().sum().sum() == 0:
        st.success("✅ No missing values — heatmap not needed.")
    else:
        st.plotly_chart(viz.missing_values_heatmap(work, max_rows=200), use_container_width=True)

    st.markdown("---")

    # Per-column deep dive
    st.subheader("Column-by-Column Profile")
    chosen_col = st.selectbox("Select column to profile", work.columns.tolist())
    s = work[chosen_col]
    rows = len(work)

    left, right = st.columns([1, 2])

    with left:
        st.markdown(f"**Column:** `{chosen_col}`")
        st.markdown(f"**Type:** `{s.dtype}`")
        st.markdown(f"**Non-null:** {s.notnull().sum():,} ({s.notnull().sum()/rows*100:.1f}%)")
        st.markdown(f"**Missing:** {s.isnull().sum():,} ({s.isnull().sum()/rows*100:.2f}%)")
        st.markdown(f"**Unique values:** {s.nunique():,}")
        st.markdown(f"**Constant column:** {'Yes ⚠️' if s.nunique() <= 1 else 'No'}")

        if pd.api.types.is_numeric_dtype(s):
            st.markdown("---")
            st.markdown("**Numerical Statistics**")
            stats = s.describe()
            for k, v in stats.items():
                st.markdown(f"- **{k}:** {v:.4f}")
            st.markdown(f"- **skewness:** {s.skew():.4f}")
            st.markdown(f"- **kurtosis:** {s.kurtosis():.4f}")
            # IQR outlier count
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            st.markdown(f"- **IQR outliers:** {outliers:,}")
        else:
            st.markdown("---")
            st.markdown("**Top 5 Values**")
            vc = s.value_counts(dropna=False).head(5)
            for val, cnt in vc.items():
                st.markdown(f"- `{val}` → {cnt:,} ({cnt/rows*100:.1f}%)")

    with right:
        if pd.api.types.is_numeric_dtype(s):
            tab_h, tab_b = st.tabs(["Histogram", "Box Plot"])
            with tab_h:
                st.plotly_chart(viz.histogram(work, chosen_col), use_container_width=True)
            with tab_b:
                st.plotly_chart(viz.box_plot(work, chosen_col), use_container_width=True)
        else:
            tab_bar, tab_pie = st.tabs(["Bar Chart", "Pie Chart"])
            with tab_bar:
                st.plotly_chart(viz.bar_chart(work, chosen_col, top_n=15), use_container_width=True)
            with tab_pie:
                st.plotly_chart(viz.pie_chart(work, chosen_col, top_n=10), use_container_width=True)

    # Full value counts table
    st.markdown("---")
    st.subheader(f"Value Counts — {chosen_col}")
    show_n = st.slider("Show top N values", 5, 100, 20)
    vc_df = s.value_counts(dropna=False).head(show_n).reset_index()
    vc_df.columns = ["Value", "Count"]
    vc_df["Percentage"] = (vc_df["Count"] / rows * 100).round(2)
    vc_df["Cumulative %"] = vc_df["Percentage"].cumsum().round(2)
    st.dataframe(vc_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 13 — Search & Filter
# ===========================================================================

def page_search_filter() -> None:
    st.title("🔍 Search & Filter")
    st.markdown(
        "Filter rows interactively by column values. "
        "The filtered view is read-only — apply changes in **Data Cleaning** if needed."
    )
    if not require_dataset("Search & Filter"):
        return

    work = get_working_df()

    st.subheader("Active Filters")
    filter_cols = st.multiselect(
        "Add filter columns", work.columns.tolist(), key="sf_cols"
    )

    filtered = work.copy()
    filter_descriptions: list[str] = []

    for col in filter_cols:
        s = work[col]
        if pd.api.types.is_numeric_dtype(s):
            col_min = float(s.min()) if not s.isnull().all() else 0.0
            col_max = float(s.max()) if not s.isnull().all() else 1.0
            if col_min == col_max:
                col_max = col_min + 1.0
            low, high = st.slider(
                f"{col} range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f"sf_{col}",
            )
            filtered = filtered[(filtered[col] >= low) & (filtered[col] <= high) | filtered[col].isnull()]
            filter_descriptions.append(f"{col}: {low} – {high}")
        else:
            options = s.dropna().unique().tolist()
            if len(options) > 100:
                keyword = st.text_input(f"Search in {col} (contains)", key=f"sf_text_{col}")
                if keyword:
                    filtered = filtered[filtered[col].astype(str).str.contains(keyword, case=False, na=False)]
                    filter_descriptions.append(f"{col} contains '{keyword}'")
            else:
                chosen = st.multiselect(
                    f"{col} values",
                    options,
                    default=options,
                    key=f"sf_cat_{col}",
                )
                filtered = filtered[filtered[col].isin(chosen) | filtered[col].isnull()]
                filter_descriptions.append(f"{col} in {chosen[:3]}{'…' if len(chosen) > 3 else ''}")

    # Global text search
    st.markdown("---")
    global_search = st.text_input(
        "🔎 Global text search (searches all string columns)",
        key="sf_global",
        placeholder="Type to search across all text columns…",
    )
    if global_search:
        str_cols = utils.categorical_cols(filtered)
        if str_cols:
            mask = filtered[str_cols].apply(
                lambda col: col.astype(str).str.contains(global_search, case=False, na=False)
            ).any(axis=1)
            filtered = filtered[mask]
            filter_descriptions.append(f"Global search: '{global_search}'")

    # Results
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Matching Rows", f"{len(filtered):,}")
    col_b.metric("Total Rows", f"{len(work):,}")
    col_c.metric("Match %", f"{len(filtered)/len(work)*100:.1f}%" if len(work) > 0 else "0%")

    if filter_descriptions:
        st.info("**Active filters:** " + " | ".join(filter_descriptions))

    if filtered.empty:
        st.warning("No rows match the current filters.")
    else:
        rows_to_show = st.slider("Rows to display", 10, min(500, len(filtered)), 50)
        st.dataframe(filtered.head(rows_to_show), use_container_width=True)

        # Download filtered result
        st.download_button(
            "⬇️ Download Filtered Dataset (CSV)",
            data=utils.df_to_csv_bytes(filtered),
            file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ===========================================================================
# PAGE 14 — Smart Cleaning Suggestions
# ===========================================================================

def page_smart_suggestions() -> None:
    st.title("💡 Smart Cleaning Suggestions")
    st.markdown(
        "Automated analysis of the current working dataset with actionable suggestions. "
        "**All suggestions are advisory — nothing is applied automatically.**"
    )
    if not require_dataset("Smart Suggestions"):
        return

    work = get_working_df()
    report = dq.generate_quality_report(work)
    rows = report["rows"]

    suggestions: list[dict] = []

    # --- Missing values ---
    for col, count in report["missing_per_column"].items():
        if count == 0:
            continue
        pct = report["missing_pct_per_column"][col]
        dtype = report["dtypes"][col]
        if pct >= 60:
            suggestions.append({
                "priority": "🔴 High",
                "column": col,
                "issue": f"{pct:.1f}% missing",
                "suggestion": f"Consider dropping column '{col}' — more than half the data is missing.",
                "action_page": "🧹 Data Cleaning → Column Operations → Drop Columns",
            })
        elif pct >= 20:
            strat = "mean/median" if "float" in dtype or "int" in dtype else "mode or 'Unknown'"
            suggestions.append({
                "priority": "🟠 Medium",
                "column": col,
                "issue": f"{pct:.1f}% missing",
                "suggestion": f"Impute with {strat} — moderate missingness.",
                "action_page": "🧹 Data Cleaning → Missing Values",
            })
        else:
            strat = "mean/median" if "float" in dtype or "int" in dtype else "mode"
            suggestions.append({
                "priority": "🟡 Low",
                "column": col,
                "issue": f"{pct:.1f}% missing",
                "suggestion": f"Impute with {strat} — small amount of missing data.",
                "action_page": "🧹 Data Cleaning → Missing Values",
            })

    # --- Duplicates ---
    if report["duplicate_rows"] > 0:
        dup_pct = report["duplicate_rows"] / rows * 100 if rows else 0
        suggestions.append({
            "priority": "🟠 Medium" if dup_pct > 1 else "🟡 Low",
            "column": "—",
            "issue": f"{report['duplicate_rows']:,} duplicate rows ({dup_pct:.2f}%)",
            "suggestion": "Remove exact duplicate rows to avoid skewing statistics.",
            "action_page": "🧹 Data Cleaning → Duplicates",
        })

    # --- Constant columns ---
    for col in report["constant_cols"]:
        suggestions.append({
            "priority": "🟠 Medium",
            "column": col,
            "issue": "Constant — only 1 unique value",
            "suggestion": f"Drop '{col}' — constant columns carry no information for modelling.",
            "action_page": "🧹 Data Cleaning → Column Operations → Drop Columns",
        })

    # --- High cardinality ---
    for col in report["high_cardinality_cols"]:
        suggestions.append({
            "priority": "🟡 Low",
            "column": col,
            "issue": f"High cardinality ({report['unique_per_column'][col]:,} unique values)",
            "suggestion": f"Consider encoding, grouping rare values, or dropping '{col}' if it is an ID.",
            "action_page": "⚙️ Feature Engineering → Binning",
        })

    # --- Outliers ---
    num_cols = report["numerical_cols"]
    if num_cols:
        outlier_info = pp.detect_outliers_iqr(work, num_cols)
        for col, info in outlier_info.items():
            if info["outlier_pct"] > 5:
                suggestions.append({
                    "priority": "🟠 Medium",
                    "column": col,
                    "issue": f"{info['outlier_count']:,} outliers ({info['outlier_pct']:.1f}%)",
                    "suggestion": f"Review outliers in '{col}'. Consider capping (winsorising) if they are data errors.",
                    "action_page": "📦 Outlier Analysis",
                })
            elif info["outlier_pct"] > 1:
                suggestions.append({
                    "priority": "🟡 Low",
                    "column": col,
                    "issue": f"{info['outlier_count']:,} outliers ({info['outlier_pct']:.1f}%)",
                    "suggestion": f"Minor outliers in '{col}'. Inspect box plot before deciding to cap or keep.",
                    "action_page": "📦 Outlier Analysis",
                })

    # --- Object columns that may be numeric ---
    for col in report["categorical_cols"]:
        sample = work[col].dropna().head(50)
        try:
            pd.to_numeric(sample)
            suggestions.append({
                "priority": "🟡 Low",
                "column": col,
                "issue": "Stored as text but may be numeric",
                "suggestion": f"Convert '{col}' to numeric — values look like numbers.",
                "action_page": "🧹 Data Cleaning → Data Types → Convert to Numeric",
            })
        except (ValueError, TypeError):
            pass

    # --- Suspicious flags from quality report ---
    for col, flags in report["suspicious_flags"].items():
        for flag in flags:
            suggestions.append({
                "priority": "🟡 Low",
                "column": col,
                "issue": "Suspicious value pattern",
                "suggestion": f"{flag} — review manually.",
                "action_page": "📋 Data Quality Report → ⚠️ Flags",
            })

    # Display
    if not suggestions:
        st.success("✅ No issues detected. The dataset looks clean!")
        return

    # Summary counts
    high = sum(1 for s in suggestions if "High" in s["priority"])
    med  = sum(1 for s in suggestions if "Medium" in s["priority"])
    low  = sum(1 for s in suggestions if "Low" in s["priority"])

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Suggestions", len(suggestions))
    sc2.metric("🔴 High Priority", high)
    sc3.metric("🟠 Medium Priority", med)
    sc4.metric("🟡 Low Priority", low)

    st.markdown("---")

    # Filter by priority
    show_priority = st.multiselect(
        "Show priorities",
        ["🔴 High", "🟠 Medium", "🟡 Low"],
        default=["🔴 High", "🟠 Medium", "🟡 Low"],
    )
    filtered_suggestions = [s for s in suggestions if s["priority"] in show_priority]

    for i, s in enumerate(filtered_suggestions, 1):
        with st.container():
            col_p, col_c, col_i = st.columns([1, 2, 5])
            col_p.markdown(s["priority"])
            col_c.markdown(f"**{s['column']}**")
            col_i.markdown(f"**Issue:** {s['issue']}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;💬 {s['suggestion']}")
            st.caption(f"→ Go to: **{s['action_page']}**")
            st.markdown("---")

    # Export suggestions
    suggestions_df = pd.DataFrame(filtered_suggestions)
    st.download_button(
        "⬇️ Download Suggestions (CSV)",
        data=utils.df_to_csv_bytes(suggestions_df),
        file_name=f"cleaning_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ===========================================================================
# Router
# ===========================================================================

PAGE_MAP = {
    "🏠 Home Dashboard": page_home,
    "📥 Data Collection": page_data_collection,
    "🔍 Data Preview": page_data_preview,
    "📋 Data Quality Report": page_quality_report,
    "🧹 Data Cleaning": page_data_cleaning,
    "📦 Outlier Analysis": page_outlier_analysis,
    "⚙️ Feature Engineering": page_feature_engineering,
    "📊 Exploratory Data Analysis": page_eda,
    "🔄 Before-and-After Comparison": page_comparison,
    "💾 Export & Documentation": page_export,
    "🔎 Data Profiling": page_data_profiling,
    "🔍 Search & Filter": page_search_filter,
    "💡 Smart Cleaning Suggestions": page_smart_suggestions,
    "ℹ️ About the Project": page_about,
}

PAGE_MAP[selected_page]()
