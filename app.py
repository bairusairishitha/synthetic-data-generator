"""
Synthetic Data Generator
-------------------------
A simple Streamlit app that lets a user upload a CSV file and generates a
synthetic version of it that preserves the overall statistical properties
of the original data, without containing exact copies of the original rows.

Generation strategy:
1. Preferred  -> Gaussian Copula (via the `copulas` library)
2. Fallback   -> Random sampling with added noise for numeric columns and
                 random sampling (respecting value frequencies) for
                 categorical columns. This fallback is used automatically
                 if the `copulas` library is not installed or if the
                 Gaussian Copula fails to fit (e.g. too few rows).
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# Try to import the copulas library. If it is not available, we will
# fall back to the simpler random-sampling-with-noise method further down.
try:
    from copulas.multivariate import GaussianMultivariate
    COPULAS_AVAILABLE = True
except ImportError:
    COPULAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_csv(uploaded_file):
    """Read an uploaded CSV file into a pandas DataFrame.

    Returns (dataframe, error_message). If loading fails, dataframe is None
    and error_message contains a human-readable explanation.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"Could not read the CSV file. Details: {exc}"

    if df.empty:
        return None, "The uploaded CSV file is empty."

    if df.shape[1] == 0:
        return None, "The uploaded CSV file has no columns."

    return df, None


def get_dataset_summary(df):
    """Return a small summary dataframe with column names, dtypes and
    missing-value counts, used for the 'dataset overview' section."""
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
    })
    return summary


def split_columns(df):
    """Split columns into numeric and categorical (everything else)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def remove_exact_duplicates(synthetic_df, original_df):
    """Drop any synthetic rows that are exact copies of a row in the
    original dataset, so the synthetic data never leaks real records."""
    merged = synthetic_df.merge(
        original_df.drop_duplicates(),
        how="left",
        indicator=True,
    )
    cleaned = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return cleaned.reset_index(drop=True)


def generate_with_gaussian_copula(df, num_rows, numeric_cols, categorical_cols):
    """Generate synthetic data using a Gaussian Copula model.

    Categorical columns are label-encoded (as integer codes) before fitting,
    because the copulas library models continuous distributions. After
    sampling, the encoded categorical columns are rounded and mapped back
    to their original category labels.
    """
    working_df = df.copy()
    category_maps = {}

    # Encode categorical columns as integer codes so the copula can model them.
    for col in categorical_cols:
        working_df[col] = working_df[col].astype("category")
        category_maps[col] = dict(enumerate(working_df[col].cat.categories))
        working_df[col] = working_df[col].cat.codes.replace(-1, np.nan)

    # Fill any missing numeric values with the column median so the model
    # can be fit (copulas does not handle NaNs).
    for col in working_df.columns:
        if working_df[col].isnull().any():
            working_df[col] = working_df[col].fillna(working_df[col].median())

    # Fit the Gaussian Copula model on the (fully numeric) working data.
    model = GaussianMultivariate()
    model.fit(working_df)

    # Sample new synthetic rows.
    synthetic = model.sample(num_rows)

    # Decode categorical columns back to their original labels.
    for col in categorical_cols:
        n_categories = len(category_maps[col])
        codes = synthetic[col].round().clip(0, max(n_categories - 1, 0)).astype(int)
        synthetic[col] = codes.map(category_maps[col])

    # Round numeric integer-like columns back to sensible values.
    for col in numeric_cols:
        if pd.api.types.is_integer_dtype(df[col]):
            synthetic[col] = synthetic[col].round().astype(df[col].dtype, errors="ignore")

    # Keep the original column order.
    synthetic = synthetic[df.columns.tolist()]
    return synthetic


def generate_with_random_noise(df, num_rows, numeric_cols, categorical_cols):
    """Fallback generator: random sampling with noise.

    * Numeric columns: sample existing values with replacement, then add a
      small amount of Gaussian noise (proportional to the column's standard
      deviation) so values are not exact copies of the originals.
    * Categorical columns: sample existing values with replacement,
      respecting the original frequency distribution.
    """
    synthetic = pd.DataFrame(index=range(num_rows))

    for col in numeric_cols:
        values = df[col].dropna()
        sampled = values.sample(n=num_rows, replace=True).reset_index(drop=True)

        std = values.std()
        noise_scale = std * 0.05 if std and not np.isnan(std) else 0.0
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=num_rows)
        new_values = sampled.values + noise

        new_values = pd.Series(new_values)
        if pd.api.types.is_integer_dtype(df[col]):
            new_values = new_values.round().astype(df[col].dtype, errors="ignore")

        synthetic[col] = new_values.values

    for col in categorical_cols:
        values = df[col].dropna()
        synthetic[col] = values.sample(
            n=num_rows, replace=True
        ).reset_index(drop=True)

    synthetic = synthetic[df.columns.tolist()]
    return synthetic


def generate_synthetic_data(df, num_rows):
    """Main entry point for generating synthetic data.

    Tries the Gaussian Copula approach first (if the library is available).
    Falls back to random sampling with noise if the copula approach is
    unavailable or fails for any reason.
    Returns (synthetic_df, method_used).
    """
    numeric_cols, categorical_cols = split_columns(df)
    method_used = None
    synthetic = None

    if COPULAS_AVAILABLE:
        try:
            synthetic = generate_with_gaussian_copula(
                df, num_rows, numeric_cols, categorical_cols
            )
            method_used = "Gaussian Copula"
        except Exception:
            synthetic = None  # Fall through to the fallback method below.

    if synthetic is None:
        synthetic = generate_with_random_noise(
            df, num_rows, numeric_cols, categorical_cols
        )
        method_used = "Random Sampling with Noise (fallback)"

    # Make sure we never accidentally return exact copies of original rows.
    synthetic = remove_exact_duplicates(synthetic, df)

    # If removing duplicates left us short of the requested number of rows,
    # top up with additional fallback-generated rows until we reach the target.
    attempts = 0
    while len(synthetic) < num_rows and attempts < 5:
        extra_needed = num_rows - len(synthetic)
        extra = generate_with_random_noise(df, extra_needed, numeric_cols, categorical_cols)
        extra = remove_exact_duplicates(extra, df)
        synthetic = pd.concat([synthetic, extra], ignore_index=True)
        attempts += 1

    synthetic = synthetic.head(num_rows).reset_index(drop=True)
    return synthetic, method_used


def convert_df_to_csv_bytes(df):
    """Convert a dataframe to CSV bytes for the Streamlit download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 Synthetic Data Generator")
st.sidebar.write(
    "Upload a CSV file and generate a synthetic version of it that keeps "
    "the same statistical shape, without copying real rows."
)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Generation Settings")
num_synthetic_rows = st.sidebar.number_input(
    "Number of synthetic rows to generate",
    min_value=1,
    max_value=1_000_000,
    value=100,
    step=1,
)

if COPULAS_AVAILABLE:
    st.sidebar.success("Gaussian Copula engine available ✅")
else:
    st.sidebar.warning(
        "`copulas` library not found. Using Random Sampling with noise instead."
    )

generate_clicked = st.sidebar.button("🚀 Generate Synthetic Data", use_container_width=True)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Synthetic Data Generator")
st.write(
    "This tool creates a synthetic dataset that mirrors the statistical "
    "properties of your uploaded CSV file, while avoiding exact duplicate "
    "records."
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
else:
    df, error = load_csv(uploaded_file)

    if error:
        st.error(error)
    else:
        # ---- Dataset preview -------------------------------------------------
        st.subheader("📄 Original Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # ---- Dataset overview ---------------------------------------------
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        col1.metric("Number of Rows", df.shape[0])
        col2.metric("Number of Columns", df.shape[1])

        st.write("**Column Data Types & Missing Values**")
        st.dataframe(get_dataset_summary(df), use_container_width=True)

        # ---- Generation ------------------------------------------------------
        if generate_clicked:
            if df.shape[0] < 2:
                st.error(
                    "The dataset needs at least 2 rows to generate synthetic data."
                )
            else:
                with st.spinner("Generating synthetic data, please wait..."):
                    try:
                        synthetic_df, method_used = generate_synthetic_data(
                            df, int(num_synthetic_rows)
                        )
                        st.session_state["synthetic_df"] = synthetic_df
                        st.session_state["method_used"] = method_used
                        st.session_state["original_shape"] = df.shape
                    except Exception as exc:
                        st.error(f"Something went wrong while generating data: {exc}")

        # ---- Show results (persisted across reruns via session_state) --------
        if "synthetic_df" in st.session_state:
            synthetic_df = st.session_state["synthetic_df"]
            method_used = st.session_state["method_used"]
            original_shape = st.session_state["original_shape"]

            st.success(f"Synthetic data generated successfully using: {method_used}")

            st.subheader("🧬 Synthetic Dataset Preview")
            st.dataframe(synthetic_df.head(20), use_container_width=True)

            st.subheader("🔍 Shape Comparison")
            comp_col1, comp_col2 = st.columns(2)
            comp_col1.metric(
                "Original Dataset Shape",
                f"{original_shape[0]} rows × {original_shape[1]} cols",
            )
            comp_col2.metric(
                "Synthetic Dataset Shape",
                f"{synthetic_df.shape[0]} rows × {synthetic_df.shape[1]} cols",
            )

            st.subheader("⬇️ Download Synthetic Dataset")
            csv_bytes = convert_df_to_csv_bytes(synthetic_df)
            st.download_button(
                label="Download as CSV",
                data=csv_bytes,
                file_name="synthetic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
