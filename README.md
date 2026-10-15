# Synthetic Data Generator

A simple Streamlit application that lets you upload a CSV file and generate
a synthetic version of it. The synthetic dataset preserves the overall
statistical properties of the original data (distributions, correlations,
value ranges) without containing exact copies of the original records.

## Features

- Upload a CSV file
- Preview the uploaded dataset
- View dataset overview: number of rows, number of columns, data types,
  and missing values
- Choose how many synthetic rows to generate
- Generate synthetic data using:
  - **Gaussian Copula** (preferred method, via the `copulas` library)
  - **Random Sampling with Noise** (automatic fallback if `copulas` is not
    available or fails to fit the data)
- Preview the generated synthetic dataset
- Compare the shape (rows × columns) of the original vs. synthetic dataset
- Download the synthetic dataset as a CSV file

## Project Structure

```text
Synthetic_Data_Generator/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Upload**: You upload a CSV file through the sidebar file uploader.
2. **Inspect**: The app shows a preview of your data along with a summary
   (row/column counts, data types, missing values per column).
3. **Configure**: You choose how many synthetic rows you'd like to generate.
4. **Generate**: When you click "Generate Synthetic Data":
   - The app first tries to fit a **Gaussian Copula** model on your data.
     Categorical columns are internally encoded as numbers so the copula
     can model them, then decoded back to their original categories after
     sampling.
   - If the `copulas` library isn't installed, or the model fails to fit
     (for example, if the dataset is too small), the app automatically
     falls back to **random sampling with noise**: numeric columns are
     resampled from existing values with a small amount of Gaussian noise
     added, and categorical columns are resampled according to their
     original frequency distribution.
   - After generation, any synthetic row that happens to exactly match an
     original row is removed and replaced, so the output never leaks real
     records.
5. **Review & Download**: You can preview the synthetic dataset, compare
   its shape to the original, and download it as a CSV file.

## Installation & Usage

1. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL shown in your terminal (usually `http://localhost:8501`)
   in your browser.

## Notes

- Only CSV files are supported.
- The app handles empty or invalid CSV files gracefully and shows an
  error message instead of crashing.
- No external APIs or databases are used — everything runs locally.
- For very small datasets (fewer than 2 rows), synthetic generation is
  disabled since there isn't enough data to model statistical properties.
