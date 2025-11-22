import glob
import os
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOG_DIR = "prediction_logs"
ANALYSIS_DIR = "analysis_results"

REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "prediction",
    "probability",
    "timestamp",
]


def load_all_logs() -> Optional[pd.DataFrame]:
    """
    Load and concatenate all CSV log files from LOG_DIR.

    Returns
    -------
    DataFrame with all logs, or None if nothing is available.
    """
    pattern = os.path.join(LOG_DIR, "predictions_*.csv")
    files: List[str] = glob.glob(pattern)

    if not files:
        print("No prediction logs found in directory:", LOG_DIR)
        return None

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if not frames:
        print("No valid log files could be read.")
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Ensure timestamp is datetime
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

    return combined


def ensure_analysis_dir() -> None:
    os.makedirs(ANALYSIS_DIR, exist_ok=True)


def basic_statistics(df: pd.DataFrame) -> None:
    """
    Print high-level stats about predictions.
    """
    total = len(df)
    if total == 0:
        print("No rows in log data.")
        return

    if "prediction" not in df.columns or "probability" not in df.columns:
        print("Required columns 'prediction' or 'probability' missing, cannot compute basic stats.")
        return

    positive = df["prediction"].sum()
    positive_rate = (df["prediction"].mean() * 100.0) if "prediction" in df.columns else float("nan")
    avg_prob = df["probability"].mean()

    print(f"Total predictions: {total}")
    print(f"Positive predictions: {positive} ({positive_rate:.1f}%)")
    print(f"Average probability for class 1: {avg_prob:.3f}")


def plot_daily_counts(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        print("No valid timestamps available, skipping daily count plot.")
        return

    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size()

    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind="bar")
    plt.title("Daily Prediction Counts")
    plt.ylabel("Number of Predictions")
    plt.xlabel("Date")
    plt.tight_layout()

    out_path = os.path.join(ANALYSIS_DIR, "daily_prediction_counts.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved daily prediction counts plot to: {out_path}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = ["Glucose", "BMI", "Age", "BloodPressure"]
    available = [f for f in features if f in df.columns]

    if not available:
        print("None of the expected features are present, skipping distributions.")
        return

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(available, start=1):
        plt.subplot(2, 2, i)
        sns.histplot(df[feat].dropna(), kde=True)
        plt.title(f"Distribution of {feat}")

    plt.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, "feature_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature distribution plot to: {out_path}")


def correlation_with_prediction(df: pd.DataFrame) -> None:
    """
    Compute correlation of numeric columns with 'prediction' and save as CSV.
    """
    if "prediction" not in df.columns:
        print("Column 'prediction' not found, skipping correlation analysis.")
        return

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr_series = numeric_df.corr()["prediction"].sort_values(ascending=False)

    print("\nCorrelation with prediction (numeric features):")
    print(corr_series)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ANALYSIS_DIR, f"correlation_analysis_{ts}.csv")
    corr_series.to_csv(out_path)
    print(f"Saved correlation analysis to: {out_path}")


def save_combined_logs(df: pd.DataFrame) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ANALYSIS_DIR, f"all_predictions_{ts}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved combined prediction logs to: {out_path}")


def main() -> None:
    ensure_analysis_dir()

    df = load_all_logs()
    if df is None:
        return

    basic_statistics(df)
    plot_daily_counts(df)
    plot_feature_distributions(df)
    correlation_with_prediction(df)
    save_combined_logs(df)

    print("\nAnalysis completed. Check the 'analysis_results' directory for outputs.")


if __name__ == "__main__":
    main()
