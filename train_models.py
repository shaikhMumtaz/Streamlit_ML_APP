"""
Train five classifiers on the UCI Dry Bean dataset.

Usage:
  1. Put Dry_Bean_Dataset.xlsx (or Dry_Bean_Dataset.csv) in this folder.
  2. Run: python train_models.py

The script creates:
  model/*.joblib, model/metrics.csv, test_data.csv, dataset_summary.json
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Class"


def load_dataset():
    xlsx = ROOT / "Dry_Bean_Dataset.xlsx"
    csv = ROOT / "Dry_Bean_Dataset.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            "Dataset not found. Download the UCI Dry Bean dataset and place "
            "'Dry_Bean_Dataset.xlsx' or 'Dry_Bean_Dataset.csv' beside train_models.py."
        )
    # Clean common accidental whitespace in column names and target labels.
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(df.columns)}")
    df = df.drop_duplicates().copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    for col in df.columns:
        if col != TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df


def build_models():
    # Scaling is included for all models for a consistent, reusable pipeline.
    # Decision trees and GaussianNB do not require scaling, but it does not
    # prevent them from working.
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                max_depth=None, random_state=RANDOM_STATE
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=2.0, probability=True,
                random_state=RANDOM_STATE
            ))
        ])
    }


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_dataset()
    feature_cols = [c for c in df.columns if c != TARGET]
    if len(feature_cols) < 12 or len(df) < 500:
        raise ValueError(
            f"Dataset does not meet assignment minimum: {len(feature_cols)} features, {len(df)} rows."
        )

    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # The CSV is intentionally labelled: metrics can only be calculated when
    # actual target values are available.
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    metrics_rows = []
    for display_name, pipeline in build_models().items():
        print(f"Training {display_name} ...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        classes = pipeline.named_steps["classifier"].classes_

        row = {
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(
                y_test, probabilities, labels=classes,
                multi_class="ovr", average="macro"
            ),
            "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }
        metrics_rows.append(row)
        filename = display_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump({
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "target_column": TARGET,
            "classes": list(classes)
        }, MODEL_DIR / filename)
        print(f"  Accuracy={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}, "
              f"F1={row['F1']:.4f}, MCC={row['MCC']:.4f}")

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    summary = {
        "dataset": "UCI Dry Bean Dataset",
        "rows_after_duplicate_removal": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "averaging": "macro",
        "auc_method": "one-vs-rest macro"
    }
    (ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCreated test_data.csv and model artifacts in the model/ folder.")


if __name__ == "__main__":
    main()
