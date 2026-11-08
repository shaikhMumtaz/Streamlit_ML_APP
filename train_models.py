"""
Train and save all five required classifiers for the Dry Bean project.

Dataset options:
1. Put Dry_Bean_Dataset.xlsx in this folder, OR
2. Put Dry_Bean_Dataset.csv in this folder, OR
3. If neither file exists, the script downloads the UCI dataset using ucimlrepo.

Run:
    python -m pip install -r requirements.txt
    python train_models.py

Generated files:
    test_data.csv
    dataset_summary.json
    model/logistic_regression.joblib
    model/decision_tree.joblib
    model/knn.joblib
    model/gaussian_naive_bayes.joblib
    model/support_vector_machine.joblib
    model/metrics.csv
"""
from pathlib import Path
import json
import sys
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset():
    """Load local CSV/Excel first; otherwise fetch dataset 602 from UCI."""
    excel_path = ROOT / "Dry_Bean_Dataset.xlsx"
    csv_path = ROOT / "Dry_Bean_Dataset.csv"

    if excel_path.exists():
        print(f"Loading local dataset: {excel_path.name}")
        df = pd.read_excel(excel_path)
    elif csv_path.exists():
        print(f"Loading local dataset: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        print("No local dataset found. Fetching UCI Dry Bean dataset (ID 602)...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=602)
            features = dataset.data.features.copy()
            targets = dataset.data.targets.copy()

            if isinstance(targets, pd.Series):
                targets = targets.to_frame(name=TARGET)
            elif TARGET not in targets.columns:
                targets.columns = [TARGET]

            df = pd.concat(
                [features.reset_index(drop=True), targets.reset_index(drop=True)],
                axis=1,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download the dataset from UCI. Either connect to the "
                "internet and install ucimlrepo, or download Dry_Bean_Dataset.xlsx "
                "from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset "
                "and place it beside train_models.py. Original error: "
                f"{exc}"
            ) from exc

    df.columns = [str(col).strip() for col in df.columns]

    # Some downloaded formats may use a different capitalization for target.
    target_match = next(
        (col for col in df.columns if col.lower() == TARGET.lower()), None
    )
    if target_match is None:
        raise ValueError(
            f"Target column '{TARGET}' was not found. Dataset columns: {list(df.columns)}"
        )
    if target_match != TARGET:
        df = df.rename(columns={target_match: TARGET})

    df[TARGET] = df[TARGET].astype(str).str.strip()
    feature_cols = [col for col in df.columns if col != TARGET]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)

    if len(feature_cols) < 12 or len(df) < 500:
        raise ValueError(
            f"Dataset does not meet assignment requirements: "
            f"{len(feature_cols)} features and {len(df)} rows."
        )

    return df, feature_cols


def build_models():
    """Return all five required models as complete preprocessing pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=2.0,
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def safe_multiclass_auc(y_true, probabilities, classes):
    """Calculate macro one-vs-rest AUC; return NaN if test data lacks a class."""
    try:
        return roc_auc_score(
            y_true,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
    except ValueError as exc:
        print(f"Warning: AUC could not be calculated: {exc}")
        return float("nan")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    df, feature_cols = load_dataset()
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Keep actual target labels in test_data.csv so the Streamlit app can
    # calculate evaluation metrics, not just predictions.
    test_data = X_test.copy()
    test_data[TARGET] = y_test.values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    results = []
    model_name_to_file = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
    }

    for model_name, pipeline in build_models().items():
        print(f"\nTraining {model_name} ...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        classifier = pipeline.named_steps["classifier"]
        classes = classifier.classes_

        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": safe_multiclass_auc(y_test, y_prob, classes),
            "Precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        results.append(row)

        artifact = {
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "target_column": TARGET,
            "classes": [str(value) for value in classes],
        }
        artifact_path = MODEL_DIR / model_name_to_file[model_name]
        joblib.dump(artifact, artifact_path)
        print(f"Saved model: {artifact_path}")

    metrics = pd.DataFrame(results)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    summary = {
        "dataset": "UCI Dry Bean Dataset",
        "rows_after_cleaning": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "precision_recall_f1_average": "macro",
        "auc": "one-vs-rest macro",
    }
    (ROOT / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n========== MODEL COMPARISON ==========")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTraining completed successfully.")
    print("Generated test_data.csv, dataset_summary.json, and five model files.")
    print("IMPORTANT: Commit the generated model/*.joblib files to GitHub for deployment.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nTRAINING FAILED: {error}", file=sys.stderr)
        raise
