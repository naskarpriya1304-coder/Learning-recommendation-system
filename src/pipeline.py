"""Auditable training and course-ranking pipeline for GR-1."""
from __future__ import annotations

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, ndcg_score, precision_score,
                             recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "learning_path_dataset.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "auto_recommender.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
TARGET = "Next_Course_Recommendation"
# Completed_Courses is intentionally audited but excluded from model inputs because
# 21.7% of rows contain the target label in that history and no event timestamps exist.
FEATURES = ["Current_Course", "Learning_Level", "Learning_Preference",
            "Time_Spent_Hours", "Performance_Score"]
CATEGORICAL = ["Current_Course", "Learning_Level", "Learning_Preference"]
NUMERIC = ["Time_Spent_Hours", "Performance_Score"]
RANDOM_STATE = 42


def parse_courses(value):
    """Parse the CSV's comma-separated course history as unique course tokens."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = str(value).split(",")
    return sorted({re.sub(r"\s+", " ", str(part)).strip() for part in raw
                   if str(part).strip()})


def clean_category(value):
    if value is None or pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return np.nan
    if "," in text:
        return ", ".join(parse_courses(text))
    return text


def load_and_validate(path=DATA_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in [*FEATURES, "Completed_Courses", TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.copy()
    for column in [*CATEGORICAL, "Completed_Courses", TARGET]:
        df[column] = df[column].map(clean_category)
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Preserve unusual observations; range checks are reported, not silently clipped.
    if df[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET} contains missing or blank values.")
    return df


def audit_dataset(path=DATA_PATH):
    raw = pd.read_csv(path)
    cleaned = load_and_validate(path)
    attrs = [{"name": str(c), "dtype": str(raw[c].dtype),
              "missing": int(raw[c].isna().sum()),
              "unique_values": int(raw[c].nunique(dropna=True)),
              "sample_values": [str(x) for x in raw[c].dropna().unique()[:5]]}
             for c in raw.columns]
    out_of_range = {
        "Time_Spent_Hours": int((~cleaned.Time_Spent_Hours.between(0, 1000)).fillna(False).sum()),
        "Performance_Score": int((~cleaned.Performance_Score.between(0, 100)).fillna(False).sum()),
    }
    target_in_history = int(sum(target in set(parse_courses(history))
                                for target, history in zip(cleaned[TARGET], cleaned["Completed_Courses"])))
    target_matches_current = int(cleaned[TARGET].eq(cleaned["Current_Course"]).sum())
    return cleaned, {"file_name": Path(path).name, "file_path": str(Path(path).resolve()),
                     "rows": int(raw.shape[0]), "columns_count": int(raw.shape[1]),
                     "duplicate_rows": int(raw.duplicated().sum()),
                     "target_exists": TARGET in raw.columns,
                     "leakage_checks": {"target_in_completed_course_history_rows": target_in_history,
                                        "target_matches_current_course_rows": target_matches_current},
                     "target_distribution": {str(k): int(v) for k, v in raw[TARGET].value_counts(dropna=False).items()} if TARGET in raw else {},
                     "out_of_range_numeric": out_of_range, "attributes": attrs}


def split_data(df, save=True):
    """Fixed-seed stratified row split; Learner_ID is unique in the audited data."""
    y = df[TARGET].astype(str)
    if int(y.value_counts().min()) < 2:
        raise ValueError("Cannot stratify: at least one target class has fewer than two rows.")
    train, test = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE,
                                   stratify=y)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    if save:
        TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(TRAIN_PATH, index=False)
        test.to_csv(TEST_PATH, index=False)
    return train, test


def build_pipeline(classifier):
    prep = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), NUMERIC),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def candidate_models():
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_and_evaluate():
    df, audit = audit_dataset()
    train, test = split_data(df)
    X_train, y_train = train[FEATURES], train[TARGET].astype(str)
    X_test, y_test = test[FEATURES], test[TARGET].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparisons = {}
    fitted_candidates = {}
    for name, classifier in candidate_models().items():
        pipeline = build_pipeline(classifier)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
                                n_jobs=1, return_train_score=False)
        comparisons[name] = {"cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                             "cv_accuracy_std": float(scores["test_accuracy"].std()),
                             "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
                             "cv_macro_f1_std": float(scores["test_macro_f1"].std())}
        fitted_candidates[name] = pipeline
    # Model selection uses training-only CV macro F1; the test set is evaluated once below.
    selected_name = max(comparisons, key=lambda name: comparisons[name]["cv_macro_f1_mean"])
    model = fitted_candidates[selected_name].fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["classifier"].classes_
    predictions = model.predict(X_test)
    labels = [str(label) for label in classes]
    relevance = np.zeros((len(y_test), len(classes)))
    class_index = {str(label): i for i, label in enumerate(classes)}
    for row, label in enumerate(y_test):
        if label in class_index:
            relevance[row, class_index[label]] = 1
    hit_rate = {}
    for k in (1, 3, 5):
        top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
        hit_rate[str(k)] = float(np.mean([y_test.iloc[row] in {str(classes[i]) for i in ids}
                                         for row, ids in enumerate(top_indices)]))
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "target": TARGET, "feature_columns": FEATURES, "rows": len(df),
        "train_rows": len(train), "test_rows": len(test), "random_state": RANDOM_STATE,
        "split": "stratified random 80/20 row split; Learner_ID is unique in the source dataset",
        "selected_model": selected_name, "model_comparison_cv": comparisons,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "hit_rate_at_k": hit_rate,
        "ndcg_at_5": float(ndcg_score(relevance, probabilities, k=5)),
        "classification_report": report, "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "audit": audit,
        "note": "Models were compared by 5-fold stratified CV on training rows. The selected pipeline was refit on training rows and evaluated once on the untouched test split. Results describe this random split only; no future or unseen-learner guarantee is possible from these fields."
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target": TARGET, "features": FEATURES,
                 "course_options": sorted(set(df.Current_Course.dropna().astype(str)) |
                                           set(df[TARGET].dropna().astype(str))),
                 "selected_model": selected_name, "random_state": RANDOM_STATE}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    write_audit_report(df, audit)
    create_eda(df, comparisons)
    return metrics


def profile_to_frame(profile, features=FEATURES):
    row = {column: profile.get(column, np.nan) for column in features}
    for column in CATEGORICAL:
        row[column] = clean_category(row[column])
    return pd.DataFrame([row], columns=features)


def recommend(profile, k=5, bundle=None):
    if not 1 <= int(k) <= 5:
        raise ValueError("k must be between 1 and 5")
    if bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact not found. Run python main.py --step train first.")
        bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    probabilities = model.predict_proba(profile_to_frame(profile, bundle["features"]))[0]
    classes = model.named_steps["classifier"].classes_
    completed = set(parse_courses(profile.get("Completed_Courses", [])))
    current = clean_category(profile.get("Current_Course"))
    eligible = [(i, str(label)) for i, label in enumerate(classes)
                if str(label) not in completed and str(label) != str(current)]
    eligible.sort(key=lambda pair: (-float(probabilities[pair[0]]), pair[1]))
    rows = []
    for rank, (index, label) in enumerate(eligible[:int(k)], 1):
        score = float(probabilities[index])
        rows.append({"Rank": rank, "Recommended course": label, "Ranking score": score,
                     "Why recommended": f"Ranked {rank} from the model's historical next-course predictions for the supplied profile. The score compares candidate labels; it is not a probability of passing or a guarantee of benefit."})
    return pd.DataFrame(rows, columns=["Rank", "Recommended course", "Ranking score", "Why recommended"])


def similar_group_summary(profile, df=None):
    if df is None:
        df, _ = audit_dataset()
    group = df[df.Current_Course.eq(profile.get("Current_Course"))]
    level = profile.get("Learning_Level")
    if level:
        narrowed = group[group.Learning_Level.eq(level)]
        if len(narrowed):
            group = narrowed
    if group.empty:
        return {"group_size": 0, "score_median": None, "hours_median": None,
                "method": "No historical rows matched the selected course and learner level."}
    return {"group_size": int(len(group)),
            "score_median": float(group.Performance_Score.median()),
            "hours_median": float(group.Time_Spent_Hours.median()),
            "method": "Observed median for rows with the same current course" +
                      (f" and learner level {level}." if level else ".")}


def create_eda(df, comparisons=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plots = [
        (TARGET, "Target class distribution", "next_course_counts.png"),
        ("Current_Course", "Current course distribution", "current_course_counts.png"),
        ("Learning_Level", "Learning level distribution", "learning_level_counts.png"),
        ("Learning_Preference", "Learning preference distribution", "learning_preference_counts.png"),
    ]
    for column, title, filename in plots:
        fig, ax = plt.subplots(figsize=(9, 5))
        df[column].value_counts().sort_values().plot.barh(ax=ax, color="#3976c5")
        ax.set_title(title); ax.set_xlabel("Rows"); fig.tight_layout()
        fig.savefig(out / filename, dpi=140); plt.close(fig)
    for column, title, filename in [("Time_Spent_Hours", "Study time distribution", "time_spent_distribution.png"),
                                    ("Performance_Score", "Performance score distribution", "performance_score_distribution.png")]:
        fig, ax = plt.subplots(figsize=(8, 4)); df[column].dropna().plot.hist(ax=ax, bins=20, color="#3976c5")
        ax.set_title(title); ax.set_xlabel(column); fig.tight_layout(); fig.savefig(out / filename, dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); df[NUMERIC].corr().pipe(lambda c: ax.imshow(c, cmap="Blues", vmin=-1, vmax=1))
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=30, ha="right"); ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    ax.set_title("Numerical feature correlation"); fig.colorbar(ax.images[0], ax=ax); fig.tight_layout()
    fig.savefig(out / "numeric_correlation.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(df["Learning_Level"], df[TARGET], normalize="index").plot.bar(stacked=True, ax=ax, legend=False)
    ax.set_title("Target proportions by learning level"); ax.set_ylabel("Within-level proportion"); fig.tight_layout()
    fig.savefig(out / "target_by_learning_level.png", dpi=140); plt.close(fig)
    for column, filename, title in [("Current_Course", "target_by_current_course.png", "Target by current course"),
                                    ("Learning_Preference", "target_by_preference.png", "Target by learning preference")]:
        table = pd.crosstab(df[column], df[TARGET], normalize="index")
        fig, ax = plt.subplots(figsize=(10, 5)); table.plot.bar(stacked=True, ax=ax, legend=False)
        ax.set_title(title); ax.set_ylabel("Within-category proportion"); ax.set_xlabel(column)
        fig.tight_layout(); fig.savefig(out / filename, dpi=140); plt.close(fig)
    quality = pd.Series({"Missing cells": int(df.isna().sum().sum()),
                         "Exact duplicate rows": int(df.duplicated().sum())})
    fig, ax = plt.subplots(figsize=(6, 4)); quality.plot.bar(ax=ax, color="#3976c5")
    ax.set_title("Missing cells and exact duplicate rows"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(out / "data_quality.png", dpi=140); plt.close(fig)
    if comparisons:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(comparisons)
        vals = [comparisons[n]["cv_macro_f1_mean"] for n in names]
        ax.bar(names, vals, color="#3976c5"); ax.set_ylim(0, 1); ax.set_ylabel("Training CV macro F1")
        ax.set_title("Model comparison (training CV only)"); ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=140); plt.close(fig)


def write_audit_report(df, audit):
    columns = "\n".join(f"| {a['name']} | {a['dtype']} | {a['missing']} | {a['unique_values']} |"
                         for a in audit["attributes"])
    targets = "\n".join(f"| {label} | {count} |" for label, count in audit["target_distribution"].items())
    path = ROOT / "reports" / "data_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# GR-1 data audit

## Source and schema

Read-only input: `{audit['file_path']}`. Pandas loaded {audit['rows']} rows and {audit['columns_count']} columns. The exact target column `{TARGET}` is present. There are {audit['duplicate_rows']} exact duplicate rows. Missing values and distinct counts below are from the original CSV before normalization.

| Column | pandas dtype | Missing | Distinct non-null |
|---|---|---:|---:|
{columns}

## Target class distribution

| Target label | Rows |
|---|---:|
{targets}

## Validation and use

The model uses these observed profile columns: `{', '.join(FEATURES)}`. `Learner_ID`, `Completed_Courses`, and the target are excluded from predictors. The dataset has no timestamps, and the target appears in `Completed_Courses` for {audit['leakage_checks']['target_in_completed_course_history_rows']} rows; history is therefore not used as a feature. The target equals `Current_Course` for {audit['leakage_checks']['target_matches_current_course_rows']} rows. Current course is available at prediction time, but these labels cannot be returned by the app because it excludes the current course. These cases remain in the evaluation and are a task-data limitation. Completed-course values entered in the app are only used to filter those labels from recommendations. Categories are imputed and one-hot encoded, and numeric values are imputed using training medians and standardized, all inside the model pipeline. Numeric observations outside 0–1000 hours or 0–100 score are flagged in `reports/metrics.json`; they are retained and are not silently clipped.

There are no resource IDs/types, timestamps, assessment records, reliable completion events, or independent learner interactions. The target supports prediction of a next-course label only. It cannot establish learning-resource effectiveness or predict academic success. The random row split is appropriate for reproducing this dataset's row-level task; the source does not support temporal evaluation. Check whether `Learner_ID` represents repeated people before interpreting results. In this file it has {df['Learner_ID'].nunique() if 'Learner_ID' in df else 'no'} distinct values across {len(df)} rows.

## Generated outputs

`data/processed/train.csv` and `data/processed/test.csv` are an 80/20 stratified split derived from this input, seed 42. Figures are saved under `reports/figures/`. See `reports/metrics.json` for training-only cross-validation model comparison and the single held-out test evaluation.
""", encoding="utf-8")
