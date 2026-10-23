from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "learning_path_dataset.csv"
MODEL_PATH = ROOT / "models" / "auto_recommender.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

def audit_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    audit = {
        "file_name": Path(path).name,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "attributes": []
    }
    for col in df.columns:
        series = df[col]
        audit["attributes"].append({
            "name": str(col),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "sample_values": [str(v) for v in series.dropna().unique()[:5]]
        })
    return df, audit

def infer_target(df):
    # Prefer explicit target-like column names, otherwise require a user choice in the UI.
    candidates = [c for c in df.columns if any(word in c.lower() for word in
        ["target", "label", "recommendation", "next_course", "outcome", "result"])]
    if len(candidates) == 1:
        return candidates[0]
    return None

def infer_identifier_columns(df, target):
    ids = []
    for c in df.columns:
        if c == target:
            continue
        name = c.lower()
        unique_ratio = df[c].nunique(dropna=True) / max(1, len(df))
        if name in {"id", "student_id", "learner_id", "user_id", "customer_id"} or (
            name.endswith("_id") and unique_ratio > 0.5
        ):
            ids.append(c)
    return ids

def build_pipeline(df, target, id_cols):
    features = [c for c in df.columns if c != target and c not in id_cols]
    if not features:
        raise ValueError("No usable feature columns remain after excluding the target and identifiers.")
    X = df[features].copy()
    y = df[target].astype(str)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers = []
    if cat_cols:
        transformers.append(("categorical", categorical, cat_cols))
    if num_cols:
        transformers.append(("numeric", numeric, num_cols))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    return model, features, cat_cols, num_cols

def train_and_evaluate(target=None):
    df, audit = audit_dataset()
    if target is None:
        target = infer_target(df)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is not in dataset.")
    id_cols = infer_identifier_columns(df, target)
    features = [c for c in df.columns if c != target and c not in id_cols]
    X = df[features].copy()
    y = df[target].astype(str)
    # Keep a truly separate test set; stratify only when each class has enough samples.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )
    model, features, cat_cols, num_cols = build_pipeline(df, target, id_cols)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(y.unique().tolist())
    metrics = {
        "target": target,
        "identifier_columns_excluded": id_cols,
        "feature_columns": features,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "audit": audit,
        "note": "Preprocessing is fitted inside the pipeline on training data only. Test set is held out for final evaluation."
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target": target, "features": features, "id_cols": id_cols,
                 "cat_cols": cat_cols, "num_cols": num_cols, "df": df}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics

def recommend(profile, k=5):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    row = {col: profile.get(col, np.nan) for col in features}
    X = pd.DataFrame([row], columns=features)
    probs = model.predict_proba(X)[0]
    classes = model.named_steps["classifier"].classes_
    order = np.argsort(probs)[::-1][:int(k)]
    rows = []
    for rank, idx in enumerate(order, 1):
        rows.append({
            "Rank": rank,
            "Recommended next course": str(classes[idx]),
            "Match score": float(probs[idx]),
            "Why recommended": "Ranked by the model from historical examples with similar supplied attributes."
        })
    return pd.DataFrame(rows)
