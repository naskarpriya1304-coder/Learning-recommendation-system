import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.pipeline import DATA_PATH, MODEL_PATH, METRICS_PATH, audit_dataset, infer_target, infer_identifier_columns, train_and_evaluate, recommend

st.set_page_config(page_title="Automatic Attribute Learning Recommender", page_icon="📚", layout="wide")
st.title("📚 Personalized Learning Recommendation System")
st.caption("Dataset attributes are discovered automatically. All learner inputs are entered manually.")

df, audit = audit_dataset()
auto_target = infer_target(df)

with st.sidebar:
    st.header("1. Dataset audit")
    st.metric("Rows", audit["rows"])
    st.metric("Attributes", audit["columns_count"])
    st.metric("Duplicate rows", audit["duplicate_rows"])
    st.write("Detected columns, data types, missing and unique values")
    st.dataframe(pd.DataFrame(audit["attributes"]), use_container_width=True, hide_index=True)
    st.divider()
    st.header("2. Choose target")
    target_options = df.columns.tolist()
    default_index = target_options.index(auto_target) if auto_target in target_options else len(target_options)-1
    target = st.selectbox("What should the model predict?", target_options, index=default_index)
    if st.button("Train and evaluate model", type="primary", use_container_width=True):
        with st.spinner("Auditing, splitting, preprocessing, training and evaluating..."):
            metrics = train_and_evaluate(target)
            st.session_state["metrics"] = metrics
        st.success("Training and held-out testing completed.")

if not MODEL_PATH.exists():
    st.info("Check the target column in the sidebar and click **Train and evaluate model**. No learner values are prefilled.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
# If user changed the target after model training, ask them to retrain instead of silently using a stale model.
if bundle["target"] != target:
    st.warning("Selected target differs from the trained model. Click **Train and evaluate model** to retrain.")
    st.stop()

features = bundle["features"]
raw_df = bundle["df"]
st.subheader("3. Enter your own input")
st.write("The form below is generated from detected feature attributes. No existing Learner ID or student row is selected.")

profile = {}
with st.form("manual_profile"):
    cols = st.columns(2)
    for i, col in enumerate(features):
        series = raw_df[col]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce").dropna()
                median = float(clean.median()) if len(clean) else 0.0
                minval = float(clean.min()) if len(clean) else 0.0
                maxval = float(clean.max()) if len(clean) else max(100.0, median)
                if minval == maxval:
                    maxval = minval + 1.0
                # Use a neutral starting value, not an existing learner's row.
                default = 0.0 if minval <= 0 <= maxval else minval
                profile[col] = st.number_input(
                    col, min_value=minval, max_value=maxval,
                    value=float(default), key=f"field_{col}"
                )
            else:
                vals = sorted(series.dropna().astype(str).unique().tolist())
                # Blank initial selection: the user must pick a category.
                profile[col] = st.selectbox(
                    col, ["-- Select --"] + vals, index=0, key=f"field_{col}"
                )
    k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    submitted = st.form_submit_button("Get recommendations", type="primary", use_container_width=True)

if submitted:
    blanks = [c for c in features if isinstance(profile[c], str) and profile[c] == "-- Select --"]
    if blanks:
        st.error("Please fill/select these fields: " + ", ".join(blanks))
        st.stop()
    results = recommend(profile, k=k)
    st.subheader("4. Ranked recommendations")
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(results, use_container_width=True, hide_index=True, column_config={
            "Match score": st.column_config.ProgressColumn("Match score", min_value=0.0, max_value=1.0, format="%.3f")
        })
    with right:
        fig, ax = plt.subplots()
        plot = results.sort_values("Match score", ascending=True)
        ax.barh(plot["Recommended next course"], plot["Match score"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted model probability")
        ax.set_title("Recommendation scores")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Scores are model probabilities, not a guarantee of success. The model predicts the selected target column.")

st.subheader("5. Held-out test performance")
metrics = st.session_state.get("metrics")
if metrics is None and METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
if metrics and metrics.get("target") == bundle["target"]:
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
    c.metric("Test records", str(metrics["test_rows"]))
    st.caption("The 80/20 split is performed before fitting imputers, encoders, scalers and the model. A 70–80% accuracy target cannot be guaranteed; report the measured test score.")
    with st.expander("Full evaluation report"):
        st.json(metrics)
