"""Streamlit interface for the saved GR-1 course recommender."""
import json

import joblib
import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import (DATA_PATH, METRICS_PATH, MODEL_PATH, audit_dataset,
                          recommend, similar_group_summary)

st.set_page_config(page_title="GR-1: Personalized Learning Recommendation System",
                   page_icon="📚", layout="wide")
st.title("GR-1: Personalized Learning Recommendation System")
st.caption("Ranks course labels observed in the supplied dataset. These are alternative predictions, not guaranteed outcomes.")


@st.cache_resource
def load_bundle(path, modified):
    del modified  # use file modification time as the cache key
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("Trained model is missing. From the project folder, run `python main.py --step all`.")
    st.stop()
if not DATA_PATH.exists():
    st.error(f"Dataset is missing: {DATA_PATH}")
    st.stop()

bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
df, audit = audit_dataset()
st.caption(f"Dataset: {audit['rows']:,} rows, {audit['columns_count']} columns. Target: `{bundle['target']}`.")

with st.form("profile"):
    st.subheader("Student input")
    col1, col2 = st.columns(2)
    with col1:
        current = st.selectbox("Current course", ["", *sorted(df.Current_Course.dropna().unique())])
        level = st.selectbox("Learning level", ["", *sorted(df.Learning_Level.dropna().unique())])
        preference = st.selectbox("Learning preference", ["", *sorted(df.Learning_Preference.dropna().unique())])
    with col2:
        completed = st.multiselect("Courses already completed", bundle["course_options"])
        hours = st.number_input("Time spent so far (hours)", min_value=0.0, max_value=1000.0,
                                value=None, step=1.0, placeholder="Enter hours")
        score = st.number_input("Performance score (0–100)", min_value=0.0, max_value=100.0,
                                value=None, step=1.0, placeholder="Enter score")
    submitted = st.form_submit_button("Get top 5 course predictions", type="primary")

if submitted:
    missing = [name for name, value in [("current course", current), ("learning level", level),
                ("learning preference", preference), ("time spent", hours), ("performance score", score)]
               if value is None or value == ""]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    else:
        profile = {"Current_Course": current, "Learning_Level": level,
                   "Learning_Preference": preference, "Completed_Courses": completed,
                   "Time_Spent_Hours": float(hours), "Performance_Score": float(score)}
        try:
            results = recommend(profile, k=5, bundle=bundle)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not generate recommendations: {exc}")
        else:
            st.subheader("Your profile")
            st.write(f"**Current course:** {current} · **Level:** {level} · **Preference:** {preference}")
            st.write(f"**Time:** {hours:g} hours · **Performance score:** {score:g} · **Completed courses:** {', '.join(completed) if completed else 'None entered'}")
            if results.empty:
                st.info("No eligible course labels remain after excluding the current and completed courses.")
            else:
                st.subheader("Ranked alternative course predictions")
                st.dataframe(results, hide_index=True, column_config={
                    "Ranking score": st.column_config.ProgressColumn("Ranking score", min_value=0.0, max_value=1.0, format="%.3f")})
                fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(results))))
                plot = results.sort_values("Ranking score")
                ax.barh(plot["Recommended course"], plot["Ranking score"], color="#3976c5")
                ax.set_xlim(0, 1); ax.set_xlabel("Model score used to order labels"); fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                st.caption("Scores rank the candidate course labels for this model. They are not probabilities of passing or guarantees of benefit.")
            comparison = similar_group_summary(profile, df)
            st.subheader("Observed profile comparison")
            if comparison["group_size"]:
                st.write(f"{comparison['group_size']} historical rows matched by current course and, when available, learning level.")
                a, b = st.columns(2)
                a.metric("Your score", f"{score:.1f}"); a.caption(f"Observed group median: {comparison['score_median']:.1f}")
                b.metric("Your study time", f"{hours:.1f} hours"); b.caption(f"Observed group median: {comparison['hours_median']:.1f} hours")
            else:
                st.info(comparison["method"])

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    with st.expander("Model evaluation on held-out data"):
        st.write(f"Selected using training-only cross-validation: **{metrics['selected_model']}**.")
        st.write(f"Stratified split: {metrics['train_rows']} training rows, {metrics['test_rows']} test rows.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Macro F1", f"{metrics['f1_macro']:.3f}")
        c3.metric("NDCG@5", f"{metrics['ndcg_at_5']:.3f}")
        st.caption(metrics["note"])
