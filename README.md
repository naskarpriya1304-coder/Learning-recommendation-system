# GR-1: Personalized Learning Recommendation System

A beginner-friendly BCA final-year project that predicts and ranks next-course labels from the supplied learner dataset. It uses the real CSV at `data/raw/learning_path_dataset.csv` as read-only input. The target is `Next_Course_Recommendation`; the system predicts course labels, not learning resources, course quality, or academic success.

## Dataset audit

The audited CSV has 1,000 rows and 8 columns, no missing values, no exact duplicate rows, and 12 target labels. Its columns are `Learner_ID`, `Current_Course`, `Completed_Courses`, `Learning_Level`, `Learning_Preference`, `Time_Spent_Hours`, `Performance_Score`, and `Next_Course_Recommendation`. See [reports/data_audit.md](reports/data_audit.md) for per-column counts, target distribution, and leakage checks. `Learner_ID` is excluded from prediction. `Completed_Courses` is excluded from model features because the target label appears in that field in 217 rows and event timing is unavailable; the app uses the value the learner enters only to filter completed labels from recommendations.

The data does not contain resource IDs or types, timestamps, assessment events, or verified course-completion events. The available fields cannot establish whether a recommendation improves learning. The 70–80% desired accuracy is a goal only; the project reports measured performance without forcing a threshold.

## Project files

```text
data/raw/learning_path_dataset.csv    Original input (read-only)
data/processed/train.csv              80% stratified training rows
data/processed/test.csv               20% stratified holdout rows
src/pipeline.py                       Audit, preprocessing, model, evaluation, recommender, EDA
main.py                               Beginner-friendly pipeline command line
app.py                                Streamlit application
models/auto_recommender.joblib        Fitted pipeline used by the app
reports/data_audit.md                 Schema, data quality, and limitations
reports/metrics.json                  CV comparison and held-out test results
reports/figures/                      EDA and model comparison charts
tests/test_pipeline.py                Automated project checks
```

## Windows and VS Code setup

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Use the interpreter selected for the `.venv` in VS Code.

## Run pipeline stages

```powershell
python main.py --step audit
python main.py --step preprocess
python main.py --step eda
python main.py --step train
python main.py --step evaluate
```

Or run the ordered workflow:

```powershell
python main.py --step all
```

`preprocess` creates `data/processed/train.csv` and `data/processed/test.csv` from the actual CSV. The fixed-seed (42), stratified 80/20 split is row based; `Learner_ID` is unique in the supplied file. The training command compares a most-frequent DummyClassifier, Logistic Regression, Decision Tree, and Random Forest using 5-fold stratified cross-validation on training rows. It selects by mean CV macro F1, fits the chosen pipeline on training rows, and evaluates it once on the untouched test rows. All imputers, encoders, scaling, and course-history vocabulary are fit within the training pipeline. `evaluate` displays the saved test result; run `train` first for fresh results.

## Model and recommendation logic

The fitted pipeline one-hot encodes current course, learner level, and learning preference and imputes and scales numeric profile fields. It ranks the model's available target labels by `predict_proba`. The app accepts a profile with blank initial selections and excludes the current course and the courses the learner enters as completed.

The displayed score is a model ranking score among candidate labels. It is not a calibrated probability of passing and does not guarantee benefit. Recommendations are alternative predictions because each dataset row has one next-course label. Student-entered completed-course history is not validated against completion records.

## Evaluation and limitations

`reports/metrics.json` records training CV comparison, accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, classification report, confusion matrix, top-k hit rates, and NDCG@5. Use those observed values as the project results; do not substitute training scores for test accuracy. The target equals the current course in 83 rows. Those rows stay in evaluation, although the app excludes the current course from its outputs, so the original label cannot be recommended for them. A row-level random split is reproducible but cannot measure future-period performance. The CSV has no timestamps, and dataset IDs alone do not establish real-world learner identity. The dataset has 12 fairly balanced target labels and only five usable model features, which may limit predictive accuracy.

## Launch Streamlit

Train the artifact first, then run:

```powershell
streamlit run app.py
```

The app loads the saved model and does not retrain on page refresh. If the artifact is absent, run `python main.py --step all`.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests check the actual dataset schema, target, splitting, course-history parsing, recommendation exclusions, and saved-pipeline interface.

## Future improvements

Collect timestamped learner-course events and verified completion outcomes; record meaningful student grouping keys; add a catalog with course metadata and resource URLs; and evaluate with learner-grouped or future-time splits. Those changes would support a more defensible personalized recommendation evaluation.
