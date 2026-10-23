
# Automatic Attribute Personalized Learning Recommender

This project inspects the supplied dataset, discovers its actual attributes, lets the user choose the prediction target, generates a manual input form from feature columns, and recommends predicted target classes.

## Dataset used
`data/raw/learning_path_dataset.csv` is the uploaded `learning_path_dataset(3).csv`.

Observed schema: 1,000 rows, 8 columns, no missing values, no duplicate rows. Columns:
- `Learner_ID`
- `Current_Course`
- `Completed_Courses`
- `Learning_Level`
- `Learning_Preference`
- `Time_Spent_Hours`
- `Performance_Score`
- `Next_Course_Recommendation`

`Next_Course_Recommendation` is automatically suggested as target because its name indicates a recommendation label. You can choose the target in the sidebar. `Learner_ID` is detected as an identifier and excluded from model features.

## Run in VS Code (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If `train.py` doesn't detect a target, start Streamlit, choose the target in the sidebar, and click Train and evaluate model.

## Leakage control
The data is split 80/20 before training. The scikit-learn pipeline fits imputers, encoders, scaler and model only on training rows. Identifier columns are excluded. Test data is only used to compute evaluation metrics.

## Important limits
This dataset predicts next-course labels, not OULAD `id_site` / `activity_type` learning resources. For resource recommendations, a dataset with resource-level interactions and metadata is required. Accuracy cannot be promised; use held-out test results and do not tune against the test set to force 70–80%.

# Learning-recommendation-system

