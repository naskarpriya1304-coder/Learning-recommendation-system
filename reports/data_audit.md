# GR-1 data audit

## Source and schema

Read-only input: `C:\Users\user\Desktop\Learning recommendation System\data\raw\learning_path_dataset.csv`. Pandas loaded 1000 rows and 8 columns. The exact target column `Next_Course_Recommendation` is present. There are 0 exact duplicate rows. Missing values and distinct counts below are from the original CSV before normalization.

| Column | pandas dtype | Missing | Distinct non-null |
|---|---|---:|---:|
| Learner_ID | int64 | 0 | 1000 |
| Current_Course | str | 0 | 12 |
| Completed_Courses | str | 0 | 597 |
| Learning_Level | str | 0 | 3 |
| Learning_Preference | str | 0 | 4 |
| Time_Spent_Hours | float64 | 0 | 949 |
| Performance_Score | float64 | 0 | 901 |
| Next_Course_Recommendation | str | 0 | 12 |

## Target class distribution

| Target label | Rows |
|---|---:|
| Math101 | 102 |
| History101 | 93 |
| CS101 | 87 |
| Math102 | 86 |
| Physics102 | 85 |
| Chem102 | 84 |
| History102 | 84 |
| Chem101 | 78 |
| Physics101 | 77 |
| CS102 | 77 |
| Bio102 | 76 |
| Bio101 | 71 |

## Validation and use

The model uses these observed profile columns: `Current_Course, Learning_Level, Learning_Preference, Time_Spent_Hours, Performance_Score`. `Learner_ID`, `Completed_Courses`, and the target are excluded from predictors. The dataset has no timestamps, and the target appears in `Completed_Courses` for 217 rows; history is therefore not used as a feature. The target equals `Current_Course` for 83 rows. Current course is available at prediction time, but these labels cannot be returned by the app because it excludes the current course. These cases remain in the evaluation and are a task-data limitation. Completed-course values entered in the app are only used to filter those labels from recommendations. Categories are imputed and one-hot encoded, and numeric values are imputed using training medians and standardized, all inside the model pipeline. Numeric observations outside 0–1000 hours or 0–100 score are flagged in `reports/metrics.json`; they are retained and are not silently clipped.

There are no resource IDs/types, timestamps, assessment records, reliable completion events, or independent learner interactions. The target supports prediction of a next-course label only. It cannot establish learning-resource effectiveness or predict academic success. The random row split is appropriate for reproducing this dataset's row-level task; the source does not support temporal evaluation. Check whether `Learner_ID` represents repeated people before interpreting results. In this file it has 1000 distinct values across 1000 rows.

## Generated outputs

`data/processed/train.csv` and `data/processed/test.csv` are an 80/20 stratified split derived from this input, seed 42. Figures are saved under `reports/figures/`. See `reports/metrics.json` for training-only cross-validation model comparison and the single held-out test evaluation.
