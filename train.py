from src.pipeline import train_and_evaluate, infer_target, audit_dataset

if __name__ == "__main__":
    df, audit = audit_dataset()
    target = infer_target(df)
    if target is None:
        print("Could not infer one target column. Set it in app.py and train from the UI.")
    else:
        print("Detected target:", target)
        metrics = train_and_evaluate(target)
        print("Rows:", metrics["rows"])
        print("Train rows:", metrics["train_rows"], "| Test rows:", metrics["test_rows"])
        print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
        print("Balanced accuracy:", round(metrics["balanced_accuracy"] * 100, 2), "%")
        print("Identifier columns excluded:", metrics["identifier_columns_excluded"])
        print("Saved model and reports.")
