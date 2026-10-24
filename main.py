"""Command-line entry point for the GR-1 project pipeline."""
import argparse
import json

from src.pipeline import (METRICS_PATH, audit_dataset, create_eda, split_data,
                          train_and_evaluate, write_audit_report)


def run(step):
    if step in {"audit", "all"}:
        df, audit = audit_dataset()
        write_audit_report(df, audit)
        print(f"Audit: {audit['rows']} rows, {audit['columns_count']} columns; target present: {audit['target_exists']}; exact duplicates: {audit['duplicate_rows']}.")
        if step == "audit":
            return
    if step in {"preprocess", "all"}:
        df, _ = audit_dataset()
        train, test = split_data(df)
        print(f"Saved actual-data split: {len(train)} training rows and {len(test)} test rows.")
        if step == "preprocess":
            return
    if step == "eda":
        df, audit = audit_dataset()
        write_audit_report(df, audit)
        create_eda(df)
        print("Saved EDA figures under reports/figures/.")
        return
    if step == "all":
        df, _ = audit_dataset()
        create_eda(df)
    if step in {"train", "all"}:
        metrics = train_and_evaluate()
        print(f"Selected model: {metrics['selected_model']}")
        print(f"Untouched holdout accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['f1_macro']:.4f}; NDCG@5: {metrics['ndcg_at_5']:.4f}")
        if step == "train":
            return
    if step == "evaluate":
        if not METRICS_PATH.is_file():
            raise FileNotFoundError("No evaluation report exists. Run `python main.py --step train` first.")
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        print(f"Saved test results for {metrics['selected_model']}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['f1_macro']:.4f}, NDCG@5={metrics['ndcg_at_5']:.4f}.")
        print(metrics["note"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a GR-1 project pipeline stage.")
    parser.add_argument("--step", choices=["audit", "preprocess", "eda", "train", "evaluate", "all"],
                        default="all", help="Pipeline stage to run (default: all).")
    run(parser.parse_args().step)
