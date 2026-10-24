"""Backward-compatible shorthand for training and evaluating GR-1."""
from src.pipeline import train_and_evaluate


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Test accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['f1_macro']:.4f} | NDCG@5: {metrics['ndcg_at_5']:.4f}")
