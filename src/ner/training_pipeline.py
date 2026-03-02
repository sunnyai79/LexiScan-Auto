"""
Full training pipeline — orchestrates data loading, model training, and evaluation.
Run this script to train LexiScan Auto's NER models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ner.data_annotation import prepare_training_data, create_sample_annotations
from src.ner.bert_ner import BertLegalNER, IDX2LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model: BertLegalNER, test_data: list) -> dict:
    """Compute per-entity precision, recall, F1 on test set."""
    from collections import defaultdict

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for tokens, true_labels in test_data:
        text = " ".join(tokens)
        predicted = model.predict_entities(text)
        pred_spans = {(e["start"], e["end"], e["label"]) for e in predicted}

        # Build gold spans from BIO labels
        gold_spans = set()
        current = None
        char_offset = 0
        for token, label in zip(tokens, true_labels):
            start = char_offset
            end = start + len(token)
            char_offset = end + 1  # +1 for space
            if label.startswith("B-"):
                if current:
                    gold_spans.add(current)
                current = (start, end, label[2:])
            elif label.startswith("I-") and current:
                current = (current[0], end, current[2])
            else:
                if current:
                    gold_spans.add(current)
                    current = None
        if current:
            gold_spans.add(current)

        for span in pred_spans:
            if span in gold_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in gold_spans:
            if span not in pred_spans:
                fn[span[2]] += 1

    metrics = {}
    labels = set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))
    for label in labels:
        p = tp[label] / (tp[label] + fp[label] + 1e-9)
        r = tp[label] / (tp[label] + fn[label] + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        metrics[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    overall_p = total_tp / (total_tp + total_fp + 1e-9)
    overall_r = total_tp / (total_tp + total_fn + 1e-9)
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r + 1e-9)
    metrics["overall"] = {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LexiScan Auto NER")
    parser.add_argument("--annotation-file", default="data/annotated/contracts.jsonl")
    parser.add_argument("--model-output", default="data/models/bert_ner")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate synthetic training data for demo")
    args = parser.parse_args()

    if args.generate_samples or not Path(args.annotation_file).exists():
        logger.info("Generating sample annotation data...")
        create_sample_annotations(args.annotation_file, n=200)

    # Load and split data
    train_data, val_data, test_data = prepare_training_data(args.annotation_file)

    # Convert to BERT format
    train_bert = [{"tokens": t, "ner_tags": l} for t, l in train_data]
    val_bert = [{"tokens": t, "ner_tags": l} for t, l in val_data]

    # Train BERT NER
    logger.info("=== Training BERT Fine-tuned NER ===")
    model = BertLegalNER()
    model._ensure_loaded()
    metrics = model.train(
        train_bert,
        val_bert,
        output_dir=args.model_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Evaluate
    logger.info("=== Evaluating on Test Set ===")
    eval_metrics = evaluate_model(model, test_data)
    logger.info("Test Metrics:\n%s", json.dumps(eval_metrics, indent=2))

    # Save metrics
    metrics_path = Path(args.model_output) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
