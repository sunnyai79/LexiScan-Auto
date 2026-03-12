"""
LexiScan Auto — NER Model Training Script
Week 1-2: Custom NER with BERT Transfer Learning

Usage:
  python train_ner.py --data ./data/annotations.jsonl --output ./models/lexiscan-ner

Data Format (Doccano JSONL export):
  {"text": "Agreement dated January 15, 2024...", "label": [[17, 33, "DATE"], ...]}

Training Strategy:
  1. Convert Doccano annotations to SpaCy DocBin format
  2. Initialize with BERT embeddings (en_core_web_trf)
  3. Fine-tune NER head on legal contract data
  4. Evaluate F1 on held-out test set
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_doccano_annotations(filepath: str) -> List[Dict]:
    """Load Doccano JSONL annotations."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"Loaded {len(examples)} annotated documents")
    return examples


def convert_to_spacy_format(
    annotations: List[Dict],
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Convert Doccano format to SpaCy NER training format.
    Splits into 80/20 train/test.
    Returns: (train_data, test_data)
    """
    spacy_data = []
    skipped = 0

    for ann in annotations:
        text = ann["text"]
        entities = []
        for start, end, label in ann.get("label", []):
            # Validate entity boundaries
            if start < 0 or end > len(text) or start >= end:
                skipped += 1
                continue
            # Map to our entity types
            mapped_label = _map_label(label)
            if mapped_label:
                entities.append((start, end, mapped_label))

        # Remove overlapping entities (SpaCy requirement)
        entities = _remove_overlaps(entities)
        spacy_data.append((text, {"entities": entities}))

    logger.info(f"Converted {len(spacy_data)} examples ({skipped} entities skipped)")

    # Train/test split
    random.shuffle(spacy_data)
    split_idx = int(len(spacy_data) * 0.8)
    return spacy_data[:split_idx], spacy_data[split_idx:]


def _map_label(label: str) -> str:
    """Map Doccano labels to SpaCy NER labels."""
    mapping = {
        "DATE": "DATE", "date": "DATE",
        "PARTY": "PARTY", "party": "PARTY", "ORG": "PARTY",
        "AMOUNT": "AMOUNT", "amount": "AMOUNT", "MONEY": "AMOUNT",
        "TERMINATION": "TERMINATION_CLAUSE",
        "TERMINATION_CLAUSE": "TERMINATION_CLAUSE",
    }
    return mapping.get(label)


def _remove_overlaps(entities: List[Tuple]) -> List[Tuple]:
    """Remove overlapping entity spans."""
    sorted_ents = sorted(entities, key=lambda e: e[0])
    result = []
    last_end = 0
    for start, end, label in sorted_ents:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return result


def create_docbin(nlp, data: List[Tuple]) -> DocBin:
    """Convert training data to SpaCy DocBin for efficient storage."""
    db = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        db.add(example.reference)
    return db


def train_model(
    train_data: List[Tuple],
    test_data: List[Tuple],
    output_dir: str,
    n_iter: int = 30,
    dropout: float = 0.2,
):
    """
    Fine-tune SpaCy NER model on legal contract data.
    
    Uses:
    - Transfer learning from en_core_web_trf (BERT)
    - Custom NER head for our 4 entity types
    - Dropout for regularization
    """
    # Load base model
    logger.info("Loading base transformer model...")
    nlp = spacy.load("en_core_web_trf", exclude=["ner"])

    # Add custom NER component
    ner = nlp.add_pipe("ner", last=True)
    for _, annotations in train_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    # Training
    optimizer = nlp.begin_training()
    best_f1 = 0.0

    logger.info(f"Starting training for {n_iter} iterations...")
    for iteration in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        examples = []

        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        # Batch training
        batches = spacy.util.minibatch(examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=dropout, losses=losses)

        # Evaluate on test set
        if (iteration + 1) % 5 == 0:
            f1 = evaluate_model(nlp, test_data)
            logger.info(
                f"Iter {iteration + 1}/{n_iter} — NER Loss: {losses.get('ner', 0):.4f} — F1: {f1:.4f}"
            )
            if f1 > best_f1:
                best_f1 = f1
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                nlp.to_disk(output_dir)
                logger.info(f"  ✓ New best model saved (F1={f1:.4f})")

    logger.info(f"\nTraining complete. Best F1: {best_f1:.4f}")
    return nlp


def evaluate_model(nlp, test_data: List[Tuple]) -> float:
    """Evaluate NER F1 score on test set."""
    scorer = spacy.scorer.Scorer()
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        pred = nlp(text)
        example.predicted = pred
        examples.append(example)

    scores = scorer.score(examples)
    return scores.get("ents_f", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train LexiScan NER model")
    parser.add_argument("--data", required=True, help="Path to Doccano JSONL annotations")
    parser.add_argument("--output", default="./models/lexiscan-ner", help="Output model directory")
    parser.add_argument("--iter", type=int, default=30, help="Training iterations")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    args = parser.parse_args()

    annotations = load_doccano_annotations(args.data)
    train_data, test_data = convert_to_spacy_format(annotations)
    logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")

    train_model(train_data, test_data, args.output, args.iter, args.dropout)


if __name__ == "__main__":
    main()
