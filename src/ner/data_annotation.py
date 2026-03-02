"""
Week 2 - Data Annotation Utilities
Converts Doccano JSON export format to training-ready token/label sequences.
Supports both sentence-level and document-level annotation.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENTITY_LABELS = ["DATE", "PARTY", "AMOUNT", "TERMINATION_CLAUSE"]


def load_doccano_export(filepath: str) -> List[Dict]:
    """
    Load Doccano JSONL export file.

    Doccano exports one JSON object per line:
    {"text": "...", "entities": [[start, end, label], ...]}
    """
    data = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: JSON parse error: %s", line_num, e)

    logger.info("Loaded %d annotated documents from %s", len(data), filepath)
    return data


def doccano_to_bio(annotation: Dict, window_size: int = 128) -> List[Tuple[List[str], List[str]]]:
    """
    Convert a single Doccano annotation to BIO-tagged sentence windows.

    Args:
        annotation: {"text": str, "entities": [[start, end, label], ...]}
        window_size: Max tokens per training example (sliding window)

    Returns:
        List of (tokens, bio_labels) tuples
    """
    text = annotation["text"]
    entities = annotation.get("entities", [])

    # Tokenize (simple whitespace; production should use spaCy tokenizer)
    words = text.split()

    # Map each character offset to a word index
    char_to_word = {}
    char_offset = 0
    for word_idx, word in enumerate(words):
        start = text.find(word, char_offset)
        for c in range(start, start + len(word)):
            char_to_word[c] = word_idx
        char_offset = start + len(word)

    # Initialize all labels as O
    labels = ["O"] * len(words)

    # Apply entity spans using BIO scheme
    for span in entities:
        if len(span) < 3:
            continue
        e_start, e_end, e_label = span[0], span[1], span[2].upper()

        if e_label not in ENTITY_LABELS:
            logger.debug("Skipping unknown label: %s", e_label)
            continue

        # Find word indices covered by this span
        covered_words = set()
        for c in range(e_start, e_end):
            if c in char_to_word:
                covered_words.add(char_to_word[c])

        if not covered_words:
            continue

        sorted_words = sorted(covered_words)
        labels[sorted_words[0]] = f"B-{e_label}"
        for w in sorted_words[1:]:
            labels[w] = f"I-{e_label}"

    # Split into sliding windows
    examples = []
    for i in range(0, max(1, len(words) - window_size + 1), window_size // 2):
        w_tokens = words[i: i + window_size]
        w_labels = labels[i: i + window_size]
        if w_tokens:
            examples.append((w_tokens, w_labels))

    return examples


def prepare_training_data(
    doccano_file: str,
    val_split: float = 0.15,
    test_split: float = 0.05,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Load and split annotated data into train/val/test sets.

    Returns:
        (train_examples, val_examples, test_examples)
        Each example is (tokens_list, bio_labels_list)
    """
    raw_data = load_doccano_export(doccano_file)
    all_examples = []
    for annotation in raw_data:
        examples = doccano_to_bio(annotation)
        all_examples.extend(examples)

    random.seed(seed)
    random.shuffle(all_examples)

    n = len(all_examples)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test = all_examples[:n_test]
    val = all_examples[n_test: n_test + n_val]
    train = all_examples[n_test + n_val:]

    logger.info(
        "Data split: train=%d, val=%d, test=%d examples", len(train), len(val), len(test)
    )
    return train, val, test


def create_sample_annotations(output_path: str, n: int = 50) -> str:
    """
    Generate synthetic annotated examples for demonstration/testing.
    In production, replace with Doccano-annotated real contract data.
    """
    import random

    sample_contracts = [
        {
            "text": "This Agreement is entered into as of January 15, 2024, by and between Acme Corporation LLC and Beta Partners Inc. The total consideration shall be $750,000.00. Either party may terminate this Agreement upon 30 days written notice.",
            "entities": [
                [44, 60, "DATE"],
                [78, 99, "PARTY"],
                [104, 122, "PARTY"],
                [154, 165, "AMOUNT"],
                [168, 241, "TERMINATION_CLAUSE"],
            ],
        },
        {
            "text": "Effective March 1, 2023, GlobalFinance Ltd. agrees to pay XYZ Holdings Corp $2,500,000 USD. This contract expires December 31, 2025. Upon breach, the non-breaching party may cancel this agreement immediately.",
            "entities": [
                [10, 24, "DATE"],
                [25, 42, "PARTY"],
                [53, 70, "PARTY"],
                [71, 88, "AMOUNT"],
                [104, 121, "DATE"],
                [124, 206, "TERMINATION_CLAUSE"],
            ],
        },
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for _ in range(n):
            sample = random.choice(sample_contracts)
            f.write(json.dumps(sample) + "\n")

    logger.info("Wrote %d sample annotations to %s", n, output_path)
    return str(path)
