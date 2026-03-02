"""
Week 2 - Transfer Learning: Fine-tuned BERT for Legal NER
Fine-tunes a pre-trained BERT/DistilBERT model on the custom annotated dataset.
Significantly boosts F1-scores through contextual embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ENTITY_LABELS = ["DATE", "PARTY", "AMOUNT", "TERMINATION_CLAUSE"]
BIO_LABELS = ["O"] + [f"B-{e}" for e in ENTITY_LABELS] + [f"I-{e}" for e in ENTITY_LABELS]
LABEL2IDX = {label: idx for idx, label in enumerate(BIO_LABELS)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}


class BertLegalNER:
    """
    Fine-tuned BERT NER model for legal contract entity extraction.

    Uses HuggingFace Transformers with:
    - Base model: distilbert-base-uncased (fast, ~66M params)
    - Token classification head on top
    - Fine-tuned on custom legal annotation data (Doccano export)

    Why BERT over BiLSTM?
    - Contextual embeddings capture long-range dependencies
    - Pre-training on large corpora transfers legal vocabulary knowledge
    - F1 improvements of 8-15 points on legal NER benchmarks
    """

    DEFAULT_MODEL = "distilbert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_path: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._loaded = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        """Lazy-load the model only when needed (saves memory at import time)."""
        if self._loaded:
            return
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                pipeline,
            )
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

        logger.info("Loading tokenizer: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Loading model: %s", self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(BIO_LABELS),
            id2label=IDX2LABEL,
            label2id=LABEL2IDX,
            ignore_mismatched_sizes=True,
        )
        self._loaded = True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        output_dir: str = "data/models/bert_ner",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
    ) -> Dict:
        """
        Fine-tune BERT on annotated legal NER data.

        Args:
            train_data: List of {"tokens": [...], "ner_tags": [...]} dicts
            val_data: Optional validation set
            output_dir: Where to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: AdamW learning rate

        Returns:
            Training metrics dict
        """
        self._ensure_loaded()

        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
            from datasets import Dataset
            import evaluate
        except ImportError:
            raise ImportError("Install: pip install transformers datasets evaluate seqeval")

        # Build HuggingFace datasets
        train_dataset = Dataset.from_list(
            [self._tokenize_and_align(d) for d in train_data]
        )
        eval_dataset = None
        if val_data:
            eval_dataset = Dataset.from_list(
                [self._tokenize_and_align(d) for d in val_data]
            )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Load seqeval for NER F1 metric
        seqeval = evaluate.load("seqeval")

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=2)
            true_preds = [
                [IDX2LABEL[p] for p, l in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            true_labels = [
                [IDX2LABEL[l] for l in label if l != -100]
                for label in labels
            ]
            results = seqeval.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            fp16=False,  # set True if GPU available
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting BERT fine-tuning for %d epochs...", epochs)
        train_result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info("Eval F1: %.4f", eval_metrics.get("eval_f1", 0))

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_entities(self, text: str) -> List[Dict]:
        """
        Run NER inference on text using fine-tuned BERT.

        Returns:
            [{"text": "January 1, 2024", "label": "DATE", "start": 12, "end": 27}, ...]
        """
        self._ensure_loaded()

        import torch

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        # Aggregate subword tokens → word-level entities
        entities = []
        current_entity = None

        for i, (label_id, (char_start, char_end)) in enumerate(
            zip(pred_ids, offset_mapping)
        ):
            # Skip special tokens [CLS], [SEP], padding
            if char_start == char_end == 0 and i > 0:
                continue

            label = IDX2LABEL.get(label_id, "O")
            token_text = text[char_start:char_end]

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token_text,
                    "label": label[2:],
                    "start": char_start,
                    "end": char_end,
                }
            elif label.startswith("I-") and current_entity:
                # Extend existing entity
                current_entity["text"] = text[current_entity["start"]: char_end]
                current_entity["end"] = char_end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize_and_align(self, example: Dict) -> Dict:
        """Align BIO labels with BERT's wordpiece tokenization."""
        tokenized = self.tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )
        word_ids = tokenized.word_ids()
        labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # special token
            elif word_idx != prev_word_idx:
                labels.append(LABEL2IDX.get(example["ner_tags"][word_idx], 0))
            else:
                # Continuation of same word: use I- label
                orig_label = example["ner_tags"][word_idx]
                if orig_label.startswith("B-"):
                    labels.append(LABEL2IDX.get("I-" + orig_label[2:], 0))
                else:
                    labels.append(LABEL2IDX.get(orig_label, 0))
            prev_word_idx = word_idx
        tokenized["labels"] = labels
        return tokenized

    def save(self, path: str) -> None:
        """Save fine-tuned model and tokenizer."""
        if self.model and self.tokenizer:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("BERT NER saved to %s", path)

    def load(self, path: str) -> None:
        """Load fine-tuned model from disk."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForTokenClassification.from_pretrained(path)
            self._loaded = True
            logger.info("BERT NER loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load BERT model: %s", e)
            raise
