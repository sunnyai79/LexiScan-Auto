"""
Week 1 - Custom NER Model
Bi-Directional LSTM with character and word embeddings for legal NER.
Trained on annotated legal contract data (Doccano format).
Entities: DATE, PARTY, AMOUNT, TERMINATION_CLAUSE
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Entity labels (BIO tagging scheme)
ENTITY_LABELS = ["DATE", "PARTY", "AMOUNT", "TERMINATION_CLAUSE"]
BIO_LABELS = ["O"] + [f"B-{e}" for e in ENTITY_LABELS] + [f"I-{e}" for e in ENTITY_LABELS]
LABEL2IDX = {label: idx for idx, label in enumerate(BIO_LABELS)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class LegalNERModel:
    """
    BiLSTM-CRF NER model for legal contract entity extraction.
    Uses TensorFlow/Keras for deep learning.

    Architecture:
        Embedding → BiLSTM → Dense → CRF (linear chain)

    Usage:
        model = LegalNERModel()
        model.build(vocab_size=10000, embedding_dim=100)
        model.train(train_data, val_data)
        entities = model.predict_entities("This Agreement is dated January 1, 2024...")
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.max_seq_len = 128

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        lstm_units: int = 128,
        dropout: float = 0.3,
    ):
        """Build the BiLSTM-CRF model graph."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required: pip install tensorflow")

        num_labels = len(BIO_LABELS)

        inputs = keras.Input(shape=(self.max_seq_len,), name="word_ids")

        # Word embedding
        x = keras.layers.Embedding(
            input_dim=vocab_size + 2,  # +2 for PAD and UNK
            output_dim=embedding_dim,
            mask_zero=True,
            name="word_embedding",
        )(inputs)

        x = keras.layers.Dropout(dropout)(x)

        # BiLSTM layers
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
            name="bilstm_1",
        )(x)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout),
            name="bilstm_2",
        )(x)

        # Dense projection
        logits = keras.layers.Dense(num_labels, activation="softmax", name="logits")(x)

        self.model = keras.Model(inputs=inputs, outputs=logits, name="LegalNER_BiLSTM")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("BiLSTM model built: %s", self.model.summary())

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocab(self, texts: List[str]) -> None:
        """Build word vocabulary from training texts."""
        from collections import Counter

        word_counts: Counter = Counter()
        for text in texts:
            word_counts.update(text.lower().split())

        # Reserve 0 for PAD, 1 for UNK
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, _ in word_counts.most_common(50000):
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        logger.info("Vocabulary size: %d", len(self.word2idx))

    def tokenize(self, text: str) -> List[int]:
        """Convert text to word ID sequence (padded/truncated to max_seq_len)."""
        tokens = text.lower().split()[: self.max_seq_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # Pad
        ids += [0] * (self.max_seq_len - len(ids))
        return ids

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_sentences: List[Tuple[List[str], List[str]]],
        val_sentences: Optional[List[Tuple[List[str], List[str]]]] = None,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> None:
        """
        Train the model.

        Args:
            train_sentences: List of (tokens, bio_labels) tuples
            val_sentences: Optional validation set
            epochs: Training epochs
            batch_size: Batch size
        """
        if self.model is None:
            raise RuntimeError("Call build() first")

        X_train, y_train = self._prepare_data(train_sentences)
        fit_kwargs = dict(epochs=epochs, batch_size=batch_size, verbose=1)

        if val_sentences:
            X_val, y_val = self._prepare_data(val_sentences)
            fit_kwargs["validation_data"] = (X_val, y_val)

        self.model.fit(X_train, y_train, **fit_kwargs)

    def _prepare_data(
        self, sentences: List[Tuple[List[str], List[str]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert token/label sequences to padded numpy arrays."""
        X, y = [], []
        for tokens, labels in sentences:
            token_ids = [self.word2idx.get(t.lower(), 1) for t in tokens][: self.max_seq_len]
            label_ids = [LABEL2IDX.get(l, 0) for l in labels][: self.max_seq_len]
            # Pad
            pad_len = self.max_seq_len - len(token_ids)
            token_ids += [0] * pad_len
            label_ids += [0] * pad_len
            X.append(token_ids)
            y.append(label_ids)
        return np.array(X), np.array(y)[..., np.newaxis]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_entities(self, text: str) -> List[Dict]:
        """
        Run NER on text and return list of extracted entities.

        Returns:
            [{"text": "...", "label": "DATE", "start": 12, "end": 25}, ...]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build()+train() or load().")

        tokens = text.split()
        token_ids = np.array([self.tokenize(text)])
        probs = self.model.predict(token_ids, verbose=0)[0]  # (seq_len, num_labels)
        label_ids = np.argmax(probs, axis=-1)

        # Reconstruct character offsets
        entities = []
        current_entity = None
        char_offset = 0

        for i, (token, label_id) in enumerate(zip(tokens, label_ids[: len(tokens)])):
            label = IDX2LABEL.get(label_id, "O")
            token_start = text.find(token, char_offset)
            token_end = token_start + len(token)
            char_offset = token_end

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": label[2:],
                    "start": token_start,
                    "end": token_end,
                    "tokens": [token],
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " " + token
                current_entity["end"] = token_end
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        # Clean up internal key
        for e in entities:
            e.pop("tokens", None)

        return entities

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and vocabulary."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p / "bilstm_model.keras"))
        with open(p / "vocab.pkl", "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and vocabulary."""
        try:
            from tensorflow import keras
            p = Path(path)
            self.model = keras.models.load_model(str(p / "bilstm_model.keras"))
            with open(p / "vocab.pkl", "rb") as f:
                vocab = pickle.load(f)
            self.word2idx = vocab["word2idx"]
            self.idx2word = vocab["idx2word"]
            logger.info("Model loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

if __name__ == "__main__":
    print("1. Initializing BiLSTM Model...")
    model = LegalNERModel()

    # 2. Create tiny dummy training data (Tokens and BIO tags)
    train_data = [
        (
            ["This", "agreement", "dated", "January", "15", ",", "2024", "with", "Acme", "Corp", "for", "$500"],
            ["O", "O", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O", "B-PARTY", "I-PARTY", "O", "B-AMOUNT"]
        )
    ]
    
    # Extract just the raw text strings to build the vocabulary
    raw_texts = [" ".join(tokens) for tokens, labels in train_data]

    # 3. Build Vocabulary and Architecture
    print("2. Building Vocabulary and Model Architecture...")
    model.build_vocab(raw_texts)
    
    # We pass the vocabulary size we just built
    vocab_size = len(model.word2idx)
    model.build(vocab_size=vocab_size, embedding_dim=50, lstm_units=64)

    # 4. Train the model
    print("3. Training the model on dummy data...")
    model.train(train_sentences=train_data, epochs=10, batch_size=1)

    # 5. Test Inference
    print("4. Testing Extraction...")
    test_text = "This agreement dated January 15 , 2024 with Acme Corp for $500"
    extracted_entities = model.predict_entities(test_text)
    
    print("\n--- EXTRACTED ENTITIES ---")
    import json
    print(json.dumps(extracted_entities, indent=2))