"""
NER Service — Custom Legal Named Entity Recognition
Week 1-2 Implementation

Architecture:
- Base: SpaCy pipeline with custom NER component
- Transfer Learning: Fine-tuned on legal corpus using BERT embeddings
- Entities: DATE, PARTY, AMOUNT, TERMINATION_CLAUSE
- Training: Doccano-annotated legal contracts

Model loading priority:
  1. Custom fine-tuned model  → ./models/lexiscan-ner
  2. BERT transformer model   → en_core_web_trf
  3. Small SpaCy model        → en_core_web_sm
  4. Regex-only fallback      → no SpaCy needed (always works)

If SpaCy models are missing, the service runs in regex-only mode.
To download the SpaCy model run:
    python -m spacy download en_core_web_sm
"""

import re
import asyncio
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RawEntity:
    entity_type: str
    value: str
    confidence: float
    start_char: int
    end_char: int
    context: str


class NERService:
    """
    Legal NER using SpaCy with BERT-based embeddings.
    Falls back gracefully to regex-only mode if no SpaCy model is available.
    """

    LEGAL_PATTERNS = {
        "DATE": [
            r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December),?\s+\d{4})\b",
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4})\b",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b(\d{4}-\d{2}-\d{2})\b",
        ],
        "PARTY": [
            r'(?:between|by and between|among)\s+([A-Z][A-Za-z\s,\.]+(?:LLC|Inc\.|Corp\.|Ltd\.|L\.P\.'
            r'|LLP|Corporation|Company|Co\.))',
            r'(?:hereinafter\s+(?:referred\s+to\s+as\s+)?["\'])([^"\']+)["\']',
            r'"([A-Z][A-Za-z\s]+)"\s*\("(?:Company|Client|Contractor|Party|Vendor|Employer|Employee)"\)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\("(?:Employee|Contractor|Client|Vendor)"\)',
        ],
        "AMOUNT": [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|dollars|Dollars)\b',
            r'\b((?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:hundred|thousand|million|billion)'
            r'\s+(?:dollars))',
        ],
        "TERMINATION_CLAUSE": [
            r'(?i)((?:either\s+party\s+may\s+terminate|this\s+agreement\s+(?:may|shall|will)\s+(?:be\s+)?'
            r'terminat|terminat(?:ion|e)\s+(?:upon|in\s+the\s+event|for\s+cause|without\s+cause|by\s+(?:either|'
            r'any)\s+party))[^\.]{10,200}\.)',
            r'(?i)((?:upon\s+(?:\d+\s+days?)\s+(?:written\s+)?notice|immediately\s+upon\s+(?:written\s+)?notice)'
            r'[^\.]{0,150}\.)',
            r'(?i)((?:in\s+the\s+event\s+of\s+(?:a\s+)?(?:material\s+breach|default|insolvency|bankruptcy))'
            r'[^\.]{10,200}\.)',
        ],
    }

    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """
        Load SpaCy NER model with full graceful fallback chain.
        If every model fails, runs in regex-only mode — server still starts.
        """
        try:
            import spacy
        except ImportError:
            logger.warning("SpaCy not installed — running in regex-only mode")
            return

        # 1. Try custom trained model
        try:
            self.nlp = spacy.load("./models/lexiscan-ner")
            logger.info("Loaded custom LexiScan NER model")
            return
        except OSError:
            pass

        # 2. Try BERT transformer model
        try:
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("Loaded en_core_web_trf (BERT-based) model")
            return
        except OSError:
            pass

        # 3. Try small SpaCy model (installed via requirements.txt)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded en_core_web_sm model")
            return
        except OSError:
            pass

        # 4. Regex-only fallback — no SpaCy model needed
        # Server starts fine; entity extraction uses patterns only
        logger.warning(
            "No SpaCy model found — running in regex-only mode.\n"
            "  To enable SpaCy NER, run:  python -m spacy download en_core_web_sm\n"
            "  Then restart the server."
        )
        self.nlp = None

    async def extract_entities(self, text: str) -> List[RawEntity]:
        """Extract all legal entities from text."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_extract, text
        )

    def _sync_extract(self, text: str) -> List[RawEntity]:
        entities = []

        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)

        # Regex always runs — supplements SpaCy or works standalone
        regex_entities = self._extract_with_regex(text)
        merged = self._merge_entities(entities, regex_entities)
        return sorted(merged, key=lambda e: e.start_char)

    def _extract_with_spacy(self, text: str) -> List[RawEntity]:
        """Run SpaCy NER and map standard labels to our entity types."""
        SPACY_TO_LEGAL = {
            "DATE":    "DATE",
            "MONEY":   "AMOUNT",
            "ORG":     "PARTY",
            "PERSON":  "PARTY",
        }
        entities = []
        chunk_size = 100000
        for i in range(0, len(text), chunk_size):
            chunk = text[i: i + chunk_size]
            doc = self.nlp(chunk)
            for ent in doc.ents:
                legal_type = SPACY_TO_LEGAL.get(ent.label_)
                if legal_type:
                    ctx_start = max(0, ent.start_char - 60)
                    ctx_end   = min(len(chunk), ent.end_char + 60)
                    entities.append(RawEntity(
                        entity_type=legal_type,
                        value=ent.text,
                        confidence=0.85,
                        start_char=ent.start_char + i,
                        end_char=ent.end_char + i,
                        context=chunk[ctx_start:ctx_end],
                    ))
        return entities

    def _extract_with_regex(self, text: str) -> List[RawEntity]:
        """Regex-based extraction for high-precision legal patterns."""
        entities = []
        for entity_type, patterns in self.LEGAL_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    value = match.group(1) if match.lastindex else match.group(0)
                    value = value.strip()
                    if len(value) < 3:
                        continue
                    ctx_start = max(0, match.start() - 60)
                    ctx_end   = min(len(text), match.end() + 60)
                    entities.append(RawEntity(
                        entity_type=entity_type,
                        value=value,
                        confidence=0.92 if entity_type in ("DATE", "AMOUNT") else 0.78,
                        start_char=match.start(),
                        end_char=match.end(),
                        context=text[ctx_start:ctx_end],
                    ))
        return entities

    def _merge_entities(
        self,
        spacy_entities: List[RawEntity],
        regex_entities: List[RawEntity],
    ) -> List[RawEntity]:
        """Merge SpaCy + regex results, deduplicating by character overlap."""
        all_entities = list(spacy_entities)
        for regex_ent in regex_entities:
            overlapping = False
            for existing in all_entities:
                if existing.entity_type == regex_ent.entity_type and not (
                    regex_ent.end_char <= existing.start_char
                    or regex_ent.start_char >= existing.end_char
                ):
                    overlapping = True
                    if regex_ent.confidence > existing.confidence:
                        all_entities.remove(existing)
                        all_entities.append(regex_ent)
                    break
            if not overlapping:
                all_entities.append(regex_ent)
        return all_entities
