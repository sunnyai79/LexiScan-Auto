"""
Validation Service — Rule-Based Post-Processing
Week 3 Implementation

Validates and normalizes raw NER output:
- DATE: normalize to ISO 8601 (YYYY-MM-DD)
- AMOUNT: ensure currency symbol, normalize to float
- PARTY: normalize whitespace, remove spurious tokens
- TERMINATION_CLAUSE: check minimum length, keyword presence
"""

import re
import logging
from typing import List, Optional
from datetime import datetime

from app.services.ner_service import RawEntity
from app.models.schemas import EntityResult

logger = logging.getLogger(__name__)

# Month name mappings
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

TERMINATION_KEYWORDS = {
    "terminat", "cancel", "expir", "rescind", "dissolv",
    "withdraw", "notice", "breach", "default", "end", "cease",
}


class ValidationService:
    """
    Post-processes NER output to ensure entity quality.
    
    Each validator:
    1. Checks if the entity is valid
    2. Normalizes the value if valid
    3. Returns validation notes for audit trail
    """

    def validate_and_normalize(self, raw_entities: List[RawEntity]) -> List[EntityResult]:
        """Validate and normalize all extracted entities."""
        results = []
        for entity in raw_entities:
            result = self._validate_entity(entity)
            results.append(result)
        
        # Deduplicate normalized values within same entity type
        results = self._deduplicate(results)
        return results

    def _validate_entity(self, entity: RawEntity) -> EntityResult:
        """Route to correct validator based on entity type."""
        validators = {
            "DATE": self._validate_date,
            "PARTY": self._validate_party,
            "AMOUNT": self._validate_amount,
            "TERMINATION_CLAUSE": self._validate_termination,
        }
        validator = validators.get(entity.entity_type, self._validate_generic)
        valid, normalized, notes = validator(entity.value)

        return EntityResult(
            entity_type=entity.entity_type,
            value=entity.value,
            normalized_value=normalized,
            confidence=entity.confidence,
            start_char=entity.start_char,
            end_char=entity.end_char,
            context=entity.context,
            valid=valid,
            validation_notes=notes,
        )

    # ─────────────────────────────────────────────
    # DATE Validator
    # ─────────────────────────────────────────────

    def _validate_date(self, value: str) -> tuple:
        """
        Validates and normalizes dates to ISO 8601 format (YYYY-MM-DD).
        Rule: Must be parseable as a real calendar date.
        """
        value = value.strip()

        # Try ISO format first
        iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
        if iso_match:
            y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            if self._is_valid_date(y, m, d):
                return True, f"{y:04d}-{m:02d}-{d:02d}", "ISO format confirmed"

        # Try numeric formats: MM/DD/YYYY, MM-DD-YYYY, DD/MM/YYYY
        numeric = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", value)
        if numeric:
            a, b, c = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
            year = c if c > 99 else (2000 + c if c < 50 else 1900 + c)
            # Try MM/DD/YYYY first (US format)
            if self._is_valid_date(year, a, b):
                return True, f"{year:04d}-{a:02d}-{b:02d}", "Normalized from MM/DD/YYYY"
            # Try DD/MM/YYYY (European)
            if self._is_valid_date(year, b, a):
                return True, f"{year:04d}-{b:02d}-{a:02d}", "Normalized from DD/MM/YYYY"

        # Try written format: "January 15, 2024" or "15th day of March, 2023"
        written = re.search(
            r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?"
            r"(january|february|march|april|may|june|july|august|"
            r"september|october|november|december),?\s+(\d{4})",
            value, re.IGNORECASE
        )
        if not written:
            written = re.search(
                r"(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})",
                value, re.IGNORECASE
            )
            if written:
                month_str, day_str, year_str = written.group(1), written.group(2), written.group(3)
            else:
                month_str = day_str = year_str = None
        else:
            day_str, month_str, year_str = written.group(1), written.group(2), written.group(3)

        if month_str and day_str and year_str:
            month = MONTH_MAP.get(month_str.lower())
            day, year = int(day_str), int(year_str)
            if month and self._is_valid_date(year, month, day):
                return True, f"{year:04d}-{month:02d}-{day:02d}", "Normalized from written format"

        return False, None, f"Could not parse date: '{value}'"

    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        try:
            datetime(year, month, day)
            return 1900 <= year <= 2100
        except ValueError:
            return False

    # ─────────────────────────────────────────────
    # AMOUNT Validator
    # ─────────────────────────────────────────────

    def _validate_amount(self, value: str) -> tuple:
        """
        Validates financial amounts.
        Rule: Must contain numeric content; normalizes to '$X,XXX.XX' format.
        """
        value = value.strip()
        # Extract numeric portion
        numeric_match = re.search(r"[\d,]+(?:\.\d{2})?", value)
        if not numeric_match:
            return False, None, "No numeric content found"

        numeric_str = numeric_match.group(0).replace(",", "")
        try:
            amount = float(numeric_str)
        except ValueError:
            return False, None, "Could not parse numeric value"

        if amount <= 0:
            return False, None, "Amount must be positive"

        # Determine currency symbol
        currency = "$"
        if "USD" in value.upper():
            currency = "$"
        elif "EUR" in value.upper() or "€" in value:
            currency = "€"
        elif "GBP" in value.upper() or "£" in value:
            currency = "£"

        normalized = f"{currency}{amount:,.2f}"
        return True, normalized, "Amount validated and normalized"

    # ─────────────────────────────────────────────
    # PARTY Validator
    # ─────────────────────────────────────────────

    def _validate_party(self, value: str) -> tuple:
        """
        Validates party names.
        Rules: Min 2 chars, must have alphabetic content, cleaned of noise.
        """
        value = value.strip()
        value = re.sub(r"\s+", " ", value)
        # Remove leading/trailing punctuation
        value = value.strip('",\'.()')

        if len(value) < 2:
            return False, None, "Party name too short"
        if not re.search(r"[a-zA-Z]", value):
            return False, None, "No alphabetic characters"
        if len(value) > 200:
            return False, None, "Party name suspiciously long"

        # Normalize common abbreviations
        normalized = value
        replacements = [
            (r"\bInc\b\.?", "Inc."), (r"\bLLC\b\.?", "LLC"),
            (r"\bCorp\b\.?", "Corp."), (r"\bLtd\b\.?", "Ltd."),
            (r"\bL\.?P\.?\b", "L.P."), (r"\bLLP\b\.?", "LLP"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        return True, normalized, "Party name validated"

    # ─────────────────────────────────────────────
    # TERMINATION CLAUSE Validator
    # ─────────────────────────────────────────────

    def _validate_termination(self, value: str) -> tuple:
        """
        Validates termination clauses.
        Rules: Min 20 chars, must contain termination-related keywords.
        """
        value = value.strip()
        if len(value) < 20:
            return False, None, "Clause too short to be meaningful"

        value_lower = value.lower()
        found_keywords = [kw for kw in TERMINATION_KEYWORDS if kw in value_lower]
        if not found_keywords:
            return False, None, "Missing termination keywords"

        # Truncate very long clauses for display (store full in value)
        normalized = value if len(value) <= 500 else value[:497] + "..."
        return (
            True,
            normalized,
            f"Termination clause validated. Keywords: {', '.join(found_keywords[:3])}",
        )

    def _validate_generic(self, value: str) -> tuple:
        return True, value.strip(), "Generic entity — no validation applied"

    # ─────────────────────────────────────────────
    # Deduplication
    # ─────────────────────────────────────────────

    def _deduplicate(self, entities: List[EntityResult]) -> List[EntityResult]:
        """Remove duplicate normalized values within same entity type."""
        seen: dict = {}
        result = []
        for entity in entities:
            key = (entity.entity_type, (entity.normalized_value or entity.value).lower().strip())
            if key not in seen:
                seen[key] = True
                result.append(entity)
        return result
