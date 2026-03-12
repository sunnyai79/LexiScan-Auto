"""
Pydantic Schemas for LexiScan Auto API
Defines request/response models for strict data validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class EntityType(str, Enum):
    DATE = "DATE"
    PARTY = "PARTY"
    AMOUNT = "AMOUNT"
    TERMINATION_CLAUSE = "TERMINATION_CLAUSE"


class EntityResult(BaseModel):
    entity_type: EntityType = Field(..., description="Type of extracted legal entity")
    value: str = Field(..., description="Extracted entity value (raw text)")
    normalized_value: Optional[str] = Field(None, description="Normalized/standardized value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    start_char: Optional[int] = Field(None, description="Start character position in source text")
    end_char: Optional[int] = Field(None, description="End character position in source text")
    context: Optional[str] = Field(None, description="Surrounding text context")
    valid: bool = Field(True, description="Whether entity passed validation rules")
    validation_notes: Optional[str] = Field(None, description="Notes from validation step")

    class Config:
        use_enum_values = True


class ProcessingMetadata(BaseModel):
    document_type: str = Field(..., description="Input type: PDF, IMAGE, or TEXT")
    ocr_applied: bool = Field(..., description="Whether OCR was used")
    page_count: int = Field(..., description="Number of pages processed (1 for image/text)")
    word_count: int = Field(..., description="Total word count")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall OCR/extraction confidence")


class ExtractionResponse(BaseModel):
    success: bool
    filename: str
    entities: List[EntityResult]
    metadata: ProcessingMetadata
    raw_text_preview: Optional[str] = Field(None, description="First 500 chars of extracted text")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "contract_2024.pdf",
                "entities": [
                    {
                        "entity_type": "DATE",
                        "value": "January 15, 2024",
                        "normalized_value": "2024-01-15",
                        "confidence": 0.97,
                        "valid": True,
                    }
                ],
                "metadata": {
                    "document_type": "PDF",
                    "ocr_applied": False,
                    "page_count": 12,
                    "word_count": 4832,
                    "processing_time_ms": 742.3,
                    "confidence_score": 0.94,
                },
            }
        }


class TextExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw contract text to process")
    document_name: Optional[str] = Field(None, description="Optional document identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This Agreement is entered into as of January 15, 2024 between Acme Corporation ('Company') and John Smith ('Contractor')...",
                "document_name": "service_agreement_2024",
            }
        }
