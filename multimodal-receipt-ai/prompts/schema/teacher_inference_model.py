"""Pydantic models for teacher inference schema validation."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class OCRLineItem(BaseModel):
    """Individual line item from OCR extraction."""

    raw_text: str
    raw_price: Optional[float] = None


class OCRData(BaseModel):
    """OCR extraction data."""

    merchant_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    line_items: List[OCRLineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None


class SemanticItem(BaseModel):
    """Semantic interpretation of a line item."""

    raw_text_ref: str
    canonical_name: Optional[str] = None
    category: Literal[
        "Food & Grocery",
        "Fresh Produce",
        "Bakery & Bread",
        "Beverages",
        "Personal Care & Hygiene",
        "Household & Cleaning",
        "OTC Health & Wellness",
        "Baby & Child Care",
        "Pet Care",
        "Other",
        "Unknown",
    ]
    quantity: Optional[float] = None
    unit_type: Literal["count", "weight", "volume", "unknown"] = "unknown"
    unit: Optional[str] = None
    price_per_unit: Optional[float] = None
    total_price: Optional[float] = None
    line_role: Literal[
        "product",
        "discount",
        "deposit_charge",
        "deposit_return",
        "tax",
        "subtotal",
        "total",
        "info",
        "unknown",
    ] = "unknown"
    semantic_confidence: float = Field(ge=0.0, le=1.0)


class Discount(BaseModel):
    """Detected discount."""

    description: str
    amount: Optional[float] = None


class Deposit(BaseModel):
    """Detected deposit."""

    description: str
    amount: Optional[float] = None


class Semantics(BaseModel):
    """Semantic interpretation data."""

    items: List[SemanticItem] = Field(default_factory=list)
    detected_discounts: List[Discount] = Field(default_factory=list)
    detected_deposits: List[Deposit] = Field(default_factory=list)


class QualityFlags(BaseModel):
    """Image quality indicators."""

    is_blurry: bool = False
    is_partial: bool = False
    is_rotated: bool = False
    is_low_contrast: bool = False


class TeacherInferenceResponse(BaseModel):
    """Complete teacher inference response schema."""

    document_type: Literal["receipt", "non_receipt", "empty_unreadable"]
    ocr: OCRData
    semantics: Semantics
    quality_flags: QualityFlags
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v
