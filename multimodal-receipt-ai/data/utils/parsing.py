"""Utilities for parsing and processing model outputs."""

import json
import re
from typing import Any, Dict, Optional
from system.logging import get_logger

logger = get_logger(__name__)


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from model response text.

    Handles:
    - Raw JSON strings
    - JSON wrapped in markdown code blocks (```json ... ``` or ``` ... ```)
    - Basic cleaning of common LLM artifacts
    """
    if not text:
        return None

    # 1. Try to find JSON in markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # 2. If no code blocks, look for something that looks like a JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
        else:
            json_str = text.strip()

    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Attempted to parse: {json_str}")
        return None


def print_parsed_response(data: Dict[str, Any]):
    """Pretty print parsed JSON to console using logger matching the teacher schema."""
    if not data:
        logger.warning("No valid data to print.")
        return

    output = []
    output.append("\n" + "=" * 50)
    output.append(f"STRUCTURED RECEIPT DATA [{data.get('document_type', 'unknown')}]")
    output.append("=" * 50)

    # OCR Namespace
    ocr = data.get("ocr", {})
    output.append(f"Merchant: {ocr.get('merchant_name', 'N/A')}")
    output.append(f"Date:     {ocr.get('date', 'N/A')} {ocr.get('time', '')}")
    output.append(f"Total:    {ocr.get('total', 'N/A')}")
    output.append(f"Method:   {ocr.get('payment_method', 'N/A')}")
    output.append(f"Confidence: {data.get('confidence', 'N/A')}")

    # Qualitative Namespace
    q = data.get("quality_flags", {})
    flags = [f for f, v in q.items() if v]
    if flags:
        output.append(f"Flags:    {', '.join(flags)}")

    output.append("-" * 30)
    output.append("OCR Line Items:")
    line_items = ocr.get("line_items", [])
    if not line_items:
        output.append("  (None)")
    else:
        for item in line_items[:10]:  # Show first 10 for brevity
            text = item.get("raw_text", "N/A")
            price = item.get("raw_price")
            if price is None:
                price = 0.0
            output.append(f"  - {text:<25} {price:>8.2f}")
        if len(line_items) > 10:
            output.append(f"  ... (+{len(line_items) - 10} more items)")

    output.append("-" * 30)
    output.append("Semantic Items (Summary):")
    semantics = data.get("semantics", {})
    items = semantics.get("items", [])
    if not items:
        output.append("  (None)")
    else:
        for item in items[:5]:
            name = item.get("canonical_name") or item.get("raw_text_ref", "N/A")
            cat = item.get("category", "N/A")
            price = item.get("total_price")
            if price is None:
                price = 0.0
            output.append(f"  * {name:<20} | {cat:<15} | {price:>8.2f}")

    output.append("=" * 50 + "\n")

    logger.info("\n".join(output))

    logger.info("\n".join(output))
