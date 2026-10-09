"""Generic schema validation utilities using Pydantic."""

import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Type, TypeVar, Optional, Tuple, Dict
from pydantic import BaseModel, ValidationError
from system.config import config
from system.logging import get_logger

logger = get_logger(__name__)


# Specialized logger for validation errors
def setup_validation_logger():
    v_logger = logging.getLogger("validation_errors")
    if not v_logger.handlers:
        log_dir = config.paths.project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "validation_errors.log"

        handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s | %(levelname)-s | %(message)s")
        handler.setFormatter(formatter)
        v_logger.addHandler(handler)
        v_logger.setLevel(logging.INFO)
    return v_logger


val_logger = setup_validation_logger()

T = TypeVar("T", bound=BaseModel)


class SchemaValidator:
    """Generic validator for Pydantic models."""

    def __init__(self, model: Type[T]):
        """
        Initialize validator with a Pydantic model.

        Args:
            model: Pydantic model class to validate against
        """
        self.model = model

    def validate(
        self, data: Any, context: Optional[str] = None
    ) -> Tuple[bool, Optional[T], Optional[Dict[str, str]]]:
        """
        Validate data against the schema.

        Args:
            data: Data to validate (dict, JSON string, or Pydantic model instance)
            context: Optional string describing the source (e.g. image filename)

        Returns:
            Tuple of (is_valid, validated_model, error_dict)
            error_dict maps mongodb-style field paths to error messages.
        """
        raw_input = data
        try:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    errors = {"json": f"Invalid JSON: {str(e)}"}
                    self._log_failure(raw_input, errors, context)
                    return False, None, errors

            validated = self.model.model_validate(data)
            return True, validated, None

        except ValidationError as e:
            errors = self._get_field_errors(e)
            self._log_failure(raw_input, errors, context)
            return False, None, errors
        except Exception as e:
            errors = {"unexpected": str(e)}
            self._log_failure(raw_input, errors, context)
            return False, None, errors

    def _get_field_errors(self, error: ValidationError) -> Dict[str, str]:
        """Map Pydantic errors to MongoDB-style field paths."""
        field_errors = {}
        for err in error.errors():
            # Convert ('ocr', 'line_items', 0, 'raw_text') to 'ocr.line_items.0.raw_text'
            path = ".".join(str(x) for x in err["loc"])
            field_errors[path] = err["msg"]
        return field_errors

    def _log_failure(self, raw_data: Any, errors: Dict[str, str], context: Optional[str] = None):
        """Log detailed failure to the validation log in one clear message."""
        ctx_str = f" | Image: {context}" if context else ""
        log_msg = (
            f"Validation Failed{ctx_str}\n"
            f"  [Errors]: {json.dumps(errors, indent=2)}\n"
            f"  [Model Output]: {str(raw_data)}\n"
            f"{'-' * 80}"
        )
        val_logger.info(log_msg)

    def validate_json_string(
        self, json_str: str
    ) -> Tuple[bool, Optional[T], Optional[Dict[str, str]]]:
        """
        Convenience method to validate a JSON string.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        return self.validate(json_str)

    def is_valid(self, data: Any) -> bool:
        """
        Quick check if data is valid.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        is_valid, _, _ = self.validate(data)
        return is_valid
