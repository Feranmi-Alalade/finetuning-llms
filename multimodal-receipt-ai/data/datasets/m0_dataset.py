"""M0 OCR Dataset for Fine-tuning.

Loads images from the gold/training/m0_ocr directory structure and matches
them with teacher-generated labels for supervised fine-tuning.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, List, Any
from PIL import Image
import json
import yaml

from system.config import config
from system.logging import get_logger

logger = get_logger(__name__)


class M0Dataset(Dataset):
    """Dataset for M0 OCR fine-tuning.

    Loads images from gold/student/training/m0_ocr/data/{split}/
    and labels from gold/student/training/m0_ocr/labels/{split}/.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path,
        processor,
        prompt_template: str,
        max_length: Optional[int] = None,
        use_custom_losses: bool = True,
    ):
        """Initialize M0Dataset.

        Args:
            split: 'train' or 'test'
            data_dir: Path to gold/student/training/m0_ocr/
            processor: Qwen processor for tokenization
            prompt_template: Prompt template from teacher_inference.yaml
            max_length: Maximum sequence length
            use_custom_losses: Whether to generate auxiliary labels for custom losses
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_length = max_length or config.training.max_seq_length
        self.max_pixels = config.data_prep.max_pixels
        self.IGNORE_INDEX = config.data.ignore_index
        self.use_custom_losses = use_custom_losses

        # Discover all images in the split directory
        self.samples = self._discover_samples()

        logger.info(
            f"Initialized M0Dataset for {split} with {len(self.samples)} samples "
            f"(custom_losses={'enabled' if use_custom_losses else 'disabled'})"
        )

    def _discover_samples(self) -> List[Dict[str, Any]]:
        """Discover all images and their labels in the split directory."""
        data_dir = self.data_dir / "data" / self.split
        labels_dir = self.data_dir / "labels" / self.split

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        samples = []

        # Discover all images
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(data_dir.glob(ext)):
                label_path = labels_dir / f"{img_path.stem}.json"

                if label_path.exists():
                    try:
                        with open(label_path, "r") as f:
                            label = json.load(f)
                        samples.append({"image_path": img_path, "label": label})
                    except Exception as e:
                        logger.warning(f"Failed to load label {label_path}: {e}")
                else:
                    logger.warning(f"Label not found for image: {img_path.name}")

        return samples

    @property
    def column_names(self) -> List[str]:
        """Return column names for TRL compatibility."""
        base_cols = ["pixel_values", "input_ids", "attention_mask", "labels", "image_grid_thw"]
        if self.use_custom_losses:
            base_cols.extend(["field_labels", "rejection_labels"])
        return base_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels,
            and optionally field_labels and rejection_labels
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")

            # Manual resize to match max_pixels (more reliable than processor args)
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (self.max_pixels / (width * height)) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.LANCZOS)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Get response from label
        # 1. Check if nested under 'response' (teacher format)
        if "response" in label:
            response_data = label["response"]
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except Exception as e:
                    logger.warning(f"Failed to parse nested response string in {image_path}: {e}")
                    # Fallback to raw string if parsing fails
        else:
            # 2. Assume flat format (synthetic/stability format)
            response_data = label

        # Filter response for M0 objectives as per manifest
        # (transcription, rejection, quality flags only - no semantics)
        # This ensures no semantic reasoning is permitted in M0 stage.
        if isinstance(response_data, dict):
            m0_response = {
                "document_type": response_data.get("document_type"),
                "ocr": response_data.get("ocr"),
                "quality_flags": response_data.get("quality_flags"),
                "confidence": response_data.get("confidence"),
            }
            # Clean OCR line items if needed (ensure only raw text/price)
            if m0_response["ocr"] and "line_items" in m0_response["ocr"]:
                cleaned_lines = []
                for line in m0_response["ocr"]["line_items"]:
                    cleaned_lines.append(
                        {"raw_text": line.get("raw_text"), "raw_price": line.get("raw_price")}
                    )
                m0_response["ocr"]["line_items"] = cleaned_lines

            response = json.dumps(m0_response, indent=2)
        else:
            # Fallback for unexpected formats
            response = str(response_data)

        # Format messages using the teacher prompt template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_template},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ]

        # Apply chat template to get full text
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Get user text (for masking)
        user_messages = [messages[0]]
        user_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize full conversation
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
            max_pixels=self.max_pixels,
        )

        # Tokenize user part (for masking)
        user_inputs = self.processor(
            text=user_text,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
            max_pixels=self.max_pixels,
        )

        # Create labels (mask user part)
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        user_length = user_inputs["input_ids"].shape[1]
        labels[:user_length] = self.IGNORE_INDEX

        # Build output item
        # Ensure pixel_values is a tensor [N, C, H, W]
        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, (list, tuple)):
            if len(pixel_values) > 0:
                pixel_values = torch.cat([torch.as_tensor(p) for p in pixel_values], dim=0)
            else:
                pixel_values = torch.empty(0)
        elif not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)

        if pixel_values.ndim == 3:  # If batch dim missing
            pixel_values = pixel_values.unsqueeze(0)

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
        }

        # Add image_grid_thw if present (ensure it's a tensor [N, 3])
        if "image_grid_thw" in inputs:
            grid_thw = inputs["image_grid_thw"]
            if isinstance(grid_thw, (list, tuple)):
                if len(grid_thw) > 0:
                    grid_thw = torch.cat([torch.as_tensor(g) for g in grid_thw], dim=0)
                    if grid_thw.ndim == 1:
                        grid_thw = grid_thw.unsqueeze(0)
                else:
                    grid_thw = torch.empty(0)
            elif isinstance(grid_thw, torch.Tensor):
                if grid_thw.ndim == 1:
                    grid_thw = grid_thw.unsqueeze(0)
            else:
                grid_thw = torch.as_tensor(grid_thw)
                if grid_thw.ndim == 1:
                    grid_thw = grid_thw.unsqueeze(0)
            item["image_grid_thw"] = grid_thw

        # Cast pixel values according to data config
        if config.data.dtype == "float16":
            item["pixel_values"] = item["pixel_values"].to(torch.float16)
        elif config.data.dtype == "bfloat16":
            item["pixel_values"] = item["pixel_values"].to(torch.bfloat16)

        # Generate auxiliary labels only if custom losses are enabled
        if self.use_custom_losses:
            # Field Presence (5 key fields including line_items)
            ocr = m0_response.get("ocr") or {}
            fields = ["merchant_name", "date", "total", "tax", "line_items"]
            field_presence = []
            for f in fields:
                val = ocr.get(f)
                if f == "line_items":
                    # Check for non-empty list
                    is_present = 1.0 if isinstance(val, list) and len(val) > 0 else 0.0
                else:
                    # Check for non-null/non-empty string/number
                    is_present = 1.0 if val is not None and len(str(val)) > 0 else 0.0
                field_presence.append(is_present)

            item["field_labels"] = torch.tensor(field_presence, dtype=torch.float32)

            # Rejection Logic (Refined)
            # Base: 0.0
            # Stability:
            #   - empty/non_receipt -> 1.0
            #   - blurry -> 1.0 if readability_score < 0.5 else 0.0

            rejection_label = 0.0
            if label.get("is_stability"):
                stability_type = label.get("stability_type")
                if stability_type in ["empty", "non_receipt"]:
                    rejection_label = 1.0
                elif stability_type == "blurry" or m0_response.get("quality_flags", {}).get(
                    "is_blurry"
                ):
                    # Calculate readability score based on the 5 fields
                    score = sum(field_presence) / len(fields)
                    if score < 0.5:
                        rejection_label = 1.0

            item["rejection_labels"] = torch.tensor(rejection_label, dtype=torch.float32)

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with proper padding for M0 training."""
    # Concatenate pixel_values along the batch dimension
    pixel_values = torch.cat([torch.as_tensor(item["pixel_values"]) for item in batch], dim=0)

    max_len = max(torch.as_tensor(item["input_ids"]).shape[0] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), config.data.ignore_index, dtype=torch.long)

    for i, item in enumerate(batch):
        ids = torch.as_tensor(item["input_ids"])
        mask = torch.as_tensor(item["attention_mask"])
        lbls = torch.as_tensor(item["labels"])

        cur_len = ids.shape[0]
        input_ids[i, :cur_len] = ids
        attention_mask[i, :cur_len] = mask
        labels[i, :cur_len] = lbls

    batch_out = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if "image_grid_thw" in batch[0]:
        batch_out["image_grid_thw"] = torch.cat(
            [torch.as_tensor(item["image_grid_thw"]) for item in batch], dim=0
        )

    # Collate auxiliary labels only if present
    if "rejection_labels" in batch[0]:
        batch_out["rejection_labels"] = torch.stack(
            [torch.as_tensor(item["rejection_labels"]) for item in batch]
        )
    if "field_labels" in batch[0]:
        batch_out["field_labels"] = torch.stack(
            [torch.as_tensor(item["field_labels"]) for item in batch]
        )

    return batch_out
