"""Receipt datasets for training and label generation."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, List, Any
from PIL import Image
import json
from datasets import load_dataset, Dataset as HFDataset

from system.config import config
from system.logging import get_logger
from data.utils.image_processing import StandardImageProcessor

logger = get_logger(__name__)


class RawImageDataset(Dataset):
    """Dataset for discovering and loading raw images using HF datasets backend.

    This class is used for inference/label generation by teacher models.
    """

    def __init__(
        self,
        image_dir: Path | str,
        max_samples: Optional[int] = None,
        skip_paths: Optional[List[str]] = None,
    ):
        self.image_dir = Path(image_dir)

        logger.info(f"Loading raw images from {self.image_dir} using HF imagefolder...")
        try:
            # Note: imagefolder automatically discovers images in subdirectories
            self.dataset = load_dataset("imagefolder", data_dir=str(self.image_dir), split="train")

            # Filter by skip_paths if provided
            if skip_paths:
                skip_set = set(skip_paths)

                # We need to find the relative path for each sample in the HF dataset
                # imagefolder usually puts the relative path in 'image' object's filename or a metadata field
                # but let's be safe and filter based on what we can extract
                def is_not_skipped(sample):
                    img = sample["image"]
                    if hasattr(img, "filename") and img.filename:
                        full_path = Path(img.filename)
                        try:
                            rel_path = str(full_path.relative_to(config.paths.sources_dir))
                        except ValueError:
                            rel_path = full_path.name
                        return rel_path not in skip_set
                    return True

                self.dataset = self.dataset.filter(is_not_skipped)

            if max_samples:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

            # Initialize image processor with max 1024x1024 / 1MP constraint
            self.processor = StandardImageProcessor()
        except Exception as e:
            logger.error(f"Failed to load images with HF datasets: {e}")
            raise

        logger.info(f"Initialized RawImageDataset with {len(self.dataset)} images")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        image = sample["image"]

        # Get relative path for metadata
        # HF imagefolder stores the full path in the PIL image object's filename attribute
        if hasattr(image, "filename") and image.filename:
            full_path = Path(image.filename)
            try:
                # Try to make it relative to the project root or source dir for consistency
                # We'll use sources_dir as the base if possible
                rel_path = full_path.relative_to(config.paths.sources_dir)
            except ValueError:
                rel_path = full_path.name
        else:
            rel_path = f"unknown_{idx}.jpg"

        # Apply standard resizing logic (max 1024 side / 1MP)
        image = self.processor.process(image)

        return {
            "image": image,
            "image_path": str(rel_path),
            "filename": Path(rel_path).name,
        }


class ReceiptDataset(Dataset):
    """Dataset for receipt tasks using Hugging Face datasets backend for training."""

    def __init__(
        self,
        jsonl_path: Path | str,
        image_root: Path | str,
        processor,
        max_length: Optional[int] = None,
        task_type: str = "ocr",
    ):
        self.processor = processor
        self.max_length = max_length or config.training.max_seq_length
        self.task_type = task_type
        self.image_root = Path(image_root)
        self.IGNORE_INDEX = config.data.ignore_index

        logger.info(f"Loading dataset from {jsonl_path} using HF datasets...")
        self.hf_dataset = load_dataset("json", data_files=str(jsonl_path), split="train")

        # Task-specific prompts from config
        self.task_prompts = config.data.task_prompts

        logger.info(f"Initialized ReceiptDataset with {len(self.hf_dataset)} samples")

    def _format_conversation(self, conversations: List[Dict]) -> tuple[str, str]:
        """Format conversation into prompt and response."""
        prompt = ""
        response = ""
        for turn in conversations:
            if turn["from"] in ["human", "user"]:
                prompt = turn["value"]
            elif turn["from"] in ["gpt", "assistant"]:
                response = turn["value"]
        return prompt, response

    def _get_prompt_response(self, sample: Dict) -> tuple[str, str]:
        """Extract prompt and response from sample."""
        if "conversations" in sample:
            return self._format_conversation(sample["conversations"])

        if "prompt" in sample and "response" in sample:
            return sample["prompt"], sample["response"]

        prompt = self.task_prompts.get(self.task_type, "Describe this receipt.")

        if self.task_type == "ocr":
            response = sample.get("ocr_text", "")
        elif self.task_type == "structured":
            res = sample.get("response", sample.get("structured_output", ""))
            response = json.dumps(res, indent=2) if isinstance(res, dict) else res
        else:
            response = sample.get("text", "")

        return prompt, response

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.hf_dataset[idx]

        image_path = self.image_root / sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        prompt, response = self._get_prompt_response(sample)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        user_messages = [messages[0]]
        user_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        user_inputs = self.processor(
            text=user_text,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        user_length = user_inputs["input_ids"].shape[1]
        labels[:user_length] = self.IGNORE_INDEX

        item = {
            "pixel_values": inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
        }

        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]

        # Cast pixel values according to data config
        if config.data.dtype == "float16":
            item["pixel_values"] = item["pixel_values"].to(torch.float16)
        elif config.data.dtype == "bfloat16":
            item["pixel_values"] = item["pixel_values"].to(torch.bfloat16)

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with proper padding."""
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)

    max_len = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), config.data.ignore_index, dtype=torch.long)

    for i, item in enumerate(batch):
        cur_len = item["input_ids"].shape[0]
        input_ids[i, :cur_len] = item["input_ids"]
        attention_mask[i, :cur_len] = item["attention_mask"]
        labels[i, :cur_len] = item["labels"]

    batch_out = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if "image_grid_thw" in batch[0]:
        batch_out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    return batch_out
