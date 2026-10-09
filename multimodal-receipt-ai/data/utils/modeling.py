"""Modeling utilities for saving and merging checkpoints."""

import torch
import os
from pathlib import Path
from typing import Optional, Union
from transformers import AutoProcessor
from peft import PeftModel
from system.logging import get_logger

# Try to import Qwen3VL or fallback to Qwen2VL
try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

logger = get_logger(__name__)


def save_checkpoint(model, processor, output_dir: Union[str, Path]):
    """Save LoRA adapters and processor.

    Args:
        model: The PEFT model to save.
        processor: The processor to save.
        output_dir: Directory to save the checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter
    model.save_pretrained(output_dir)

    # Save processor
    if processor:
        processor.save_pretrained(output_dir)

    logger.info(f"Checkpoint saved to {output_dir}")


def merge_and_save_model(
    base_model_id: str,
    adapter_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cpu",
):
    """Merge LoRA adapters into base model and save.

    Args:
        base_model_id: HF ID or path of the base model.
        adapter_path: Path to the saved adapters.
        output_dir: Directory to save the merged model.
        device: Device to use for merging ('cpu' recommended for memory).
    """
    logger.info(f"Merging {adapter_path} into {base_model_id}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
    }

    if device != "cpu":
        model_kwargs["device_map"] = device

    base_model = ModelClass.from_pretrained(base_model_id, **model_kwargs)

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_dir)

    # Try to load and save processor from adapter path first, then base
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    processor.save_pretrained(output_dir)

    logger.info(f"Merged model saved to {output_dir}")
    return merged_model


class M0Inference:
    """High-level utility for M0 OCR inference."""

    def __init__(self, model_path: str, device: str = "auto", load_in_4bit: bool = False):
        """Initialize the inference module.

        Args:
            model_path: Path to the merged HF model or base model with adapters.
            device: Device to load onto ('mps', 'cuda', 'cpu', 'auto').
            load_in_4bit: Whether to load in 4-bit precision.
        """
        self.device = device
        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(f"Loading M0 Inferrer on {self.device}...")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = ModelClass.from_pretrained(model_path, **model_kwargs).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def infer(
        self, image_path: Union[str, Path, "PIL.Image.Image"], prompt_template: Optional[str] = None
    ):
        """Run OCR inference on an image.

        Args:
            image_path: Path to image or PIL Image object.
            prompt_template: Optional custom prompt template. Defaults to standard Receipt OCR.

        Returns:
            Dict containing OCR results.
        """
        from PIL import Image
        import yaml

        # Load default prompt if not provided
        if prompt_template is None:
            prompt_path = Path(__file__).parents[2] / "prompts/teacher_inference.yaml"
            if prompt_path.exists():
                with open(prompt_path, "r") as f:
                    prompts = yaml.safe_load(f)
                    prompt_template = prompts.get("single_receipt", "OCR this receipt.")
            else:
                prompt_template = "OCR this receipt."

        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        # Prepare input
        # Format for Qwen-VL: <|vision_start|>...<|vision_end|>
        # Note: Processor usually handles the tag injection if we pass images
        inputs = self.processor(
            text=[prompt_template], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # Decode
        # Slicing input_ids from generated_ids to get only new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
