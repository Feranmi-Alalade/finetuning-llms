"""Generate training labels using teacher models with OO approach and HF backend."""

import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
from torch.utils.data import DataLoader

from system.config import config
from system.logging import get_logger, disable_third_party_logs
from teachers.registry import get_teacher
from data.datasets.teacher_labels_dataset import RawImageDataset
from data.utils.jsonl import append_jsonl
from data.utils.parsing import parse_json_response, print_parsed_response
from data.utils.s3 import s3_sync_pipeline

# Disable third-party logs
disable_third_party_logs()
logger = get_logger(__name__)


import json
from datetime import datetime


def identity_collate(batch):
    """Simple collation that returns the batch as a list of items."""
    return batch


class TeacherLabelGenerator:
    """Object-oriented label generator using HF backend and torch DataLoader."""

    def __init__(
        self,
        teacher_name: str = "auto",
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        use_cache: bool = True,
    ):
        self.teacher_name = teacher_name
        self.batch_size = batch_size or config.generation.batch_size
        self.num_workers = num_workers if num_workers is not None else config.data_prep.num_workers

        # Load and resolve prompt templates
        self.prompts = self._load_and_resolve_prompts()

        # Initialize teacher
        cache_dir = config.paths.cache_dir / teacher_name if use_cache else None
        self.teacher = get_teacher(teacher_name, cache_dir=cache_dir)
        logger.info(f"Initialized teacher: {teacher_name}")
        logger.info(f"Model Info: {self.teacher.get_model_info()}")

    def _load_and_resolve_prompts(self) -> Dict[str, str]:
        """Load prompts from YAML and resolve hierarchical templates."""
        import yaml

        prompt_path_str = config.data.prompt_files.get("prompt")
        schema_path_str = config.data.prompt_files.get("schema")

        if not prompt_path_str:
            return {"single_receipt": "Analyze this receipt."}

        prompt_path = config.paths.project_root / prompt_path_str
        schema_path = config.paths.project_root / schema_path_str if schema_path_str else None

        try:
            with open(prompt_path, "r") as f:
                prompts = yaml.safe_load(f)

            # Ensure it's a dict
            if isinstance(prompts, str):
                prompts = {"single_receipt": prompts}

            # 1. Load Schema
            schema_json = ""
            if schema_path and schema_path.exists():
                with open(schema_path, "r") as f:
                    schema_json = f.read().strip()

            # 2. First pass: Inject Schema into all templates (especially single_receipt)
            for key in prompts:
                if isinstance(prompts[key], str):
                    # Handle both old and new markers for backward compatibility if needed,
                    # but favoring <SCHEMA>
                    prompts[key] = prompts[key].replace("<SCHEMA>", schema_json)

            # 3. Second pass: Inject resolved single_receipt into batch_receipt
            single_prompt = prompts.get("single_receipt", "")
            if "batch_receipt" in prompts and single_prompt:
                prompts["batch_receipt"] = prompts["batch_receipt"].replace(
                    "<SINGLE_PROMPT>", single_prompt
                )

            return prompts
        except Exception as e:
            logger.error(f"Failed to load/resolve prompts: {e}")
            return {"single_receipt": "Analyze this receipt."}

    def _load_existing_results(self, output_file: Path) -> set:
        """Load existing results and return a set of successfully processed image paths."""
        processed_paths = set()
        if not output_file.exists():
            return processed_paths

        logger.info(f"Loading existing results from {output_file}...")
        try:
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        image_path = data.get("image_path")
                        response = data.get("response")

                        # Only skip if response is valid (not None, not "null", not empty)
                        if image_path and response and str(response).strip().lower() != "null":
                            processed_paths.add(image_path)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")

        logger.info(f"Found {len(processed_paths)} already processed images. Skipping these.")
        return processed_paths

    def _pack_batches(self, dataloader, batch_prompt: str):
        """Standardize batch packing based on token counts (50% threshold)."""
        input_limit = int(0.5 * config.generation.out_seq_length)
        prompt_tokens = self.teacher.count_tokens([batch_prompt])

        current_batch = []
        current_tokens = prompt_tokens

        for item_list in dataloader:
            sample = item_list[0]
            # Estimate tokens for this image
            img_tokens = self.teacher.count_tokens([sample["image"]])

            # If this image would tip us over the limit (70%), yield current batch
            if current_batch and (current_tokens + img_tokens > input_limit):
                yield current_batch
                current_batch = []
                current_tokens = prompt_tokens

            current_batch.append(sample)
            current_tokens += img_tokens

        if current_batch:
            yield current_batch

    @s3_sync_pipeline
    def run(
        self,
        image_dir: Path | str,
        output_file: Path | str,
        max_samples: Optional[int] = None,
        print_frequency: int = 10,
    ):
        """Run the label generation process with token-aware packing."""
        image_dir = Path(image_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 3. Create metadata directory in the same gold folder
        metadata_dir = output_file.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting label generation on {image_dir}")

        # 1. Load existing results to support resume and re-attempts
        skip_paths = self._load_existing_results(output_file)

        # 2. Initialize dataset with skip_paths
        dataset = RawImageDataset(image_dir, max_samples=max_samples, skip_paths=list(skip_paths))

        if len(dataset) == 0:
            logger.info("No new images to process. Everything is already in cache.")
            return

        # Pull individual samples for dynamic batching
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=identity_collate,
        )

        batch_prompt = self.prompts["batch_receipt"]
        single_prompt = self.prompts["single_receipt"]

        results_count = 0
        last_response = ""

        # Use the packer generator to iterate through token-optimized batches
        for batch in self._pack_batches(dataloader, batch_prompt):
            images = [s["image"] for s in batch]
            logger.info(f"Processing batch of {len(images)} images...")

            try:
                responses = self.teacher.batch_generate(
                    images=images,
                    prompts=[batch_prompt, single_prompt],
                    max_tokens=config.generation.out_seq_length,
                    temperature=config.generation.temperature,
                )

                for sample, response in zip(batch, responses):
                    # Only process valid (non-null) responses
                    if not response or str(response).strip().lower() == "null":
                        logger.warning(
                            f"Skipping failed/null prediction for {sample['image_path']}"
                        )
                        continue

                    # Metadata enrichment
                    result = {
                        "image_path": sample["image_path"],
                        "filename": sample["filename"],
                        "prompt": single_prompt,
                        "response": response,
                        "teacher_model": self.teacher.model_name,
                        "model_version": self.teacher.get_model_info().get("version", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # 4. Save to master JSONL (Clean: no nulls)
                    append_jsonl(result, output_file)

                    # 5. Save to individual metadata file
                    try:
                        # Parse response for clean structure in individual file
                        try:
                            resp_json = json.loads(response)
                        except:
                            resp_json = response

                        meta_file = metadata_dir / f"{Path(sample['filename']).stem}.json"
                        meta_data = {
                            **result,
                            "response": resp_json,  # Save as object if possible
                        }
                        with open(meta_file, "w") as mf:
                            json.dump(meta_data, mf, indent=2)
                    except Exception as me:
                        logger.warning(
                            f"Failed to save individual metadata for {sample['filename']}: {me}"
                        )

                    results_count += 1
                    last_response = response

                    if results_count % print_frequency == 0:
                        logger.info(f"\n--- Sample Response (every {print_frequency} images) ---")
                        parsed = parse_json_response(response)
                        if parsed:
                            print_parsed_response(parsed)
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")

        logger.info(f"Done! Generated {results_count} labels to {output_file}")

        # Final sample check
        if results_count > 0:
            logger.info(f"Generated {results_count} labels for {self.teacher_name}")
            logger.info("\nSample raw response:")
            logger.info(last_response[:500] + "...")
            parsed = parse_json_response(last_response)
            if parsed:
                print_parsed_response(parsed)



def main():
    parser = argparse.ArgumentParser(
        description="Generate training labels using TeacherLabelGenerator"
    )
    parser.add_argument(
        "--teacher", type=str, default="auto", help="Teacher model to use (auto, gemini, qwen)"
    )
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers for data loading"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (testing)")
    parser.add_argument(
        "--print-frequency",
        type=int,
        default=5,
        help="Frequency of pretty-printing (every N images)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    # Use config defaults if not provided
    image_dir_str = args.image_dir or "artifacts/receipts/silver/base/images"
    output_str = args.output or "artifacts/receipts/platinum/teacher/teacher_labels.jsonl"

    image_dir = Path(image_dir_str)
    if not image_dir.is_absolute():
        image_dir = config.paths.project_root / image_dir

    output_file = Path(output_str)
    if not output_file.is_absolute():
        output_file = config.paths.project_root / output_file

    generator = TeacherLabelGenerator(
        teacher_name=args.teacher,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cache=not args.no_cache,
    )

    generator.run(
        image_dir=image_dir.resolve(),
        output_file=output_file.resolve(),
        max_samples=args.max_samples,
        print_frequency=args.print_frequency,
    )


if __name__ == "__main__":
    main()
