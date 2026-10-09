"""Evaluation pipeline for VLM fine-tuned models."""

import argparse
import json
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from system.config import config
from system.logging import setup_logger
from models.assemblies.ocr_only import M0_OCR_Model
from data.receipt_dataset import ReceiptDataset, collate_fn
from evaluation.metrics.ocr_metrics import OCRMetrics
from data.utils.validation import SchemaValidator

logger = setup_logger(__name__)


def load_eval_contract(lineage: str) -> dict:
    """Load evaluation contract for lineage."""
    lineage_dir = config.get_lineage_dir(lineage)
    with open(lineage_dir / "eval_contract.yaml") as f:
        return yaml.safe_load(f)


def load_model(lineage: str, checkpoint_path: str = None) -> M0_OCR_Model:
    """Load trained model from checkpoint."""
    if checkpoint_path is None:
        # Load last checkpoint
        checkpoint_dir = config.get_lineage_dir(lineage) / "checkpoints"
        checkpoint_path = checkpoint_dir / "last.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    if lineage == "m0_ocr":
        model = M0_OCR_Model.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown lineage: {lineage}")

    model.eval()
    return model


def evaluate_ocr(
    model: M0_OCR_Model,
    dataloader: DataLoader,
    device: str = "cuda",
    max_samples: int = None,
) -> dict:
    """Evaluate OCR model on test set."""
    metrics = OCRMetrics()
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_references = []

    num_processed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            batch_size = pixel_values.size(0)

            # Generate predictions
            # Create simple prompts for evaluation
            prompts = ["Transcribe all text from this receipt exactly as shown."] * batch_size

            predictions = model.generate_from_batch(
                pixel_values=pixel_values,
                prompts=prompts,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=False,
            )

            # Decode references (ground truth)
            references = []
            for label_ids in labels:
                # Remove ignore tokens and padding
                valid_ids = label_ids[label_ids != -100]
                if len(valid_ids) > 0:
                    ref_text = model.processor.tokenizer.decode(
                        valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    references.append(ref_text)
                else:
                    references.append("")

            # Add to metrics
            metrics.add_batch(predictions, references)
            all_predictions.extend(predictions)
            all_references.extend(references)

            num_processed += batch_size
            if max_samples and num_processed >= max_samples:
                break

    # Compute metrics
    results = metrics.compute()

    # Add sample outputs for inspection
    results["sample_outputs"] = [
        {
            "prediction": pred,
            "reference": ref,
        }
        for pred, ref in zip(all_predictions[:5], all_references[:5])
    ]

    return results


def check_contract(metrics: dict, contract: dict, lineage: str) -> bool:
    """Check if metrics meet evaluation contract."""
    passed = True

    thresholds = contract.get("thresholds", {})

    logger.info("\n" + "=" * 80)
    logger.info(f"Evaluation Contract Check: {lineage}")
    logger.info("=" * 80)

    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            logger.warning(f"⚠️  Metric '{metric_name}' not found in results")
            continue

        value = metrics[metric_name]

        # Check max threshold
        if "max" in threshold:
            max_val = threshold["max"]
            if value <= max_val:
                logger.info(f"✓ {metric_name}: {value:.4f} ≤ {max_val:.4f} (PASS)")
            else:
                logger.error(f"✗ {metric_name}: {value:.4f} > {max_val:.4f} (FAIL)")
                passed = False

        # Check min threshold
        if "min" in threshold:
            min_val = threshold["min"]
            if value >= min_val:
                logger.info(f"✓ {metric_name}: {value:.4f} ≥ {min_val:.4f} (PASS)")
            else:
                logger.error(f"✗ {metric_name}: {value:.4f} < {min_val:.4f} (FAIL)")
                passed = False

    logger.info("=" * 80)

    if passed:
        logger.info(f"✓ {lineage} PASSED all evaluation contracts")
    else:
        logger.error(f"✗ {lineage} FAILED evaluation contracts")

    logger.info("=" * 80 + "\n")

    return passed


def evaluate(
    lineage: str,
    checkpoint_path: str = None,
    output_dir: str = None,
    max_samples: int = None,
):
    """Run evaluation for a lineage."""
    logger.info(f"Evaluating lineage: {lineage}")

    # Load model
    model = load_model(lineage, checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frozen test set
    frozen_set_path = config.project_root / "evaluation" / "frozen_sets" / f"{lineage}.jsonl"

    if not frozen_set_path.exists():
        logger.warning(f"Frozen test set not found: {frozen_set_path}")
        logger.warning("Using validation set instead")
        frozen_set_path = config.artifacts_dir / "model_scoped" / lineage / "val.jsonl"

    image_root = config.sources_dir / "receipts_raw"

    # Create dataset
    test_dataset = ReceiptDataset(
        jsonl_path=frozen_set_path,
        image_root=image_root,
        processor=model.processor,
        max_length=2048,
        task_type="ocr",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.default_num_workers,
        collate_fn=collate_fn,
    )

    # Evaluate
    logger.info(f"Running evaluation on {len(test_dataset)} samples...")
    metrics = evaluate_ocr(model, test_loader, device, max_samples)

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 40)
    for metric_name, value in metrics.items():
        if metric_name != "sample_outputs":
            logger.info(f"  {metric_name}: {value:.4f}")

    # Print sample outputs
    if "sample_outputs" in metrics:
        logger.info("\nSample Outputs:")
        logger.info("-" * 40)
        for i, sample in enumerate(metrics["sample_outputs"][:3]):
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"Reference: {sample['reference'][:200]}...")
            logger.info(f"Prediction: {sample['prediction'][:200]}...")

    # Check contract
    contract = load_eval_contract(lineage)
    passed = check_contract(metrics, contract, lineage)

    # Save results
    if output_dir:
        output_path = Path(output_dir) / f"{lineage}_eval_results.json"
    else:
        output_path = config.get_lineage_dir(lineage) / "eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove sample outputs from saved results (can be large)
    save_metrics = {k: v for k, v in metrics.items() if k != "sample_outputs"}

    with open(output_path, "w") as f:
        json.dump(
            {
                "lineage": lineage,
                "metrics": save_metrics,
                "contract_passed": passed,
                "checkpoint": str(checkpoint_path),
            },
            f,
            indent=2,
        )

    logger.info(f"\nResults saved to {output_path}")

    return metrics, passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model lineage")
    parser.add_argument("--lineage", type=str, required=True, help="Model lineage to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to evaluate")
    parser.add_argument("--output-dir", type=str, help="Directory to save results")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate (for testing)")

    args = parser.parse_args()

    metrics, passed = evaluate(
        lineage=args.lineage,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    # Exit with error code if contract failed
    if not passed:
        exit(1)


if __name__ == "__main__":
    main()
