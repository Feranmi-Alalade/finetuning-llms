"""
Frozen Test Evaluation Script
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

from system.logger import get_logger
from evaluation.metrics.ocr_metrics import compute_ocr_accuracy

logger = get_logger(__name__)


def run_eval(model_path: str, test_set_dir: str):
    logger.info(f"Running frozen evaluation on {test_set_dir} using model {model_path}")

    test_path = Path(test_set_dir)
    labels_file = test_path / "labels.jsonl"

    logger.info("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    predictions = []
    references = []

    with open(labels_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            image_path = test_path / entry["image"]
            gt = entry["ground_truth"]["raw_text"]

            # Predict (simplified logic)
            image = Image.open(image_path).convert("RGB")
            # ... model.generate logic ...
            pred_text = "placeholder prediction"

            predictions.append(pred_text)
            references.append(gt)

    metrics = compute_ocr_accuracy(predictions, references)
    logger.info(f"Evaluation Metrics: {metrics}")

    # Save results
    with open("evaluation/reports/latest_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_set", type=str, default="data/test_sets/v1_ocr_frozen/")

    args = parser.parse_args()
    run_eval(args.model, args.test_set)
