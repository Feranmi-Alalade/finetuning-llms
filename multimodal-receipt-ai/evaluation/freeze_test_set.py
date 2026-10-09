"""
Frozen Test Set Ground Truth Generation and Freezing
"""

import json
import argparse
import shutil
from pathlib import Path
from system.logger import get_logger
from training.teacher.generate_ocr_labels import TeacherOCR

logger = get_logger(__name__)


def freeze_test_set(image_dir: str, output_dir: str):
    logger.info(f"Freezing test set from {image_dir}")

    image_path = Path(image_dir)
    dest_path = Path(output_dir)
    dest_images = dest_path / "images"
    dest_images.mkdir(parents=True, exist_ok=True)

    teacher = TeacherOCR()
    images = (
        list(image_path.glob("*.jpg"))
        + list(image_path.glob("*.png"))
        + list(image_path.glob("*.jpeg"))
    )

    ground_truth = []

    for img in images:
        logger.info(f"Processing {img.name}")
        # Copy image
        shutil.copy(img, dest_images / img.name)

        # Generate label
        result = teacher.generate(str(img))

        entry = {"image": f"images/{img.name}", "ground_truth": result}
        ground_truth.append(entry)

    gt_file = dest_path / "labels.jsonl"
    with open(gt_file, "w", encoding="utf-8") as f:
        for entry in ground_truth:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Test set frozen at {dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/test_sets/v1_ocr_frozen/")

    args = parser.parse_args()
    freeze_test_set(args.image_dir, args.output_dir)
