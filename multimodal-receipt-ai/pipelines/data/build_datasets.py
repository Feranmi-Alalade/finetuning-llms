"""Pipeline A: Ingest & Augment (Bronze -> Silver).

This script implements the "Ingest & Augment" pipeline:
1. Normalize Bronze data (raw receipts/PDFs) -> Silver Base (Images + Metadata).
2. Generate orthogonal Stability variants for ALL base images -> Silver Stability folders.
3. Use snake_case naming and metadata for full traceability.
4. Generate synthetic labels for stability variants (Teacher not used for stability).
"""

import argparse
import json
import re
import shutil
import random
import os
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np
from tqdm import tqdm

from system.config import config
from system.logging import get_logger
from data.utils.document_conversion import preprocess_raw_image_data
from data.utils.image_processing import (
    ImageTransformEngine,
    BlurTransform,
    LightingTransform,
    PerspectiveTransform,
    VerticalPartialCropTransform,
    PartialMaskTransform,
)
from data.utils.s3 import s3_sync_pipeline

logger = get_logger(__name__)


def to_snake_case(name: str) -> str:
    """Normalize string to snake_case."""
    # Remove extension
    stem = Path(name).stem
    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-zA-Z0-9]", "_", stem)
    # Convert camelCase to snake_case
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class DataIngestionPipeline:
    def __init__(
        self,
        bronze_dir: Path,
        silver_root: Path,
        gold_root: Path,
        resample_stability: bool = False,
        reprocess: bool = False,
        enhance_ocr: bool = True,
        aggressive_ocr: bool = False,
    ):
        self.project_root = config.paths.project_root
        self.bronze_dir = bronze_dir
        self.silver_root = silver_root
        self.gold_root = gold_root
        self.resample_stability = resample_stability
        self.reprocess = reprocess
        self.enhance_ocr = enhance_ocr
        self.aggressive_ocr = aggressive_ocr

        # Stability modes
        self.stability_modes = [
            "blurry",
            "low_contrast",
            "rotation",
            "partial",
            "empty",
            "non_receipt",
        ]

        logger.info(f"Silver root: {self.silver_root}")
        logger.info(f"Gold root: {self.gold_root}")

    def _get_silver_dir(self, mode: str, stability_type: str = None) -> Path:
        if mode == "base":
            return self.silver_root / "base"
        return self.silver_root / "stability" / stability_type

    def _get_gold_dir(
        self, mode: str, stability_type: str = None, teacher_mode: str = "single"
    ) -> Path:
        if mode == "base":
            return self.gold_root / "base" / teacher_mode
        return self.gold_root / "stability" / stability_type

    def _get_split(self, id_str: str) -> str:
        """Deterministic split based on hash of ID."""
        import hashlib

        h = int(hashlib.md5(id_str.encode()).hexdigest(), 16) % 100
        if h < 70:  # 70% train
            return "train"
        elif h < 85:  # 15% val
            return "val"
        else:  # 15% test
            return "test"

    def _save_silver_artifact(
        self,
        img: Image.Image,
        image_id: str,
        receipt_id: str,
        split: str,
        dest_dir: Path,
        metadata: Dict,
    ):
        """Save image and basic metadata to silver layer."""
        img_dir = dest_dir / "images"
        meta_dir = dest_dir / "metadata"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"{image_id}.jpg"
        img.save(img_path, "JPEG", quality=95)

        basic_meta = {
            "image_id": image_id,
            "receipt_id": receipt_id,
            "split": split,
            "path": str(img_path.relative_to(self.project_root)),
            "artifact_type": "silver",
        }
        basic_meta.update(metadata)

        meta_path = meta_dir / f"{image_id}.json"
        with open(meta_path, "w") as f:
            json.dump(basic_meta, f, indent=2)

        return img_path

    def _save_gold_synthetic_label(
        self,
        image_id: str,
        receipt_id: str,
        split: str,
        silver_path: Path,
        dest_dir: Path,
        label_data: Dict,
    ):
        """Save synthetic teacher label to gold layer."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        gold_label = {
            "receipt_id": receipt_id,
            "image_id": image_id,
            "split": split,
            "source_image": str(silver_path.relative_to(self.project_root)),
            "m0_ocr": None,
            "m1_semantic": None,
            "control_signals": {
                "overall_confidence": label_data.get("confidence", 0.0),
                "quality_flags": label_data.get("quality_flags", {}),
                "document_type": label_data.get("document_type", "receipt"),
            },
        }

        label_path = dest_dir / f"{image_id}.teacher.json"
        with open(label_path, "w") as f:
            json.dump(gold_label, f, indent=2)

    def process_bronze(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 1: Normalize Bronze -> Silver Base.

        Returns:
            (all_records, new_records)
        """
        logger.info("Step 1: Normalizing Bronze to Silver Base...")
        all_records = []
        new_records = []

        files = list(self.bronze_dir.glob("*"))
        valid_exts = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}

        for f in tqdm(files, desc="Ingesting Bronze"):
            if f.suffix.lower() not in valid_exts:
                continue

            receipt_id = to_snake_case(f.name)
            split = self._get_split(receipt_id)
            silver_base_img_dir = self._get_silver_dir("base") / "images"
            existing_base = list(silver_base_img_dir.glob(f"{receipt_id}*"))

            if not self.reprocess and existing_base:
                logger.info(f"Skipping already ingested bronze file: {f.name}")
                for eb in existing_base:
                    record = {
                        "image_id": eb.stem,
                        "receipt_id": receipt_id,
                        "split": split,
                        "path": eb,
                    }
                    all_records.append(record)
                continue

            try:
                processed_id = "temp" if f.suffix.lower() == ".pdf" else receipt_id
                processed = preprocess_raw_image_data(
                    f,
                    processed_id,
                    self.silver_root,
                    enhance_for_ocr=self.enhance_ocr,
                    aggressive_enhancement=self.aggressive_ocr,
                )

                for idx, p_path in enumerate(processed):
                    img = Image.open(p_path).convert("RGB")
                
                    image_id = (
                        f"{receipt_id}_p{idx:03d}" if f.suffix.lower() == ".pdf" else receipt_id
                    )

                    saved_path = self._save_silver_artifact(
                        img,
                        image_id,
                        receipt_id,
                        split,
                        self._get_silver_dir("base"),
                        {
                            "source": f.name,
                            "mode": "base",
                            "origin": "bronze",
                            "enhanced": self.enhance_ocr,
                        },
                    )
                    record = {
                        "image_id": image_id,
                        "receipt_id": receipt_id,
                        "split": split,
                        "path": saved_path,
                    }
                    all_records.append(record)
                    new_records.append(record)

                # Cleanup PDF temp files
                if f.suffix.lower() == ".pdf":
                    temp_silver = self.silver_root / "processed" / "temp"
                    if temp_silver.exists():
                        shutil.rmtree(temp_silver)

            except Exception as e:
                logger.error(f"Failed to ingest {f.name}: {e}")

        return all_records, new_records

    def generate_augmentations(self, base_records: List[Dict[str, Any]]):
        """Step 2: Generate Orthogonal Stability Variants for ALL base images."""
        logger.info("Step 2: Generating Stability Augmentations...")

        # Engines
        engines = {
            "blurry": ImageTransformEngine([PartialMaskTransform(BlurTransform(radius=15.0))]),
            "low_contrast": ImageTransformEngine(
                [
                    PartialMaskTransform(
                        LightingTransform(brightness=1.2, contrast=0.1, is_random=False)
                    )
                ]
            ),
            "rotation": ImageTransformEngine([PerspectiveTransform(distortion_level=0.15)]),
            "partial": ImageTransformEngine([VerticalPartialCropTransform(scale=(0.5, 0.8))]),
        }

        # Control Signals for Gold Labels
        labels = {
            "blurry": {
                "confidence": 0.15,
                "quality_flags": {"is_blurry": True},
                "document_type": "receipt",
            },
            "low_contrast": {
                "confidence": 0.2,
                "quality_flags": {"is_low_contrast": True},
                "document_type": "receipt",
            },
            "rotation": {
                "confidence": 0.3,
                "quality_flags": {"is_rotated": True},
                "document_type": "receipt",
            },
            "partial": {
                "confidence": 0.25,
                "quality_flags": {"is_partial": True},
                "document_type": "receipt",
            },
        }

        for record in tqdm(base_records, desc="Augmenting"):
            receipt_id = record["receipt_id"]
            image_id_base = record["image_id"]
            split = record["split"]
            img_path = record["path"]
            img = Image.open(img_path).convert("RGB")

            for mode, engine in engines.items():
                image_id_aug = f"{image_id_base}_{mode}"

                # Conditional Resampling check
                if not self.resample_stability:
                    existing_path = (
                        self._get_silver_dir("stability", mode) / "images" / f"{image_id_aug}.jpg"
                    )
                    if existing_path.exists():
                        # logger.debug(f"Skipping existing stability variant: {image_id_aug}")
                        continue

                aug_img = engine.apply_all(img.copy())

                # Save Silver (Image)
                silver_path = self._save_silver_artifact(
                    aug_img,
                    image_id_aug,
                    receipt_id,
                    split,
                    self._get_silver_dir("stability", mode),
                    {"source_base": image_id_base, "mode": "stability", "stability_type": mode},
                )

                # Save Gold (Label)
                self._save_gold_synthetic_label(
                    image_id_aug,
                    receipt_id,
                    split,
                    silver_path,
                    self._get_gold_dir("stability", mode),
                    labels[mode],
                )

    def generate_synthetic_samples(self):
        """Step 3: Generate Generative Stability Samples (Empty, Non-Receipt)."""
        logger.info("Step 3: Generating Synthetic Stability Samples...")

        # 10% of base count, or at least 12
        base_count = len(list((self.silver_root / "base/images").glob("*.jpg")))
        count = max(12, int(base_count * 0.1))

        # Empty
        for i in range(count):
            receipt_id = f"gen_receipt_empty_{i:03d}"
            image_id = receipt_id

            # Conditional Resampling check
            if not self.resample_stability:
                existing_path = (
                    self._get_silver_dir("stability", "empty") / "images" / f"{image_id}.jpg"
                )
                if existing_path.exists():
                    continue

            split = self._get_split(receipt_id)

            color = tuple(np.random.randint(220, 256, 3))
            img = Image.new("RGB", (800, 1000), color=color)

            silver_path = self._save_silver_artifact(
                img,
                image_id,
                receipt_id,
                split,
                self._get_silver_dir("stability", "empty"),
                {"mode": "stability", "stability_type": "empty"},
            )

            self._save_gold_synthetic_label(
                image_id,
                receipt_id,
                split,
                silver_path,
                self._get_gold_dir("stability", "empty"),
                {"confidence": 0.0, "document_type": "empty_unreadable"},
            )

        # Non-receipt
        from PIL import ImageDraw

        for i in range(count):
            receipt_id = f"gen_receipt_nr_{i:03d}"
            image_id = receipt_id

            # Conditional Resampling check
            if not self.resample_stability:
                existing_path = (
                    self._get_silver_dir("stability", "non_receipt") / "images" / f"{image_id}.jpg"
                )
                if existing_path.exists():
                    continue

            split = self._get_split(receipt_id)

            img = Image.new("RGB", (800, 1000), color="white")
            draw = ImageDraw.Draw(img)

            if i % 3 == 0:
                category = "text_document"
                draw.text((50, 50), "Standard Document Page\n" + "Section " + str(i), fill="black")
                doc_type = "non_receipt"
            elif i % 3 == 1:
                category = "random_pattern"
                for _ in range(50):
                    x1, y1 = np.random.randint(0, 800), np.random.randint(0, 1000)
                    x2, y2 = x1 + np.random.randint(10, 100), y1 + np.random.randint(10, 100)
                    draw.rectangle([x1, y1, x2, y2], fill=tuple(np.random.randint(0, 256, 3)))
                doc_type = "non_receipt"
            else:
                noise = np.random.randint(0, 256, (1000, 800, 3), dtype=np.uint8)
                img = Image.fromarray(noise)
                category = "noise"
                doc_type = "non_receipt"

            silver_path = self._save_silver_artifact(
                img,
                image_id,
                receipt_id,
                split,
                self._get_silver_dir("stability", "non_receipt"),
                {"mode": "stability", "stability_type": "non_receipt", "details": category},
            )

            self._save_gold_synthetic_label(
                image_id,
                receipt_id,
                split,
                silver_path,
                self._get_gold_dir("stability", "non_receipt"),
                {"confidence": 0.0, "document_type": doc_type},
            )

    @s3_sync_pipeline
    def run(self):
        # 0. Clean and prepare base artifacts only if reprocess is enabled
        if self.reprocess and self.silver_root.exists():
            logger.info("Reprocess enabled. Clearing silver base artifacts...")
            shutil.rmtree(self.silver_root)

        if self.resample_stability:
            if (self.silver_root / "stability").exists():
                logger.info("Resampling enabled. Clearing silver stability images...")
                shutil.rmtree(self.silver_root / "stability")
            if (self.gold_root / "stability").exists():
                logger.info("Resampling enabled. Clearing gold stability labels...")
                shutil.rmtree(self.gold_root / "stability")
        else:
            logger.info("Resampling disabled. Preserving existing stability artifacts.")

        # 1. Ingest
        all_records, new_records = self.process_bronze()
        if not all_records:
            logger.warning("No base images processed. Check bronze folder.")
            return

        # 2. Augment
        augment_targets = all_records if self.resample_stability else new_records
        if augment_targets:
            logger.info(f"Generating augmentations for {len(augment_targets)} images...")
            self.generate_augmentations(augment_targets)
        else:
            logger.info("No new images to augment.")

        # 3. Synthetic
        if self.resample_stability or new_records:
            self.generate_synthetic_samples()
        else:
            logger.info("Skipping synthetic sample generation (already exists).")

        logger.info("Pipeline A finished successfully.")

        # 4. Final local cleanup of temporary processing files
        processed_root = self.silver_root / "processed"
        if processed_root.exists():
            logger.info("Cleaning up temporary processing files...")
            shutil.rmtree(processed_root)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline A: Ingest & Augment (Bronze -> Silver)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.paths.sources_dir / "receipts_raw"),
        help="Input bronze directory",
    )
    parser.add_argument(
        "--silver-root",
        type=str,
        default="artifacts/receipts/silver",
        help="Silver output root",
    )
    parser.add_argument(
        "--gold-root",
        type=str,
        default="artifacts/receipts/gold",
        help="Gold output root",
    )
    parser.add_argument(
        "--resample-stability",
        action="store_true",
        default=False,
        help="Whether to regenerate stability images if they already exist",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="Whether to reprocess already ingested bronze files",
    )
    parser.add_argument(
        "--enhance-ocr",
        action="store_true",
        default=True,
        help="Apply OCR-optimized image enhancement",
    )
    parser.add_argument(
        "--aggressive-ocr",
        action="store_true",
        default=False,
        help="Apply aggressive enhancement for faded receipts",
    )
    args = parser.parse_args()
    project_root = config.paths.project_root

    bronze_dir = Path(args.input)
    if not bronze_dir.is_absolute():
        bronze_dir = project_root / bronze_dir

    silver_root = Path(args.silver_root)
    if not silver_root.is_absolute():
        silver_root = project_root / silver_root

    gold_root = Path(args.gold_root)
    if not gold_root.is_absolute():
        gold_root = project_root / gold_root

    pipeline = DataIngestionPipeline(
        bronze_dir=bronze_dir,
        silver_root=silver_root,
        gold_root=gold_root,
        resample_stability=args.resample_stability,
        reprocess=args.reprocess,
        enhance_ocr=args.enhance_ocr,
        aggressive_ocr=args.aggressive_ocr,
    )
    pipeline.run()
