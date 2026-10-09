#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml

from system.logging import get_logger
from data.utils.s3 import s3_sync_pipeline

# Setup logger
logger = get_logger(__name__)

# Constants
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "multimodal-receipt-ai"
ARTIFACTS_DIR = DATA_ROOT / "artifacts" / "receipts"
SILVER_DIR = ARTIFACTS_DIR / "silver"
PLATINUM_DIR = ARTIFACTS_DIR / "platinum" / "m0_ocr" / "student"
MANIFEST_PATH = DATA_ROOT / "data" / "manifests" / "m0_ocr.yaml"
TEACHER_LABELS_PATH = ARTIFACTS_DIR / "platinum" / "teacher" / "teacher_labels.jsonl"


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load the YAML manifest configuration."""
    if not path.exists():
        logger.error(f"Manifest not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_teacher_labels(path: Path) -> List[Dict[str, Any]]:
    """Load and parse teacher labels line by line."""
    if not path.exists():
        logger.error(f"Teacher labels not found at {path}")
        sys.exit(1)

    labels = []
    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Parse the nested JSON response string if it exists and is a string
                if isinstance(data.get("response"), str):
                    data["response"] = json.loads(data["response"])
                labels.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in teacher labels: {e}")

    logger.info(f"Loaded {len(labels)} teacher labels")
    return labels


def normalize_merchant(name: Optional[str]) -> str:
    """Normalize merchant names to group variations."""
    if not name:
        return "Unknown"
    return name.strip().lower().replace(" gmbh", "").replace(" & co. kg", "").title()


def get_line_item_bin(count: int) -> str:
    """Bin line item counts for stratification."""
    if count <= 3:
        return "1-3"
    elif count <= 7:
        return "4-7"
    elif count <= 12:
        return "8-12"
    else:
        return "13+"


def extract_metadata(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract relevant metadata for stratification from teacher labels."""
    metadata = []
    for label in labels:
        try:
            # The 'image_path' field contains the filename
            image_filename = label.get("image_path", "")
            if not image_filename:
                # Fallback to request if image_path missing
                image_filename = label.get("request", "")

            if not image_filename:
                continue

            # Check if image exists in silver/base
            base_image_path = SILVER_DIR / "base" / "images" / image_filename
            if not base_image_path.exists():
                logger.warning(f"Base image not found: {image_filename}")
                continue

            response = label.get("response", {})
            ocr = response.get("ocr", {})
            doc_type = response.get("document_type", "receipt")

            if doc_type != "receipt":
                # We mainly stratify receipts. Non-receipts handling might differ.
                merchant = "Non-Receipt"
                item_count = 0
            else:
                merchant = normalize_merchant(ocr.get("merchant_name"))
                items = ocr.get("line_items", [])
                item_count = len(items) if items else 0

            metadata.append(
                {
                    "filename": image_filename,
                    "merchant": merchant,
                    "item_count": item_count,
                    "bin": get_line_item_bin(item_count),
                    "doc_type": doc_type,
                    "original_label": label,
                }
            )
        except Exception as e:
            logger.warning(f"Error extracting metadata for label: {e}")
            continue

    return metadata


def create_stratified_splits(
    metadata: List[Dict[str, Any]], split_ratio: float = 0.75, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create train/test splits stratified by merchant and line item count.
    Returns lists of filenames.
    """
    random.seed(seed)

    # 1. Bucket data
    buckets = defaultdict(list)
    for item in metadata:
        # Create a unique bucket key
        key = (item["merchant"], item["bin"])
        buckets[key].append(item["filename"])

    train_files = []
    test_files = []

    # 2. Split each bucket
    for key, filenames in buckets.items():
        random.shuffle(filenames)
        n_train = int(len(filenames) * split_ratio)

        # Ensure at least one in train if we have data, unless it's a singleton which goes to train usually
        # But for very small datasets, we might just split strictly.
        # If we have 1 item: 1 * 0.75 = 0. We put it in train.
        if len(filenames) > 0 and n_train == 0:
            n_train = 1

        # If we have items but n_train took all, try to leave one for test if possible?
        # Standard approach: just strict split.

        train_bucket = filenames[:n_train]
        test_bucket = filenames[n_train:]

        train_files.extend(train_bucket)
        test_files.extend(test_bucket)

    return train_files, test_files


def save_dataset_item(
    filename: str,
    source_path: Path,
    label: Dict[str, Any],
    base_dest_dir: Path,
    split: str,
    is_stability: bool = False,
    stability_type: str = None,
):
    """Save both image and label for a dataset item."""
    # 1. Image
    img_dest_dir = base_dest_dir / "data" / split
    img_dest_dir.mkdir(parents=True, exist_ok=True)

    img_dest_path = img_dest_dir / filename
    if source_path.exists():
        shutil.copy2(source_path, img_dest_path)
    else:
        logger.warning(f"Source file missing: {source_path}")
        return

    # 2. Label
    label_dest_dir = base_dest_dir / "labels" / split
    label_dest_dir.mkdir(parents=True, exist_ok=True)

    # Modify label for this specific instance
    final_label = label.copy()
    final_label["image_path"] = filename
    final_label["filename"] = filename
    final_label["is_stability"] = is_stability

    if is_stability and stability_type:
        final_label["stability_type"] = stability_type

        # Handle specific stability types
        # For blurred/partial images, nullify fields if they are unreadable
        # Since we don't have granular mask info, we flag them.
        if stability_type == "blurry":
            if "quality_flags" not in final_label:
                final_label["quality_flags"] = {}
            final_label["quality_flags"]["is_blurry"] = True

        elif stability_type in ["empty", "non_receipt"]:
            final_label["document_type"] = (
                "empty_unreadable" if stability_type == "empty" else "non_receipt"
            )
            final_label["ocr"] = None
            final_label["semantics"] = None

    # Save label
    label_dest_path = label_dest_dir / f"{Path(filename).stem}.json"
    with open(label_dest_path, "w") as f:
        json.dump(final_label, f, indent=2)


def get_stability_variant_filename(base_filename: str, stability_type: str) -> str:
    """
    Construct the filename for a stability variant.

    Expectation:
    Base: scanned_document_17_01_26_at_18_10_15_p000.jpg
    Blurry: scanned_document_17_01_26_at_18_10_15_p000_blurry.jpg

    We need to handle the extension carefully.
    """
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix
    return f"{stem}_{stability_type}{suffix}"


def verify_stability_file(
    base_filename: str, stability_type: str, stability_dir: Path
) -> Optional[str]:
    """
    Check if a stability variant exists.
    Sometimes the naming might vary slightly, this function can be made more robust.
    """
    expected_name = get_stability_variant_filename(base_filename, stability_type)
    file_path = stability_dir / "images" / expected_name
    if file_path.exists():
        return expected_name
    return None


@s3_sync_pipeline
def main():
    parser = argparse.ArgumentParser(description="Sample M0 Dataset for Fine-tuning")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually copy files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--clean", action="store_true", help="Clean output directory before running"
    )
    args = parser.parse_args()

    # 1. Load Configurations
    manifest = load_manifest(MANIFEST_PATH)
    data_mix = manifest.get("mix", {}).get("components", {})
    splits = manifest.get("splits", {})
    train_ratio = splits.get("train", 0.75)

    logger.info(f"Loaded manifest: m0_ocr (Train Ratio: {train_ratio})")

    # 2. Load Teacher Labels & Extract Metadata
    logger.info("Loading teacher labels...")
    labels = load_teacher_labels(TEACHER_LABELS_PATH)
    metadata = extract_metadata(labels)
    logger.info(f"Extracted metadata for {len(metadata)} valid receipts")

    # 3. Stratified Split of Base Images
    logger.info("Performing stratified split...")
    train_base_files, test_base_files = create_stratified_splits(
        metadata, split_ratio=train_ratio, seed=args.seed
    )

    logger.info(f"Split results - Train: {len(train_base_files)}, Test: {len(test_base_files)}")

    # 4. Prepare Output Directories
    if args.clean and PLATINUM_DIR.exists():
        logger.info("Cleaning output directory...")
        if not args.dry_run:
            shutil.rmtree(PLATINUM_DIR)

    # Define paths
    paths = {
        "train": {
            "base": PLATINUM_DIR / "train" / "base",
            "stability": PLATINUM_DIR / "train" / "stability",
        },
        "test": {"base": PLATINUM_DIR / "test" / "base", "stability": PLATINUM_DIR / "test" / "stability"},
    }

    # 5. Process Base Images
    # 5. Process Base Images
    # Create lookup for base labels
    base_labels_map = {m["filename"]: m["original_label"] for m in metadata}

    if not args.dry_run:
        logger.info("Processing base images...")
        for filename in tqdm(train_base_files, desc="Train Base"):
            if filename in base_labels_map:
                save_dataset_item(
                    filename,
                    SILVER_DIR / "base" / "images" / filename,
                    base_labels_map[filename],
                    PLATINUM_DIR,
                    "train",
                    is_stability=False,
                )

        for filename in tqdm(test_base_files, desc="Test Base"):
            if filename in base_labels_map:
                save_dataset_item(
                    filename,
                    SILVER_DIR / "base" / "images" / filename,
                    base_labels_map[filename],
                    PLATINUM_DIR,
                    "test",
                    is_stability=False,
                )

    # 6. Process Stability Images (Data Mix)
    # We iterate through required stability components from manifest
    # Components: base, blurry, low_contrast, rotation, partial, empty, non_receipt

    stability_stats = {"train": defaultdict(int), "test": defaultdict(int)}

    for component_name, config in data_mix.items():
        if component_name == "base":
            continue

        weight = config.get("weight", 0.0)
        tag = config.get("tag", "")
        # Stability type is usually the second part of the tag (stability:blurry -> blurry)
        if ":" in tag:
            stability_type = tag.split(":")[1]
        else:
            stability_type = component_name

        logger.info(
            f"Processing stability component: {component_name} (Type: {stability_type}, Weight: {weight})"
        )

        # Source directory for this stability type
        source_dir = SILVER_DIR / "stability" / stability_type
        if not source_dir.exists():
            logger.warning(f"Stability directory not found: {source_dir}")
            continue

        # We must maintain the split!
        # For each base file in train, check if it has a stability variant.
        # If we select it for stability inclusion (based on weight?), we put it in train.
        # Wait, the weight implies the FRACTION of the final dataset that should be this type.
        #
        # But we also have a constraint: "no data leakage".
        #
        # Strategy:
        # We want X% of the final dataset to be 'blurry'.
        # We should sample from the available stability images corresponding to the base images in that split.
        #
        # Actually, simpler interpretation:
        # For 'blurry' (weight 0.05), we want roughly 5% of our tokens/images to be blurry.
        #
        # Let's take ALL available stability images that match our train/test split base images,
        # and then subsample if necessary to match the weight.
        # Or, usually in these pipelines, we generate specific amounts.
        # Since we already generated them in silver (presumably all of them?),
        # we check how many we have.
        #
        # If we have generated stability images for ALL base images,
        # then we simply select a subset to match the weight.
        #
        # Let's assume we want to fill the quota defined by weight relative to the base count.
        # Base weight is 0.75. Blurry is 0.05.
        # Ratio Blurry/Base = 0.05 / 0.75 = 1/15.
        # So for every 15 base images, we want 1 blurry image.
        #
        # To change logic:
        # 1. Start with the set of Base Images for the split (Train or Test).
        # 2. Identify available stability variants for this set.
        # 3. Calculate target count: target = (Total_Count * Weight).
        #    Total_Count = Count_Base / Weight_Base (approximate Total Dataset Size)
        #    Target_Stability = (Count_Base / Weight_Base) * Weight_Stability
        #    Target_Stability = Count_Base * (Weight_Stability / Weight_Base)

        base_weight = data_mix.get("base", {}).get("weight", 0.75)

        for split_name, base_files in [("train", train_base_files), ("test", test_base_files)]:
            # Split-specific logic
            # Train: Restrict stability to avoid overshadowing base images (User rule: 30% ratio)
            # Test: Use all available variants for thorough evaluation (User rule: "we have lots of images")
            is_train = split_name == "train"

            # Check if this is an independent type (generated synthetic)
            # We assume 'empty' and 'non_receipt' are independent as they don't map 1:1 to base images
            is_independent = stability_type in ["empty", "non_receipt"]

            # User rule parity: "the test stablity count should be as much as the train stablity count"
            # We calculate the target count based on the training base set (30% ratio)
            # and use this SAME absolute count for the test set too.

            # 1. Calculate normalization factor for stability weights
            total_stability_weight = sum(
                block.get("weight", 0) for name, block in data_mix.items() if name != "base"
            )

            if total_stability_weight <= 0:
                continue

            # 2. Calculate target count based on TRAIN base size (30% ratio)
            relative_share = weight / total_stability_weight
            n_train_base = len(train_base_files)
            total_stability_budget_for_train = n_train_base * 0.3
            target_count = max(1, int(round(total_stability_budget_for_train * relative_share)))

            if is_independent:
                # 1. List all images
                all_images = list(source_dir.glob("images/*.jpg"))
                if not all_images:
                    continue

                # 2. Filter candidate pool by current split metadata
                candidates = []
                for img_path in all_images:
                    meta_path = source_dir / "metadata" / f"{img_path.stem}.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r") as f:
                                meta = json.load(f)
                                if meta.get("split") == split_name:
                                    candidates.append(img_path.name)
                                elif split_name == "train" and meta.get("split") == "train":
                                    candidates.append(img_path.name)
                                elif split_name == "test" and meta.get("split") in ["val", "test"]:
                                    candidates.append(img_path.name)
                        except:
                            pass

                # 3. Apply target count
                selected_variants = candidates
                if len(candidates) > target_count:
                    selected_variants = random.sample(candidates, target_count)

            else:
                # Standard variants (blurry, etc.)
                # Find all valid candidates for THIS split's base images
                candidates = []
                for base_file in base_files:
                    variant_name = verify_stability_file(base_file, stability_type, source_dir)
                    if variant_name:
                        candidates.append(variant_name)

                # Apply target count
                selected_variants = candidates
                if len(candidates) > target_count:
                    selected_variants = random.sample(candidates, target_count)

            stability_stats[split_name][component_name] = len(selected_variants)

            # Copy
            if not args.dry_run:
                for variant_name in selected_variants:
                    # Find base filename to get label
                    # Variant name: base_stem_type.ext
                    # This logic assumes simple suffix appending
                    # We need to reverse engineer the base filename or rely on lookups
                    # But wait, for independent types (empty), there IS NO base label in the teacher file.

                    base_label = {}
                    if is_independent:
                        # Use a dummy label structure
                        base_label = {
                            "image_path": variant_name,
                            "filename": variant_name,
                            "document_type": "receipt",  # Will be overwritten
                            "ocr": None,
                            "semantics": None,
                        }
                    else:
                        # Try to find base label
                        # Common pattern: stem + _type + suffix
                        # We need to strip _type to find base stem
                        # However, name construction is complex.
                        # Easier way: We know the base file from the loop iteration if we use the candidates list construction logic
                        # BUT here we just have selected_variants list.
                        # HACK: Try to find matching base label by checking if any base filename is a prefix?
                        # Or rely on the candidates loop construction?

                        # Re-implement robust base finding:
                        # We know variant = {base_stem}_{type}.ext
                        # So base_stem = variant_stem without _{type}
                        variant_stem = Path(variant_name).stem
                        if variant_stem.endswith(f"_{stability_type}"):
                            base_stem = variant_stem[: -len(stability_type) - 1]  # remove _type
                            base_filename = f"{base_stem}{Path(variant_name).suffix}"
                            if base_filename in base_labels_map:
                                base_label = base_labels_map[base_filename]

                    save_dataset_item(
                        variant_name,
                        source_dir / "images" / variant_name,
                        base_label,
                        PLATINUM_DIR,
                        split_name,
                        is_stability=True,
                        stability_type=stability_type,
                    )

    # 7. Generate Summary & Manifest
    summary = {
        "dataset": "m0_ocr",
        "train_base": len(train_base_files),
        "test_base": len(test_base_files),
        "train_stability": dict(stability_stats["train"]),
        "test_stability": dict(stability_stats["test"]),
        "merchant_distribution": defaultdict(int),
    }

    # Calculate merchant distribution for train to verify balance
    for m in metadata:
        if m["filename"] in train_base_files:
            summary["merchant_distribution"][m["merchant"]] += 1

    if not args.dry_run:
        with open(PLATINUM_DIR / "sampling_manifest.json", "w") as f:
            json.dump(summary, f, indent=2)

    # 8. Print Report
    logger.info("========================================")
    logger.info("      M0 Dataset Sampling Report")
    logger.info("========================================")
    logger.info(f"{'Component':<20} | {'Train':>8} | {'Test':>8}")
    logger.info("-" * 44)

    logger.info(f"{'Base':<20} | {len(train_base_files):>8} | {len(test_base_files):>8}")
    for comp in stability_stats["train"].keys():
        train_count = stability_stats["train"][comp]
        test_count = stability_stats["test"].get(comp, 0)
        logger.info(f"{comp:<20} | {train_count:>8} | {test_count:>8}")

    logger.info("========================================")
    logger.info(f"Success! Dataset sampled to {PLATINUM_DIR}")



if __name__ == "__main__":
    main()
