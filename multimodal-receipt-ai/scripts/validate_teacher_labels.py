"""
Validation and cleanup script for teacher labels.

This script:
1. Validates all metadata files in gold/metadata/
2. Validates all entries in teacher_labels.jsonl
3. Moves invalid metadata files to marked_for_deletion/
4. Regenerates a clean teacher_labels.jsonl without invalid entries
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from system.config import config
from system.logging import get_logger
from data.utils.validation import SchemaValidator
from data.utils.s3 import sync_artifacts_to_s3
from prompts.schema.teacher_inference_model import TeacherInferenceResponse

logger = get_logger(__name__)


class TeacherLabelsValidator:
    """Validator for teacher labels and metadata."""

    def __init__(self, gold_root: Path):
        self.gold_root = gold_root
        self.metadata_dir = gold_root / "metadata"
        self.jsonl_path = gold_root / "teacher_labels.jsonl"
        self.marked_for_deletion_dir = gold_root / "marked_for_deletion"
        self.validator = SchemaValidator(TeacherInferenceResponse)

        # Statistics
        self.stats = {
            "metadata_total": 0,
            "metadata_valid": 0,
            "metadata_invalid": 0,
            "jsonl_total": 0,
            "jsonl_valid": 0,
            "jsonl_invalid": 0,
        }

    def validate_metadata_files(self) -> List[Path]:
        """
        Validate all metadata JSON files.

        Returns:
            List of invalid file paths
        """
        logger.info(f"Validating metadata files in {self.metadata_dir}")

        if not self.metadata_dir.exists():
            logger.warning(f"Metadata directory does not exist: {self.metadata_dir}")
            return []

        invalid_files = []
        metadata_files = list(self.metadata_dir.glob("*.json"))
        self.stats["metadata_total"] = len(metadata_files)

        for meta_file in tqdm(metadata_files, desc="Validating metadata"):
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)

                # Extract the response field
                response = data.get("response")
                if not response:
                    logger.warning(f"No response field in {meta_file.name}")
                    invalid_files.append(meta_file)
                    self.stats["metadata_invalid"] += 1
                    continue

                # Validate the response
                is_valid, _, error_dict = self.validator.validate(response, context=meta_file.name)

                if not is_valid:
                    logger.warning(
                        f"Invalid response in {meta_file.name}: {json.dumps(error_dict)}"
                    )
                    invalid_files.append(meta_file)
                    self.stats["metadata_invalid"] += 1
                else:
                    self.stats["metadata_valid"] += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {meta_file.name}: {e}")
                invalid_files.append(meta_file)
                self.stats["metadata_invalid"] += 1
            except Exception as e:
                logger.error(f"Error processing {meta_file.name}: {e}")
                invalid_files.append(meta_file)
                self.stats["metadata_invalid"] += 1

        return invalid_files

    def validate_jsonl(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate teacher_labels.jsonl entries.

        Returns:
            Tuple of (valid_entries, invalid_entries)
        """
        logger.info(f"Validating JSONL file: {self.jsonl_path}")

        if not self.jsonl_path.exists():
            logger.warning(f"JSONL file does not exist: {self.jsonl_path}")
            return [], []

        valid_entries = []
        invalid_entries = []

        with open(self.jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                self.stats["jsonl_total"] += 1
                try:
                    entry = json.loads(line.strip())

                    # Extract response field
                    response = entry.get("response")
                    if not response:
                        logger.warning(f"Line {line_num}: No response field")
                        invalid_entries.append(entry)
                        self.stats["jsonl_invalid"] += 1
                        continue

                    # Validate response
                    is_valid, _, error_dict = self.validator.validate(
                        response, context=f"jsonl_{self.jsonl_path.name}_line_{line_num}"
                    )

                    if not is_valid:
                        logger.warning(
                            f"Line {line_num}: Invalid response - {json.dumps(error_dict)}"
                        )
                        invalid_entries.append(entry)
                        self.stats["jsonl_invalid"] += 1
                    else:
                        valid_entries.append(entry)
                        self.stats["jsonl_valid"] += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode error - {e}")
                    self.stats["jsonl_invalid"] += 1
                except Exception as e:
                    logger.error(f"Line {line_num}: Error - {e}")
                    self.stats["jsonl_invalid"] += 1

        return valid_entries, invalid_entries

    def mark_for_deletion(self, invalid_files: List[Path]):
        """
        Move invalid metadata files to marked_for_deletion directory.

        Args:
            invalid_files: List of invalid file paths
        """
        if not invalid_files:
            logger.info("No invalid files to mark for deletion")
            return

        self.marked_for_deletion_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Moving {len(invalid_files)} invalid files to {self.marked_for_deletion_dir}")

        for file_path in tqdm(invalid_files, desc="Marking for deletion"):
            dest_path = self.marked_for_deletion_dir / file_path.name
            try:
                shutil.move(str(file_path), str(dest_path))
                logger.debug(f"Moved {file_path.name} to marked_for_deletion")
            except Exception as e:
                logger.error(f"Failed to move {file_path.name}: {e}")

    def regenerate_jsonl(self, valid_entries: List[Dict]):
        """
        Regenerate teacher_labels.jsonl with only valid entries.

        Args:
            valid_entries: List of valid JSONL entries
        """
        if not valid_entries:
            logger.warning("No valid entries to write to JSONL")
            return

        # Backup original
        backup_path = self.jsonl_path.with_suffix(".jsonl.backup")
        if self.jsonl_path.exists():
            shutil.copy(str(self.jsonl_path), str(backup_path))
            logger.info(f"Backed up original JSONL to {backup_path}")

        # Write clean JSONL
        logger.info(f"Writing {len(valid_entries)} valid entries to {self.jsonl_path}")
        with open(self.jsonl_path, "w") as f:
            for entry in valid_entries:
                f.write(json.dumps(entry) + "\n")

    def print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Metadata Files:")
        logger.info(f"  Total:   {self.stats['metadata_total']}")
        logger.info(f"  Valid:   {self.stats['metadata_valid']}")
        logger.info(f"  Invalid: {self.stats['metadata_invalid']}")
        logger.info(f"\nJSONL Entries:")
        logger.info(f"  Total:   {self.stats['jsonl_total']}")
        logger.info(f"  Valid:   {self.stats['jsonl_valid']}")
        logger.info(f"  Invalid: {self.stats['jsonl_invalid']}")
        logger.info("=" * 60)

    def run(self):
        """Run the complete validation and cleanup process."""
        logger.info("Starting teacher labels validation and cleanup...")

        # 1. Validate metadata files
        invalid_metadata = self.validate_metadata_files()

        # 2. Validate JSONL
        valid_jsonl, invalid_jsonl = self.validate_jsonl()

        # 3. Mark invalid metadata for deletion
        self.mark_for_deletion(invalid_metadata)

        # 4. Regenerate clean JSONL
        self.regenerate_jsonl(valid_jsonl)

        # 5. Print summary
        self.print_summary()

        logger.info("\nValidation and cleanup complete!")
        logger.info(f"Invalid metadata files moved to: {self.marked_for_deletion_dir}")
        logger.info(f"Clean JSONL written to: {self.jsonl_path}")
        logger.info(f"Original JSONL backed up to: {self.jsonl_path.with_suffix('.jsonl.backup')}")

        # 6. Sync to S3 to mirror deletions and updates
        sync_artifacts_to_s3(str(self.gold_root.parent), delete=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and cleanup teacher labels")
    parser.add_argument(
        "--gold-root",
        type=str,
        default="artifacts/receipts/gold/teacher",
        help="Path to gold/teacher directory containing metadata and teacher_labels.jsonl",
    )

    args = parser.parse_args()

    # Resolve path
    project_root = config.paths.project_root
    gold_root = Path(args.gold_root)
    if not gold_root.is_absolute():
        gold_root = project_root / gold_root

    # Run validation
    validator = TeacherLabelsValidator(gold_root)
    validator.run()


if __name__ == "__main__":
    main()
