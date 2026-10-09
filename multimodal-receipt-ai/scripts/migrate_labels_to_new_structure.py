"""Migration script to move existing teacher labels to the new directory structure.

Moves:
- gold/teacher_labels.jsonl -> gold/teacher/teacher_labels.jsonl
- gold/metadata/ -> gold/teacher/metadata/
"""

import shutil
from pathlib import Path
from system.config import config
from system.logging import get_logger

logger = get_logger(__name__)


def migrate():
    project_root = Path(__file__).parent.parent.parent
    gold_dir = project_root / "multimodal-receipt-ai/artifacts/receipts/gold"
    new_teacher_dir = gold_dir / "teacher"

    # 1. Create new teacher directory
    new_teacher_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring directory exists: {new_teacher_dir}")

    # 2. Move teacher_labels.jsonl
    old_jsonl = gold_dir / "teacher_labels.jsonl"
    new_jsonl = new_teacher_dir / "teacher_labels.jsonl"

    if old_jsonl.exists():
        logger.info(f"Moving {old_jsonl} to {new_jsonl}")
        shutil.move(str(old_jsonl), str(new_jsonl))
    elif not new_jsonl.exists():
        logger.warning(f"Old JSONL not found at {old_jsonl}")

    # 3. Move metadata directory
    old_metadata = gold_dir / "metadata"
    new_metadata = new_teacher_dir / "metadata"

    if old_metadata.exists():
        if new_metadata.exists():
            logger.info(f"Merging {old_metadata} into {new_metadata}")
            for item in old_metadata.iterdir():
                dest = new_metadata / item.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(item), str(dest))
            old_metadata.rmdir()
        else:
            logger.info(f"Moving {old_metadata} to {new_metadata}")
            shutil.move(str(old_metadata), str(new_metadata))

    # 4. Run Sampler to create new student structure
    import subprocess

    sampler_script = project_root / "multimodal-receipt-ai/pipelines/data/sample_m0_dataset.py"
    logger.info(f"Running sampler script: {sampler_script}")
    try:
        subprocess.run(
            ["uv", "run", "python", str(sampler_script), "--clean"],
            check=True,
            cwd=project_root / "multimodal-receipt-ai",
        )
        logger.info("Sampler run successful.")
    except Exception as e:
        logger.error(f"Sampler run failed: {e}")
        return

    # 5. Cleanup legacy folders
    for legacy in ["training", "stability"]:
        legacy_path = gold_dir / legacy
        if legacy_path.exists():
            logger.info(f"Removing legacy folder: {legacy_path}")
            shutil.rmtree(legacy_path)

    logger.info("Full restructuring complete!")


if __name__ == "__main__":
    migrate()
