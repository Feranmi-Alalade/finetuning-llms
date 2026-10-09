
import os
import subprocess
import functools
from pathlib import Path
from dotenv import load_dotenv
from system.logging import get_logger
from system.config import config, load_config

logger = get_logger(__name__)

# Load env immediately
load_dotenv()


def get_bucket_name():
    return os.getenv("S3_BUCKET")


def _run_s3_sync(source: str, dest: str, delete: bool = False, label: str = "Sync"):
    """
    Core sync helper utilizing AWS CLI.
    """
    if os.getenv("SKIP_S3_SYNC"):
        logger.info(f"[{label}] SKIP_S3_SYNC is set. Skipping.")
        return

    cmd = ["aws", "s3", "sync", source, dest]
    if delete:
        cmd.append("--delete")

    logger.info(f"[{label}] {source} -> {dest}")

    try:
        # Check aws cli existence first to give clear error?
        # subprocess will raise FileNotFoundError if aws not found
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            count = len(lines)
            if count > 0:
                logger.info(f"[{label}] Transferred {count} files.")
        else:
            logger.info(f"[{label}] No changes to sync.")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"[{label}] Failed (Code {e.returncode}): {e.stderr}")
        # We don't necessarily raise here to avoid crashing the pipeline
        # if the network is flaky, but for PROD this might be critical?
        # For now, we log error and continue, but for Sync Down in Prod we might want to fail hard.
    except FileNotFoundError:
        logger.error(f"[{label}] AWS CLI not found. Please install it.")
    except Exception as e:
        logger.error(f"[{label}] Error: {e}")


def sync_down_artifacts(bucket_name: str, artifacts_dir: Path):
    """Sync s3://{bucket} -> local/artifacts"""
    s3_uri = f"s3://{bucket_name}"
    _run_s3_sync(s3_uri, str(artifacts_dir), delete=False, label="Artifacts Sync Down")


def sync_up_artifacts(bucket_name: str, artifacts_dir: Path):
    """Sync local/artifacts -> s3://{bucket}"""
    s3_uri = f"s3://{bucket_name}"
    _run_s3_sync(str(artifacts_dir), s3_uri, delete=True, label="Artifacts Sync Up")


def s3_sync_pipeline(func):
    """
    Decorator to handle S3 synchronization for pipeline entry points.
    
    Logic:
    1. Before Execution:
       - If ENV == 'prod', sync DOWN artifacts from S3.
    2. Execution:
       - Run the pipeline function.
    3. After Execution (Success Only):
       - Sync UP artifacts to S3 (All Envs, or configurable).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = os.getenv("ENV", "dev").lower()
        bucket = get_bucket_name()
        try:
            # Re-read config paths to ensure relevance
            artifacts_dir = config.paths.artifacts_dir
        except Exception as e:
            logger.warning(f"Could not load config paths: {e}. Decorator skipping sync.")
            return func(*args, **kwargs)
        # ---------------------------------------------------------
        # 1. Pre-Run: Sync Artifacts DOWN (Prod Only)
        # ---------------------------------------------------------
        if env == "prod":
            if bucket:
                logger.info(f"=== [S3] Production Env detected. Syncing Artifacts DOWN... ===")
                sync_down_artifacts(bucket, artifacts_dir)
            else:
                logger.warning("=== [S3] Production Env but S3_BUCKET not set! ===")
        # ---------------------------------------------------------
        # 2. Run Pipeline
        # ---------------------------------------------------------
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            raise e
        # ---------------------------------------------------------
        # 3. Post-Run: Sync Artifacts UP (If bucket exists)
        # ---------------------------------------------------------
        if bucket:
            logger.info(f"=== [S3] Pipeline Success. Syncing Artifacts UP... ===")
            sync_up_artifacts(bucket, artifacts_dir)
        else:
             if env == "prod":
                 logger.error("=== [S3] S3_BUCKET not set. Cannot back up artifacts! ===")
        
        return result

    return wrapper
