"""Master Export Script: Merge -> HF -> GGUF -> Ollama."""

import argparse
import os

# Set global config for fine-tuning BEFORE other imports
os.environ["APP_CONFIG"] = "fine_tuning.yaml"

import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer
from huggingface_hub import HfApi

# Adjust path to find system modules
sys.path.append(str(Path(__file__).parents[2]))

from system.logging import get_logger, disable_third_party_logs
from system.config import config, load_config
# Try to import Qwen3-VL/Qwen2-VL names
try:
    from transformers import Qwen3VLForConditionalGeneration as QwenModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenModel

disable_third_party_logs()
logger = get_logger(__name__)


def load_deployment_config():
    """Load deployment.yaml manually since it's not in the main config object yet."""
    config_path = Path("config/deployment.yaml")
    if not config_path.exists():
        logger.warning("deployment.yaml not found. Skipping strict validation.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_deployment(adapter_path: Path, model_name: str, deploy_config: dict):
    """Validate that the artifact matches the requested deployment model."""
    metadata_path = adapter_path / "training_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    if not deploy_config:
        return metadata

    # Get expected stage from config
    model_cfg = deploy_config.get("models", {}).get(model_name, {})
    required_stage = model_cfg.get("required_metadata", {}).get("stage")

    if required_stage and metadata.get("stage") != required_stage:
        error_msg = (
            f"Deployment Mismatch! Requested deploy for '{model_name}' (Stage: {required_stage}), "
            f"but artifact is Stage: '{metadata.get('stage')}'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Validation Passed: Artifact Stage '{metadata.get('stage')}' matches '{model_name}'")
    return metadata


def push_to_hf_hub(model_path: Path, repo_id: str, private: bool = True):
    """Push the merged model to Hugging Face Hub."""
    logger.info(f"Pushing to Hugging Face Hub: {repo_id}")
    api = HfApi()
    
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Successfully pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"HF Hub Push failed: {e}")
        # We don't raise here to allow GGUF/Ollama to continue if one target fails
        return


def convert_to_gguf(merged_path: Path, gguf_path: Path, quantization: str = "q4_k_m"):
    """Call llama.cpp conversion script for both text and vision (mmproj)."""
    logger.info(f"Converting {merged_path} to GGUF (Text + Vision) with quantization {quantization}...")
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    quantize_bin = Path("llama.cpp/build/bin/llama-quantize")

    if not convert_script.exists():
        logger.error("llama.cpp/convert_hf_to_gguf.py not found. Skipping GGUF conversion.")
        return False, None
    
    # Paths for output components
    text_gguf = gguf_path
    mmproj_gguf = gguf_path.parent / f"{gguf_path.stem}_mmproj.gguf"
    f16_path = gguf_path.with_suffix(".f16.gguf")
    
    # Native types supported by the conversion script directly
    native_types = ["f32", "f16", "bf16", "q8_0", "auto"]
    
    try:
        # STEP 1: Convert Text Model
        logger.info("Step 1/3: Converting Text Model...")
        if quantization in native_types:
            convert_cmd = [
                "uv", "run", "python", str(convert_script),
                str(merged_path),
                "--outfile", str(text_gguf),
                "--outtype", quantization
            ]
            subprocess.run(convert_cmd, check=True)
        else:
            # Two-step for k-quants
            convert_f16_cmd = [
                "uv", "run", "python", str(convert_script),
                str(merged_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ]
            subprocess.run(convert_f16_cmd, check=True)
            
            if quantize_bin.exists():
                logger.info(f"Step 2/3: Quantizing text model to {quantization}...")
                subprocess.run([str(quantize_bin), str(f16_path), str(text_gguf), quantization], check=True)
                f16_path.unlink()
            else:
                logger.warning("llama-quantize not found. Keeping f16 text model.")
                f16_path.rename(text_gguf)

        # STEP 2: Convert Vision Model (mmproj)
        logger.info("Step 3/3: Converting Vision Projector (mmproj)...")
        convert_mmproj_cmd = [
            "uv", "run", "python", str(convert_script),
            str(merged_path),
            "--outfile", str(mmproj_gguf),
            "--mmproj"
        ]
        subprocess.run(convert_mmproj_cmd, check=True)
        
        logger.info(f"GGUF conversion successful: {text_gguf} and {mmproj_gguf}")
        return True, mmproj_gguf
    except Exception as e:
        logger.error(f"Multimodal GGUF conversion failed: {e}")
        if f16_path.exists(): f16_path.unlink()
        return False, None


def deploy_to_ollama(model_name: str, gguf_path: Path, mmproj_path: Path, adapter_path: Path):
    """Create and deploy Ollama model using the teacher prompt and proper template."""
    if not gguf_path.exists():
        logger.warning(f"GGUF file not found at {gguf_path}. Skipping Ollama deployment.")
        return

    # Load Model-Specific Prompt & Schema
    prompt_map = {
        "m0": "prompts/m0_ocr.yaml",
        "m1": "prompts/m1_semantic.yaml",
        "unified": "prompts/unified.yaml"
    }
    
    # Check if model_name starts with known keys (e.g. m0-v1 -> m0)
    prompt_key = "teacher_inference"
    for key in prompt_map:
        if model_name.startswith(key):
            prompt_key = key
            break
            
    prompt_file = prompt_map.get(prompt_key, "prompts/teacher_inference.yaml")
    schema_file = f"prompts/schema/{prompt_key}.json"
    if prompt_key == "teacher_inference":
        schema_file = "prompts/schema/teacher_inference.json"

    prompt_path = Path(prompt_file)
    schema_path = Path(schema_file)
    system_prompt = "You are a specialized OCR model for receipts."
    
    if prompt_path.exists():
        try:
            import yaml
            with open(prompt_path, "r") as f:
                prompts = yaml.safe_load(f)
                system_prompt = prompts.get("single_receipt", system_prompt)
            
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    schema = f.read().strip()
                system_prompt = system_prompt.replace("<SCHEMA>", schema)
            else:
                 logger.warning(f"Schema file not found: {schema_path}")
                 
        except Exception as e:
            logger.warning(f"Failed to load prompt/schema for Modelfile: {e}")

    modelfile_path = adapter_path.parent / "Modelfile"
    
    # We use Qwen style ChatML template
    modelfile_content = f"""
FROM {gguf_path.absolute()}
ADAPTER {mmproj_path.absolute()}
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
SYSTEM \"\"\"{system_prompt}\"\"\"
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
"""
    
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    logger.info(f"Creating Ollama model: {model_name}")
    try:
        subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)
        logger.info(f"Ollama model '{model_name}' created successfully with teacher prompt instructions.")
    except Exception as e:
        logger.error(f"Ollama deployment failed: {e}")
        return


def log_deployment(model_name: str, stage: str, env: str, adapter_path: str):
    """Append entry to DEPLOYMENTS.md."""
    log_file = Path("DEPLOYMENTS.md")
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("# Deployment Log\n\n| Date | Model | Stage | Env | Status | Adapter Path |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_path = str(adapter_path).replace("|", r"\|")
    
    line = f"| {timestamp} | {model_name} | {stage} | {env} | Success | `{clean_path}` |\n"
    
    with open(log_file, "a") as f:
        f.write(line)
    
    logger.info(f"Logged deployment to {log_file}")


def generate_model_card(model_path: Path, metadata: dict, model_name: str):
    """Generate a README.md (Model Card) for the Hugging Face Hub."""
    stage = metadata.get("stage", "unknown")
    base_model = metadata.get("base_model", "unknown")
    timestamp = metadata.get("timestamp", "unknown")
    
    # Define descriptions based on model_name/stage
    descriptions = {
        "m0": "Stage M0: OCR & Stability. Optimized for high-accuracy receipt transcription and field detection (Merchant, Date, Total).",
        "m1": "Stage M1: Semantic Understanding. Optimized for line-item extraction, categorization, and complex reasoning on receipt data.",
        "unified": "Unified Model: End-to-End Receipt Intelligence. Combines OCR, stability handling, and semantic understanding in a single pass."
    }
    desc = descriptions.get(model_name, "Specialized multimodal model for receipt understanding.")

    readme_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- vision
- multimodal
- ocr
- receipt-understanding
- qwen3-vl
- bildcraft
pipeline_tag: visual-question-answering
---

# {model_name.upper()} - Receipt AI Model ({stage.upper()})

Developed by **Bildcraft**.

## Model Description
{desc}

- **Stage**: {stage}
- **Base Model**: {base_model}
- **Training Environment**: {metadata.get('env', 'unknown')}
- **Export Timestamp**: {timestamp}

## Intended Use
This model is designed for professional receipt processing pipelines. 

- **M0**: Use for raw transcription and checking if an image is a valid receipt.
- **M1**: Use for extracting structured semantic data (Item lists, tax details).
- **Unified**: Full end-to-end extraction.

## Deployment
This model is compatible with:
- `transformers`
- `vLLM`
- `Ollama` (via GGUF)

## Training Metadata
- **Mode**: {metadata.get('mode', 'lora')}
- **Custom Losses**: {metadata.get('custom_losses', True)}
"""
    
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    logger.info(f"Generated Model Card at {readme_path}")


def merge_and_save_model(base_model_id, adapter_path, output_path):
    """Merge LoRA adapter into base model and save."""
    logger.info(f"Loading base model: {base_model_id}")
    
    # Load base model
    base_model = QwenModel.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete.")


def run_export_pipeline(adapter_path: Path, model_name: str, env: str, skip_gguf: bool = False, quantization: str = "q4_k_m"):
    """Run full export pipeline based on config."""
    
    # 1. Load Configs & Validate
    deploy_config = load_deployment_config()
    metadata = validate_deployment(adapter_path, model_name, deploy_config)
    
    model_cfg = deploy_config.get("models", {}).get(model_name, {})
    org = deploy_config.get("defaults", {}).get("org", "bildcraft")
    
    output_dir = adapter_path.parent
    merged_path = output_dir / "merged_model"
    gguf_path = output_dir / f"{model_name}_{quantization}.gguf"
    
    # 2. Merge & Save
    if not merged_path.exists():
        logger.info("Merged model not found. Creating it...")
        base_model_id = metadata.get("base_model", config.models.fine_tune_base.name)
        merge_and_save_model(base_model_id, str(adapter_path), str(merged_path))
    else:
        logger.info(f"Merged model already exists at {merged_path}. Skipping merge.")

    # 3. HF Upload (Prod only)
    hf_cfg = model_cfg.get("targets", {}).get("huggingface", {})
    if env == "prod" and hf_cfg:
        base_model_name = metadata.get("base_model", "").lower()
        size = "8b" if "8b" in base_model_name else "2b"
        
        repo_template = hf_cfg.get("repo_template", "{org}/qwen3-vl-{size}-{model}-{type}")
        repo_id = repo_template.format(org=org, size=size, model=model_name, type="merged")
        
        # Generate Model Card (README.md)
        generate_model_card(merged_path, metadata, model_name)
        
        push_to_hf_hub(merged_path, repo_id, private=True)

    # 4. GGUF & Ollama
    if not skip_gguf:
        success, mmproj_path = convert_to_gguf(merged_path, gguf_path, quantization)
        if success:
            ollama_name = model_cfg.get("targets", {}).get("ollama", {}).get("model_name", f"{org}/receipt-ai-{model_name}")
            deploy_to_ollama(ollama_name, gguf_path, mmproj_path, adapter_path)

    # 5. Log Success
    stage = metadata.get("stage", "unknown")
    log_deployment(model_name, stage, env, str(adapter_path))
    
    logger.info("Export pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Export Trained Model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter directory")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (e.g. m0, m1, unified)")
    parser.add_argument("--env", type=str, default="dev", help="Target Env (dev/prod)")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="Quantization type (f16, q4_k_m, q8_0, etc.)")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF/Ollama steps")
    
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve()
    
    if not adapter_path.exists():
        logger.error(f"Adapter path not found: {adapter_path}")
        sys.exit(1)

    try:
        run_export_pipeline(adapter_path, args.model_name, args.env, skip_gguf=args.skip_gguf, quantization=args.quantization)
    except Exception as e:
        logger.error(f"Export Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
