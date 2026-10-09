import subprocess
import yaml
import json
from pathlib import Path

def run_inference():
    # Paths
    project_root = Path("/Users/david/codes-and-scripts/projects/bildcraft/research/finetuning-llms/multimodal-receipt-ai")
    prompt_file = project_root / "prompts/m0_ocr.yaml"
    schema_file = project_root / "prompts/schema/m0_ocr.json"
    model_dir = project_root / "outputs/fine_tuned_qwen_8b_dev/m0_ocr_lora_20260129_233429_dev_no_custom_loss"
    model_path = model_dir / "m0_q4_k_m.gguf"
    mmproj_path = model_dir / "m0_q4_k_m_mmproj.gguf"
    image_path = project_root / "artifacts/receipts/silver/base/images/scanned_document_17_01_26_at_18_10_15_p000.jpg"
    cli_path = project_root / "llama.cpp/build/bin/llama-mtmd-cli"

    # Load prompt and schema
    with open(prompt_file, "r") as f:
        prompts = yaml.safe_load(f)
        prompt = prompts["single_receipt"]
    
    with open(schema_file, "r") as f:
        schema = f.read().strip()
    
    # Inject schema
    full_prompt = prompt.replace("<SCHEMA>", schema)

    # Run command
    cmd = [
        str(cli_path),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--image", str(image_path),
        "-p", full_prompt,
        "-n", "512",
        "--temp", "0.1"
    ]

    print(f"Executing inference for {image_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n--- INFERENCE OUTPUT ---")
        print(result.stdout)
    else:
        print("\n--- ERROR ---")
        print(result.stderr)

if __name__ == "__main__":
    run_inference()
