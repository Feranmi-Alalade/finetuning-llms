import subprocess
import yaml
import json
import os
from pathlib import Path
from tqdm import tqdm

def run_evaluation():
    # Paths
    project_root = Path("/Users/david/codes-and-scripts/projects/bildcraft/research/finetuning-llms/multimodal-receipt-ai")
    prompt_file = project_root / "prompts/m0_ocr.yaml"
    schema_file = project_root / "prompts/schema/m0_ocr.json"
    model_dir = project_root / "outputs/fine_tuned_qwen_8b_dev/m0_ocr_lora_20260129_233429_dev_no_custom_loss"
    model_path = model_dir / "m0_q4_k_m.gguf"
    mmproj_path = model_dir / "m0_q4_k_m_mmproj.gguf"
    
    test_data_dir = project_root / "artifacts/receipts/gold/student/training/m0_ocr/data/test"
    test_labels_dir = project_root / "artifacts/receipts/gold/student/training/m0_ocr/labels/test"
    
    eval_dir = project_root / "evaluation/m0_ocr_evaluation"
    results_dir = eval_dir / "results"
    results_file = eval_dir / "evaluation_summary.jsonl"
    
    cli_path = project_root / "llama.cpp/build/bin/llama-mtmd-cli"

    # Load prompt and schema
    with open(prompt_file, "r") as f:
        prompts = yaml.safe_load(f)
        prompt = prompts["single_receipt"]
    
    with open(schema_file, "r") as f:
        schema = f.read().strip()
    
    # Inject schema
    full_prompt = prompt.replace("<SCHEMA>", schema)

    # Get test cases
    image_paths = list(test_data_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} test images.")

    eval_results = []

    # Filter Ground Truth to M0 schema
    def filter_gt(gt_full):
        if not gt_full or "response" not in gt_full:
            return gt_full
        resp = gt_full["response"]
        return {
            "document_type": resp.get("document_type"),
            "ocr": resp.get("ocr"),
            "quality_flags": resp.get("quality_flags"),
            "confidence": resp.get("confidence")
        }

    for img_path in tqdm(image_paths):
        label_path = test_labels_dir / f"{img_path.stem}.json"
        
        # Load ground truth
        gt = {}
        if label_path.exists():
            with open(label_path, "r") as f:
                gt = filter_gt(json.load(f))
        
        # Run inference
        cmd = [
            str(cli_path),
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "--image", str(img_path),
            "-p", full_prompt,
            "-n", "1024",
            "--temp", "0.1"
            # Removed --quiet
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        prediction = {}
        if result.returncode == 0:
            # Try to find JSON in output
            output = result.stdout
            try:
                # Basic JSON extraction - find the first { and last }
                start = output.find('{')
                end = output.rfind('}') + 1
                if start != -1 and end != -1:
                    prediction = json.loads(output[start:end])
                else:
                    prediction = {"error": "No JSON found in output", "raw": output}
            except Exception as e:
                prediction = {"error": f"Failed to parse JSON: {e}", "raw": output}
        else:
            prediction = {"error": f"CLI failed: {result.stderr}"}

        # Combine for saving
        eval_item = {
            "image": img_path.name,
            "ground_truth": gt,
            "prediction": prediction
        }
        
        eval_results.append(eval_item)
        
        # Save individual result
        with open(results_dir / f"{img_path.stem}_eval.json", "w") as f:
            json.dump(eval_item, f, indent=2)

    # Save summary
    with open(results_file, "w") as f:
        for item in eval_results:
            f.write(json.dumps(item) + "\n")

    print(f"\nEvaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    run_evaluation()
