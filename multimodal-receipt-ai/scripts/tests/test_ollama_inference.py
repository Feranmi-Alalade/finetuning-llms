import requests
import json
import base64
import argparse
from pathlib import Path

def load_model_prompt(model_name: str):
    """Load the prompt for a specific model."""
    prompt_map = {
        "bildcraft/receipt-ai-m0": "m0_ocr",
        "bildcraft/receipt-ai-m1": "m1_semantic",
        "bildcraft/receipt-ai-unified": "unified"
    }
    
    # Identify model type
    prompt_key = "teacher_inference"
    for key, value in prompt_map.items():
        if model_name.startswith(key) or model_name == key:
            prompt_key = value
            break
            
    # Default fallback logic for convenience
    if prompt_key == "teacher_inference" and "m0" in model_name: prompt_key = "m0_ocr"
    if prompt_key == "teacher_inference" and "m1" in model_name: prompt_key = "m1_semantic"
    
    prompt_file = f"prompts/{prompt_key}.yaml"
    schema_file = f"prompts/schema/{prompt_key}.json"
    
    # Special handle for teacher default
    if prompt_key == "teacher_inference":
        schema_file = "prompts/schema/teacher_inference.json"
        
    prompt_path = Path(prompt_file)
    schema_path = Path(schema_file)
    
    if not prompt_path.exists():
        print(f"Warning: Prompt file {prompt_path} not found.")
        return "Perform OCR on this receipt."
        
    try:
        import yaml
        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
            prompt = prompts.get("single_receipt", "Perform OCR on this receipt.")
            
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema = f.read().strip()
            prompt = prompt.replace("<SCHEMA>", schema)
        else:
             print(f"Warning: Schema file {schema_path} not found.")
            
        return prompt
    except Exception as e:
        print(f"Warning: Failed to load prompt: {e}")
        return "Perform OCR on this receipt."

def test_inference(model_name: str, image_path: str):
    print(f"Testing inference on {model_name} with {image_path}...")
    
    prompt = load_model_prompt(model_name)
    url = "http://localhost:11434/api/generate"
    
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- INFERENCE RESULT ---\n")
            print(result.get("response", "No response field in JSON"))
            print("\n------------------------\n")
        else:
            print(f"Error: Ollama API returned status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Detail: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Raw body: {response.text}")
            
            if response.status_code == 500:
                print("\n💡 TIP: A 500 error usually means your Ollama version is too old for this model architecture.")
                print("   Try running 'brew upgrade ollama' or downloading the latest version from ollama.com.")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Is the server running?")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bildcraft/receipt-ai-m0")
    parser.add_argument("--image", type=str, help="Path to test image")
    args = parser.parse_args()
    
    if not args.image:
        # Try to find a sample image in artifacts
        search_dirs = [
            Path("artifacts/receipts/silver/base"),
            Path("artifacts/receipts/silver/stability/non_receipt/images"),
            Path("artifacts/receipts/silver/stability/blurry/images"),
        ]
        
        found = False
        for d in search_dirs:
            if d.exists():
                samples = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if samples:
                    args.image = str(samples[0])
                    print(f"Using auto-detected sample: {args.image}")
                    found = True
                    break
        
        if not found:
            print("No sample images found in artifacts.")
            exit(1)

    test_inference(args.model, args.image)
