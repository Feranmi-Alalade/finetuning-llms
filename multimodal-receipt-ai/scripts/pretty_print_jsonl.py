import sys
import os
import json
from pathlib import Path

# Add the project root to PYTHONPATH so we can import from data, system, etc.
# This assumes the script is in project_root/scripts/
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from data.utils.parsing import parse_json_response, print_parsed_response


def interactive_pretty_print():
    print("\n" + "=" * 60)
    print("      RECEIPT JSONL PRETTY PRINTER")
    print("=" * 60)

    # Prompt for file path
    default_path = "data/artifacts/shared/ocr_structured_labels.jsonl"
    print(f"\nDefault path: {default_path}")
    file_path_input = input("Enter path to JSONL file (or press Enter for default): ").strip()
    file_path = file_path_input if file_path_input else default_path

    full_path = project_root / file_path
    if not full_path.exists():
        # Try absolute path just in case
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"\n[ERROR] File not found: {file_path}")
            return

    # Read all lines
    try:
        with open(full_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"\n[ERROR] Failed to read file: {e}")
        return

    num_lines = len(lines)
    if num_lines == 0:
        print(f"\n[WARNING] File is empty: {full_path}")
        return

    print(f"\n[SUCCESS] Loaded {num_lines} lines from {full_path.name}")

    while True:
        print("\n" + "-" * 40)
        line_num_input = input(
            f"Enter line number (1-{num_lines}) to pretty print, or 'q' to quit: "
        ).strip()

        if line_num_input.lower() == "q":
            print("\nExiting. Goodbye!\n")
            break

        try:
            line_num = int(line_num_input)
            line_idx = line_num - 1

            if 0 <= line_idx < num_lines:
                line_data = json.loads(lines[line_idx])

                # Check for 'response' field which contains the structured JSON
                response_text = line_data.get("response", "")

                if not response_text:
                    # Maybe it's a student output with a different key?
                    # Let's check common keys
                    response_text = line_data.get("prediction", line_data.get("output", ""))

                if not response_text:
                    print(
                        f"\n[WARNING] Line {line_num} does not contain a recognizable 'response' or 'prediction' field."
                    )
                    print(f"Available keys: {list(line_data.keys())}")
                    continue

                # Use the provided project function to parse and print
                parsed_data = parse_json_response(response_text)
                if parsed_data:
                    print(f"\n--- Line {line_num} Content ---")
                    image_path = line_data.get("image_path", "N/A")
                    model = line_data.get("teacher_model", line_data.get("model", "N/A"))
                    print(f"Image: {image_path}")
                    print(f"Model: {model}")

                    print_parsed_response(parsed_data)
                else:
                    print(
                        f"\n[ERROR] Could not parse valid JSON from the response text in line {line_num}."
                    )
                    print("Raw response text preview:")
                    print(
                        f"{response_text[:200]}..." if len(response_text) > 200 else response_text
                    )
            else:
                print(
                    f"\n[ERROR] Invalid line number. Please enter a value between 1 and {num_lines}."
                )
        except ValueError:
            print("\n[ERROR] Invalid input. Please enter a number or 'q'.")
        except json.JSONDecodeError:
            print(f"\n[ERROR] Line {line_num} is not a valid JSON object.")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        interactive_pretty_print()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
