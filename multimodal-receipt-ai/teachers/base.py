"""Teacher models for generating training labels."""

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from data.utils.image_processing import Qwen3ImageProcessor
from data.utils.validation import SchemaValidator
from prompts.schema.teacher_inference_model import TeacherInferenceResponse
from system.config import ModelSettings, config
from system.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound="Teacher")


class RateLimiter:
    """Simple rate limiter to enforce API constraints."""

    def __init__(self, max_calls: int, period: float, sleep_after_batch: float = 0.0):
        self.max_calls = max_calls
        self.period = period
        self.sleep_after_batch = sleep_after_batch
        self.call_times = []
        self.total_calls = 0

    def wait(self):
        """Wait if necessary to remain within rate limits."""
        now = time.time()

        # Remove calls outside the current period
        self.call_times = [t for t in self.call_times if t > now - self.period]

        if len(self.call_times) >= self.max_calls:
            sleep_time = self.call_times[0] + self.period - now
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached ({self.max_calls} calls /{self.period}s). Sleeping for {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

        # Enforce batch-level break if requested
        if (
            self.sleep_after_batch > 0
            and self.total_calls > 0
            and self.total_calls % self.max_calls == 0
        ):
            logger.debug(f"Enforcing mandatory {self.sleep_after_batch}s break.")
            time.sleep(self.sleep_after_batch)

        self.call_times.append(time.time())
        self.total_calls += 1


class Teacher(ABC):
    """Abstract teacher model interface with built-in caching and validation."""

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.validator = SchemaValidator(TeacherInferenceResponse)
        self.telemetry = {
            "total_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "batch_efficiency": [],
        }

    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate a stable MD5 hash for image content."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def _get_cache_path(self, image: Image.Image) -> Optional[Path]:
        """Get path to cached response file."""
        if not self.cache_dir:
            return None
        image_hash = self._get_image_hash(image)
        return self.cache_dir / f"{image_hash}.json"

    def _save_to_cache(self, image: Image.Image, response: str):
        """Save successful and valid response to cache."""
        cache_path = self._get_cache_path(image)
        if not cache_path or not response or response.strip().lower() == "null":
            return

        # Double check validity before caching
        if not self.validator.is_valid(response):
            return

        cache_data = {
            "model": self.model_name,
            "timestamp": time.time(),
            "response": response,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _load_from_cache(self, image: Image.Image) -> Optional[str]:
        """Load response from cache if it exists and is valid."""
        cache_path = self._get_cache_path(image)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    response = data.get("response")
                    if response and response.strip().lower() != "null":
                        # Validate cached content to be safe
                        if self.validator.is_valid(response):
                            self.telemetry["cache_hits"] += 1
                            return response
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> str:
        """Generate response from single image and prompt."""
        pass

    @abstractmethod
    def batch_generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> List[str]:
        """Generate responses from batch of images and prompts."""
        pass

    def count_tokens(self, contents: List[Any]) -> int:
        """Count tokens for a list of contents (text or images)."""
        count = 0
        for item in contents:
            if isinstance(item, str):
                count += len(item) // 4
            elif isinstance(item, Image.Image):
                count += 1000  # Rough ballpark fallback
        return count

    def get_model_info(self) -> Dict[str, Any]:
        """Provide detailed model metadata and telemetry."""
        info = {
            "model": self.model_name,
            "telemetry": self.telemetry,
        }
        if self.telemetry["total_requests"] > 0:
            eff = self.telemetry["batch_efficiency"]
            info["avg_batch_efficiency"] = sum(eff) / len(eff) if eff else 1
        return info


class GeminiTeacher(Teacher):
    """Teacher using Google's Gemini SDK (v3 / google-genai)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(model_name, cache_dir)
        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=api_key)
            self.types = types
        except ImportError:
            logger.error("google-genai not installed. Run 'pip install google-genai'")
            raise

        self.model_id = model_name
        self.rate_limiter = RateLimiter(max_calls=2, period=60, sleep_after_batch=10.0)

    def count_tokens(self, contents: Any) -> int:
        """Calculate prompt and image token counts using the SDK."""
        try:
            if not isinstance(contents, list):
                contents = [contents]
            response = self.client.models.count_tokens(model=self.model_id, contents=contents)
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Gemini token count failed: {e}")
            return super().count_tokens(contents)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> str:
        if use_cache:
            cached = self._load_from_cache(image)
            if cached:
                return cached

        self.rate_limiter.wait()
        self.telemetry["total_requests"] += 1

        contents = [prompt, image]
        self.telemetry["total_tokens"] += self.count_tokens(contents)

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            text = response.text
            if use_cache and text:
                self._save_to_cache(image, text)
            return text or "null"
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return "null"

    def batch_generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> List[str]:
        if not images:
            return []

        # Logic: prompts[0] is batch_prompt, prompts[1] is single_prompt (for retries)
        batch_prompt = prompts[0] if prompts else "Analyze these receipts."
        single_prompt = prompts[1] if len(prompts) > 1 else batch_prompt

        results_map = {}
        to_process_indices = []

        # 1. Check Cache
        for i, img in enumerate(images):
            if use_cache:
                cached = self._load_from_cache(img)
                if cached:
                    results_map[i] = cached
                    continue
            to_process_indices.append(i)

        if not to_process_indices:
            return [results_map[i] for i in range(len(images))]

        # 2. Execute Batch Request
        batch_images = [images[i] for i in to_process_indices]
        self.rate_limiter.wait()
        self.telemetry["total_requests"] += 1

        contents = [batch_prompt] + batch_images
        self.telemetry["total_tokens"] += self.count_tokens(contents)
        self.telemetry["batch_efficiency"].append(len(batch_images))

        logger.info(f"Sending batch request for {len(batch_images)} images")

        raw_batch_response = ""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            raw_batch_response = response.text
        except Exception as e:
            logger.error(f"Gemini batch request failed: {e}")

        # 3. Parse and Validate
        failed_indices = []
        if raw_batch_response:
            parsed_results, unfound_indices = self._parse_batch_data_response(
                raw_batch_response, len(to_process_indices)
            )
            # Map logical batch IDs back to original image indices
            for batch_idx, text in parsed_results.items():
                orig_idx = to_process_indices[batch_idx]
                is_valid, _, error_dict = self.validator.validate(
                    text, context=f"batch_index_{orig_idx}"
                )
                if is_valid:
                    results_map[orig_idx] = text
                    if use_cache:
                        self._save_to_cache(images[orig_idx], text)
                else:
                    logger.warning(
                        f"Validation failed for image {orig_idx}: {json.dumps(error_dict)}"
                    )
                    failed_indices.append(orig_idx)

            # Record indices that weren't even in the parse result
            for batch_idx in unfound_indices:
                failed_indices.append(to_process_indices[batch_idx])
        else:
            failed_indices = to_process_indices

        # 4. Individual Retries for Failures
        if failed_indices:
            logger.info(f"Retrying {len(failed_indices)} items individually...")
            for idx in failed_indices:
                res = self.generate(
                    images[idx],
                    single_prompt,
                    max_tokens=max_tokens // len(images),
                    temperature=temperature,
                    use_cache=use_cache,
                )
                is_valid, _, error_dict = self.validator.validate(res, context=f"retry_index_{idx}")
                if is_valid:
                    results_map[idx] = res
                else:
                    logger.error(
                        f"Individual retry also failed validation for {idx}: {json.dumps(error_dict)}"
                    )
                    results_map[idx] = "null"

        return [results_map.get(i, "null") for i in range(len(images))]

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract balanced JSON object/list, handling strings and escapes."""
        stack, in_string, escape = [], False, False
        for i, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string, escape = True, False
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    return None
                if (char == "}" and stack[-1] == "{") or (char == "]" and stack[-1] == "["):
                    stack.pop()
                    if not stack:
                        return text[: i + 1]
                else:
                    return None
        return None

    def _parse_batch_data_response(self, text: str, count: int) -> Tuple[Dict[int, str], List[int]]:
        """Robustly extract individual item JSONs from a potentially messy batch response."""
        results = {}
        failed = list(range(count))

        clean = re.sub(r"^```(?:json)?\s*", "", text.strip())
        clean = re.sub(r"\s*```$", "", clean)

        # Try full parse first
        try:
            data = json.loads(clean)
            items = data if isinstance(data, list) else [data.get(str(i)) for i in range(count)]
            for i, val in enumerate(items):
                if val:
                    results[i] = json.dumps(val)
                    if i in failed:
                        failed.remove(i)
        except Exception:
            logger.debug("Full batch JSON parse failed, attempting heuristic recovery.")

        if not failed:
            return results, []

        # Heuristic regex recovery for missing indices
        for i in failed[:]:
            for pattern in [f'"{i}"\\s*:', f"'{i}'\\s*:", f"{i}\\s*:"]:
                match = re.search(pattern, clean)
                if match:
                    brace_start = clean.find("{", match.end())
                    if brace_start != -1:
                        obj = self._extract_balanced_json(clean[brace_start:])
                        if obj:
                            try:
                                json.loads(obj)
                                results[i] = obj
                                failed.remove(i)
                                break
                            except:
                                pass
        return results, failed


class QwenTeacher(Teacher):
    """Teacher using local Qwen3-VL model from HuggingFace."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
    ):
        settings: ModelSettings = config.active_model_settings
        model_name = model_name or settings.name
        super().__init__(model_name, cache_dir)

        self.image_processor = Qwen3ImageProcessor()
        self._setup_device(device or settings.device)

        logger.info(f"Loading Qwen model: {model_name} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=settings.trust_remote_code
        )

        model_kwargs = {
            "trust_remote_code": settings.trust_remote_code,
            "attn_implementation": settings.attn_implementation,
        }
        self._configure_quantization(model_kwargs, settings, load_in_4bit, load_in_8bit)

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
            if self.device in ["cuda", "mps"] and "quantization_config" not in model_kwargs:
                self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

    def _setup_device(self, device: str):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _configure_quantization(
        self, kwargs: Dict, settings: ModelSettings, bit4: bool, bit8: bool
    ):
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(settings.dtype.lower(), torch.float32)

        bit4 = bit4 if bit4 is not None else settings.load_in_4bit
        bit8 = bit8 if bit8 is not None else settings.load_in_8bit

        if self.device == "cuda" and (bit4 or bit8):
            if bit4:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype
                )
            else:
                kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch_dtype

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> str:
        if use_cache:
            cached = self._load_from_cache(image)
            if cached:
                return cached

        self.telemetry["total_requests"] += 1
        processed_img = self.image_processor.process_for_inference(image)

        inputs = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        max_new_tokens = max_tokens or config.generation.out_seq_length
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature or config.generation.temperature,
                do_sample=not config.generation.greedy,
            )

        trimmed_ids = [ids[len(inputs.input_ids[0]) :] for ids in generated_ids]
        response = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

        if use_cache:
            self._save_to_cache(image, response)
        return response

    def batch_generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[str]:
        """High-throughput batch generation with per-item caching."""
        results_map = {}
        to_process_indices = []

        # 1. Filter assignments from cache
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            if use_cache:
                cached = self._load_from_cache(img)
                if cached:
                    results_map[i] = cached
                    continue
            to_process_indices.append(i)

        if not to_process_indices:
            return [results_map[i] for i in range(len(images))]

        # 2. Parallel Generation for cache misses
        logger.info(f"Running Qwen batch inference for {len(to_process_indices)} images")
        batch_messages = []
        for idx in to_process_indices:
            processed_img = self.image_processor.process_for_inference(images[idx])
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": processed_img},
                            {"type": "text", "text": prompts[idx]},
                        ],
                    }
                ]
            )

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or config.generation.out_seq_length,
                temperature=temperature or config.generation.temperature,
                do_sample=not config.generation.greedy,
            )

        # 3. Decode, Validate, and Cache
        input_len = inputs.input_ids.shape[1]
        trimmed_ids = [ids[input_len:] for ids in generated_ids]
        raw_responses = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)

        for i, raw_res in enumerate(raw_responses):
            orig_idx = to_process_indices[i]
            is_valid, _, error_dict = self.validator.validate(
                raw_res, context=f"qwen_batch_{orig_idx}"
            )
            if is_valid:
                results_map[orig_idx] = raw_res
                if use_cache:
                    self._save_to_cache(images[orig_idx], raw_res)
            else:
                logger.warning(
                    f"Qwen batch item {orig_idx} failed validation: {json.dumps(error_dict)}. Retrying individually..."
                )
                # Immediate individual retry for failed batch items
                retry_res = self.generate(
                    images[orig_idx], prompts[orig_idx], max_tokens, temperature, use_cache=False
                )
                is_valid_retry, _, error_dict_retry = self.validator.validate(
                    retry_res, context=f"qwen_retry_{orig_idx}"
                )
                if is_valid_retry:
                    results_map[orig_idx] = retry_res
                else:
                    logger.error(
                        f"Qwen retry also failed for {orig_idx}: {json.dumps(error_dict_retry)}"
                    )
                    results_map[orig_idx] = "null"

        return [results_map.get(i, "null") for i in range(len(images))]
