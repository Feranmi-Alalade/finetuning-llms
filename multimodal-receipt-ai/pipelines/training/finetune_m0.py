"""Fine-tune Qwen3-VL 8B using TRL with PyTorch Lightning-style organization.

Supports LoRA and QLoRA fine-tuning with experiment tracking and automated model export.
Dev mode: Local Mac training (CPU/MPS)
Prod mode: Multi-GPU cluster training with FSDP
"""

import os

# Set global config for fine-tuning BEFORE other imports
os.environ["APP_CONFIG"] = "fine_tuning.yaml"
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import json

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
from jiwer import cer, wer


class GenerativeMetricsCallback(TrainerCallback):
    """Callback to compute generative metrics (CER, WER) during evaluation.

    This callback runs actual text generation on a subset of the evaluation set
    to measure real OCR quality (CER/WER) and auxiliary task performance.
    """

    def __init__(self, processor, eval_dataset, log_steps=10, use_custom_losses=True):
        """Initialize the generative metrics callback.

        Args:
            processor: Qwen processor for tokenization
            eval_dataset: Evaluation dataset to sample from
            log_steps: How often to log (unused, kept for compatibility)
            use_custom_losses: Whether to compute auxiliary metrics (field/rejection)
        """
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_steps = log_steps
        self.use_custom_losses = use_custom_losses

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Run generation on a subset of the evaluation set.

        Computes:
        - CER (Character Error Rate): Character-level accuracy
        - WER (Word Error Rate): Word-level accuracy
        - Field Accuracy: How well the model predicts field presence (if custom losses enabled)
        - Rejection Accuracy: How well the model rejects bad images (if custom losses enabled)
        
        Note: Skipped in dev mode to save memory.
        """
        # Skip expensive generation in dev mode to prevent OOM
        if hasattr(args, 'use_mps_device') and args.use_mps_device:
            logger.info("Skipping generative eval in dev mode (memory optimization)")
            return
            
        model = kwargs.get("model")
        tokenizer = self.processor.tokenizer
        device = model.device

        model.eval()

        # Take a subset for speed (2 samples in eval to minimize memory)
        num_samples = min(2, len(self.eval_dataset))

        preds = []
        labels_text = []

        # For custom losses - track field and rejection predictions
        if self.use_custom_losses:
            field_preds_list = []
            field_labels_list = []
            reject_preds_list = []
            reject_labels_list = []

        logger.info(f"Running generative evaluation on {num_samples} samples...")

        for i in range(num_samples):
            item = self.eval_dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            pixel_values = item["pixel_values"].to(device)
            image_grid_thw = item.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            # Find the split between instruction and response
            label_ids = item["labels"]
            response_start = (label_ids != -100).nonzero()
            if len(response_start) == 0:
                continue

            start_idx = response_start[0].item()
            prompt_ids = input_ids[:, :start_idx]
            ground_truth = tokenizer.decode(label_ids[start_idx:], skip_special_tokens=True)

            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": prompt_ids,
                    "pixel_values": pixel_values,
                    "max_new_tokens": 512,
                    "do_sample": False,
                }
                if image_grid_thw is not None:
                    generate_kwargs["image_grid_thw"] = image_grid_thw

                # Generate text
                generated_ids = model.generate(**generate_kwargs)

                # For custom losses, compute auxiliary predictions from forward pass
                if self.use_custom_losses and hasattr(model, "field_head"):
                    forward_outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                    )
                    last_hidden = forward_outputs.hidden_states[-1]
                    pooled = last_hidden.mean(dim=1)

                    # Get auxiliary predictions
                    if hasattr(model, "field_head"):
                        field_logits = model.field_head(pooled)
                        field_preds = (torch.sigmoid(field_logits) > 0.5).float()
                        field_preds_list.append(field_preds.cpu())
                        if "field_labels" in item:
                            field_labels_list.append(item["field_labels"])

                    if hasattr(model, "rejection_head"):
                        reject_logits = model.rejection_head(pooled)
                        reject_preds = (torch.sigmoid(reject_logits) > 0.5).float()
                        reject_preds_list.append(reject_preds.cpu())
                        if "rejection_labels" in item:
                            reject_labels_list.append(item["rejection_labels"].unsqueeze(0))

            # Decode output
            pred_text = tokenizer.decode(
                generated_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True
            )
            preds.append(pred_text)
            labels_text.append(ground_truth)

        # Compute OCR metrics
        if preds:
            cer_val = cer(labels_text, preds)
            wer_val = wer(labels_text, preds)

            logger.info(f"Generative Metrics - CER: {cer_val:.4f}, WER: {wer_val:.4f}")

            log_dict = {
                "eval/generative_cer": cer_val,
                "eval/generative_wer": wer_val,
            }

            # Compute auxiliary metrics if custom losses are enabled
            if self.use_custom_losses:
                if field_preds_list and field_labels_list:
                    field_preds_all = torch.cat(field_preds_list, dim=0)
                    field_labels_all = torch.stack(field_labels_list, dim=0)
                    field_acc = (field_preds_all == field_labels_all).float().mean()
                    log_dict["eval/generative_field_accuracy"] = field_acc.item()
                    logger.info(f"Field Accuracy: {field_acc.item():.4f}")

                if reject_preds_list and reject_labels_list:
                    reject_preds_all = torch.cat(reject_preds_list, dim=0)
                    reject_labels_all = torch.stack(reject_labels_list, dim=0)
                    reject_acc = (reject_preds_all == reject_labels_all).float().mean()
                    log_dict["eval/generative_reject_accuracy"] = reject_acc.item()
                    logger.info(f"Rejection Accuracy: {reject_acc.item():.4f}")

            # Log to Trainer
            trainer = kwargs.get("trainer")
            if trainer:
                trainer.log(log_dict)


        # Aggressive memory cleanup for MPS
        if torch.backends.mps.is_available():
            # Clear intermediate activations/tensors
            import gc
            gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        model.train()
        return control


from datasets import Dataset as HFDataset
from system.config import config, load_config
from system.logging import get_logger, disable_third_party_logs
from data.datasets.m0_dataset import M0Dataset, collate_fn
from data.utils.modeling import save_checkpoint
from data.utils.s3 import s3_sync_pipeline

# Try to import Qwen3-VL names
try:
    from transformers import Qwen3VLForConditionalGeneration as QwenModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenModel

# Config is automatically loaded via APP_CONFIG env var set at top

disable_third_party_logs()
logger = get_logger(__name__)


class M0SFTTrainer(SFTTrainer):
    """Custom SFTTrainer implementing the M0 Composite Objective.

    Objectives (when use_custom_losses=True):
    1. OCR Token Loss (Cross Entropy) - Weight: 0.75
    2. Field Presence Loss (BCE) - Weight: 0.15
    3. Rejection Loss (BCE) - Weight: 0.10
    """

    def __init__(self, *args, use_custom_losses=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_custom_losses = use_custom_losses
        logger.info(
            f"M0SFTTrainer initialized with custom_losses={'enabled' if use_custom_losses else 'disabled'}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to calculate weighted composite loss.

        M0 Composite Objective (when use_custom_losses=True):

        L_total = 0.75 * L_ocr + 0.15 * L_field + 0.10 * L_reject

        Where:
        - L_ocr: Standard token-level cross-entropy loss for OCR generation
        - L_field: Binary cross-entropy for 5 key fields (merchant, date, total, tax, line_items)
        - L_reject: Binary cross-entropy for image rejection (empty/non-receipt/severely degraded)

        When use_custom_losses=False:
        - Returns only L_ocr (standard supervised fine-tuning)

        Args:
            model: The PEFT model with optional auxiliary heads
            inputs: Batch dict with input_ids, pixel_values, labels, and optional auxiliary labels
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items (unused, for compatibility)

        Returns:
            loss (torch.Tensor): Scalar loss value
            OR
            (loss, outputs) tuple if return_outputs=True
        """

        # Extract auxiliary labels if custom losses are enabled
        # These are popped from inputs so they don't interfere with the model forward pass
        field_labels = inputs.pop("field_labels", None) if self.use_custom_losses else None
        reject_labels = inputs.pop("rejection_labels", None) if self.use_custom_losses else None

        # Forward pass
        if self.use_custom_losses:
            # Enable output_hidden_states to compute auxiliary losses
            # We need hidden states to get a pooled representation for auxiliary heads
            inputs["output_hidden_states"] = True
            outputs = model(**inputs)
        else:
            # Standard forward pass without hidden states (more efficient)
            outputs = model(**inputs)

        # 1. OCR Token Loss (L_ocr)
        # This is the standard language modeling loss from the base model
        ocr_loss = outputs.loss

        if not self.use_custom_losses:
            # Return standard loss without custom objectives (sanity check mode)
            return (ocr_loss, outputs) if return_outputs else ocr_loss

        # 2. Custom Auxiliary Objectives
        # We use the mean of the last hidden state as a global representation
        # This pooled vector encodes the entire image + prompt understanding
        last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        pooled_output = last_hidden_state.mean(dim=1)  # [batch, hidden_dim]

        # Field Presence Loss (L_field)
        # Predicts which of the 5 key fields are present in the receipt
        field_loss = torch.tensor(0.0, device=ocr_loss.device)
        if field_labels is not None and hasattr(model, "field_head"):
            field_logits = model.field_head(pooled_output)  # [batch, 5]
            field_loss = F.binary_cross_entropy_with_logits(field_logits, field_labels.float())

        # Rejection Loss (L_reject)
        # Predicts whether the image should be rejected (empty, non-receipt, or severely degraded)
        reject_loss = torch.tensor(0.0, device=ocr_loss.device)
        if reject_labels is not None and hasattr(model, "rejection_head"):
            reject_logits = model.rejection_head(pooled_output)  # [batch, 1]
            reject_loss = F.binary_cross_entropy_with_logits(
                reject_logits, reject_labels.float().unsqueeze(-1)
            )

        # 3. Weighted Sum (from m0_ocr.yaml specification)
        total_loss = (0.75 * ocr_loss) + (0.15 * field_loss) + (0.10 * reject_loss)

        # Log individual losses for monitoring
        # We log at every logging_steps during training, and always during evaluation
        if self.state.global_step % self.args.logging_steps == 0 or not model.training:
            prefix = "train" if model.training else "eval"
            log_dict = {
                f"{prefix}/ocr_loss": ocr_loss.item(),
                f"{prefix}/field_loss": field_loss.item(),
                f"{prefix}/reject_loss": reject_loss.item(),
                f"{prefix}/total_loss": total_loss.item(),
            }

            # Add accuracies for auxiliary heads
            # These help us monitor training progress on the auxiliary tasks
            if field_labels is not None and hasattr(model, "field_head"):
                with torch.no_grad():
                    field_preds = (torch.sigmoid(field_logits) > 0.5).float()
                    field_acc = (field_preds == field_labels).float().mean()
                    log_dict[f"{prefix}/field_accuracy"] = field_acc.item()

            if reject_labels is not None and hasattr(model, "rejection_head"):
                with torch.no_grad():
                    reject_preds = (torch.sigmoid(reject_logits) > 0.5).float()
                    reject_acc = (reject_preds == reject_labels.unsqueeze(-1)).float().mean()
                    log_dict[f"{prefix}/reject_accuracy"] = reject_acc.item()

            self.log(log_dict)

        return (total_loss, outputs) if return_outputs else total_loss


class M0Module:
    """Lightning-style module for M0 OCR Fine-tuning using TRL."""

    def __init__(self, mode: str = "lora", env: str = "dev", use_custom_losses: bool = True):
        """Initialize M0Module.

        Args:
            mode: Training mode ('lora' or 'qlora')
            env: Environment ('dev' for local Mac, 'prod' for GPU cluster)
            use_custom_losses: Whether to use custom loss functions
        """
        self.mode = mode
        self.env = env
        self.use_custom_losses = use_custom_losses
        self.settings = config.models.fine_tune_base
        self.train_cfg = config.training
        
        # Map string dtype to torch dtype
        self.dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        self.torch_dtype = self.dtype_mapping.get(self.settings.dtype.lower(), torch.bfloat16)

        logger.info(
            f"Initializing M0Module in {mode} mode for {env} environment "
            f"(custom_losses={'enabled' if use_custom_losses else 'disabled'})"
        )

        self.processor = None
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize Qwen3-VL model with LoRA/QLoRA configuration.

        Steps:
        1. Load processor for tokenization and image processing
        2. Configure model loading based on environment (dev/prod) and mode (lora/qlora)
        3. Load base Qwen3-VL model
        4. Move to appropriate device (CPU/MPS for dev, GPU for prod)
        5. Attach auxiliary heads for custom losses (if enabled)
        6. Prepare model for PEFT (LoRA/QLoRA)
        7. Apply PEFT configuration and print trainable parameters
        """
        logger.info(f"Loading model: {self.settings.name}")

        # 1. Load Processor
        # The processor handles both text tokenization and image preprocessing
        self.processor = AutoProcessor.from_pretrained(
            self.settings.name, trust_remote_code=self.settings.trust_remote_code
        )

        # 2. Configure model kwargs based on environment and mode
        model_kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
            "attn_implementation": self.settings.attn_implementation,  # Usually "flash_attention_2"
        }

        # Use the resolved torch_dtype from __init__
        torch_dtype = self.torch_dtype

        # Dev mode: Use CPU/MPS on Mac (no quantization for compatibility)
        if self.env == "dev":
            # Respect config dtype for memory efficiency
            logger.info(f"Dev mode: Using CPU/MPS device with dtype {torch_dtype}")
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = None  # Manual device placement
        else:
            # Prod mode: GPU cluster with optional quantization
            model_kwargs["torch_dtype"] = torch_dtype

            # QLoRA: 4-bit quantization for memory efficiency
            if self.mode == "qlora":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,  # Double quantization for better quality
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                )
                logger.info("Prod mode: Using 4-bit quantization (QLoRA)")
            else:
                logger.info("Prod mode: Using standard LoRA")

        # 3. Load Base Model
        self.model = QwenModel.from_pretrained(self.settings.name, **model_kwargs)

        # 4. Move to device in dev mode (prod mode uses auto device_map)
        if self.env == "dev":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
            self.model = self.model.to(device)

        # 5. Attach Auxiliary Heads (only if custom losses enabled)
        # These are simple linear layers that predict field presence and rejection
        if self.use_custom_losses:
            # Get hidden size from model config
            if hasattr(self.model.config, "text_config"):
                hidden_size = self.model.config.text_config.hidden_size
            else:
                hidden_size = self.model.config.hidden_size

            # Field head: predicts 5 binary values (one per key field)
            self.model.field_head = torch.nn.Linear(hidden_size, 5).to(self.model.device)

            # Rejection head: predicts 1 binary value (reject/accept)
            self.model.rejection_head = torch.nn.Linear(hidden_size, 1).to(self.model.device)

            logger.info(f"Attached auxiliary heads with hidden_size={hidden_size}")
        else:
            logger.info("Skipping auxiliary heads (custom losses disabled)")

        if getattr(self.train_cfg, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False  # Required for gradient checkpointing
            logger.info("Gradient checkpointing enabled")

        # 6. Prepare for PEFT (Parameter-Efficient Fine-Tuning)
        if self.mode == "qlora" and self.env == "prod":
            # Prepare quantized model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)

        # 7. Configure and apply LoRA
        # modules_to_save: Additional modules to save in the adapter (not frozen)
        modules_to_save = ["field_head", "rejection_head"] if self.use_custom_losses else []

        peft_config = LoraConfig(
            r=self.train_cfg.lora_r,  # LoRA rank (usually 8-32)
            lora_alpha=self.train_cfg.lora_alpha,  # LoRA scaling factor
            target_modules=self.train_cfg.lora_target_modules,  # Which modules to apply LoRA to
            lora_dropout=self.train_cfg.lora_dropout,  # Dropout for LoRA layers
            bias="none",  # Don't adapt bias parameters
            task_type="CAUSAL_LM",  # Causal language modeling task
            modules_to_save=modules_to_save,  # Save auxiliary heads
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # Shows how many params are trainable

    def get_datasets(self):
        """Load training and validation datasets."""
        data_base_dir = config.paths.artifacts_dir / "receipts/platinum/m0_ocr/student"

        # Load M0 OCR Prompt & Schema
        prompts_path = Path("prompts/m0_ocr.yaml")
        schema_path = Path("prompts/schema/m0_ocr.json")
        
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
            prompt_template = prompts["single_receipt"]
            
        # Inject M0 Schema
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema_json = f.read().strip()
            prompt_template = prompt_template.replace("<SCHEMA>", schema_json)
        else:
            logger.warning(f"Schema not found at {schema_path}, prompt may contain <SCHEMA> placeholder")

        def create_hf_dataset(split):
            dataset = M0Dataset(
                split=split,
                data_dir=data_base_dir,
                processor=self.processor,
                prompt_template=prompt_template,
                max_length=self.train_cfg.max_seq_length,
                use_custom_losses=self.use_custom_losses,
            )

            def gen():
                for i in range(len(dataset)):
                    yield dataset[i]

            return HFDataset.from_generator(gen, keep_in_memory=True)

        train_ds = create_hf_dataset("train")
        val_ds = create_hf_dataset("test")

        logger.info(f"Loaded training and test datasets")
        return train_ds, val_ds

    def configure_training_args(self, output_dir: Path) -> SFTConfig:
        """Configure training arguments based on environment."""

        args_dict = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": self.train_cfg.batch_size,
            "per_device_eval_batch_size": getattr(self.train_cfg, "eval_batch_size", 1),
            "gradient_accumulation_steps": self.train_cfg.gradient_accumulation_steps,
            "learning_rate": self.train_cfg.learning_rate,
            "lr_scheduler_type": self.train_cfg.lr_scheduler_type,
            "warmup_ratio": self.train_cfg.warmup_ratio,
            "weight_decay": self.train_cfg.weight_decay,
            "num_train_epochs": self.train_cfg.num_train_epochs,
            "max_steps": self.train_cfg.max_steps,
            "logging_steps": self.train_cfg.logging_steps,
            "save_steps": self.train_cfg.save_steps,
            "eval_steps": self.train_cfg.eval_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval/total_loss" if self.use_custom_losses else "loss",
            "greater_is_better": False,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "gradient_checkpointing": getattr(self.train_cfg, "gradient_checkpointing", True),
            "dataloader_num_workers": 0 if self.env == "dev" else getattr(config.data_prep, "num_workers", 0),
            "max_length": self.train_cfg.max_seq_length,
        }

        if self.env == "dev":
            logger.info("Configuring for dev mode (local Mac)")
            # Determine precision based on model dtype
            use_bf16 = self.torch_dtype == torch.bfloat16 and torch.backends.mps.is_available()
            use_fp16 = self.torch_dtype == torch.float16 and torch.backends.mps.is_available()
            
            args_dict.update(
                {
                    # Respect config values, don't override
                    "use_cpu": not torch.backends.mps.is_available(),
                    "use_mps_device": torch.backends.mps.is_available(),
                    "fp16": use_fp16,
                    "bf16": use_bf16,
                    "report_to": "wandb" if config.logging.wandb_project else "none",
                    # Memory optimizations for Mac
                    "eval_accumulation_steps": 1,  # Don't accumulate eval batches
                    "dataloader_pin_memory": False,  # MPS doesn't support pinned memory
                }
            )
            logger.info(f"Dev precision: bf16={use_bf16}, fp16={use_fp16}")
        else:
            logger.info("Configuring for prod mode (GPU cluster with FSDP)")
            args_dict.update(
                {
                    "bf16": torch.cuda.is_bf16_supported(),
                    "fp16": not torch.cuda.is_bf16_supported(),
                    "report_to": "wandb",
                    "ddp_find_unused_parameters": False,
                    "fsdp": "full_shard auto_wrap",
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2VLDecoderLayer",
                }
            )

        return SFTConfig(**args_dict)

    def compute_metrics(self, eval_preds):
        """Placeholder - metrics computed in callbacks."""
        logits, labels = eval_preds

        # Standard SFTTrainer might return a tuple of (logits, auxiliary_logits)
        # Handle the structure of eval_preds
        if isinstance(logits, tuple):
            text_logits = logits[0]
            # Aux logits are likely concatenated or separate depending on model setup
            # For our patched heads, we might need a custom strategy if Trainer doesn't
            # automatically capture them.
            pass
        else:
            text_logits = logits

        # 1. OCR Metrics (CER/WER)
        # Note: In-trainer metrics usually use logits. For CER/WER we need decoded text.
        # SFTTrainer evaluation doesn't easily provide decoded text by default
        # without a significant performance hit (generative eval).
        # We'll log the base losses as proxies and implement a lighter check.

        # 2. Auxiliary Head Accuracies
        # We'll use the logits from the heads if they are visible in eval_preds.
        # If not, we rely on the val_loss components logged in compute_loss.

        return {}  # Placeholder as Trainer handles loss components automatically in logs


class M0Trainer:
    """Trainer orchestrator for M0 fine-tuning."""

    def __init__(self, mode: str = "lora", env: str = "dev", use_custom_losses: bool = True):
        self.mode = mode
        self.env = env
        self.use_custom_losses = use_custom_losses
        self.train_cfg = config.training
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.train_cfg.experiment_name:
            self.exp_name = self.train_cfg.experiment_name
        else:
            suffix = "" if use_custom_losses else "_no_custom_loss"
            self.exp_name = f"m0_ocr_{mode}_{self.timestamp}_{env}{suffix}"

        self.output_dir = Path(self.train_cfg.output_dir) / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized M0Trainer: {self.exp_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Custom losses: {'enabled' if use_custom_losses else 'disabled'}")

        self.module = M0Module(mode=mode, env=env, use_custom_losses=use_custom_losses)

    def train(self):
        """Execute training using Custom M0SFTTrainer."""
        train_ds, val_ds = self.module.get_datasets()
        training_args = self.module.configure_training_args(self.output_dir)
        
        # Ensure eval batch size is set for memory safety
        if training_args.per_device_eval_batch_size is None:
            training_args.per_device_eval_batch_size = 1

        # Initialize W&B if enabled
        if config.logging.wandb_api_key or (
            hasattr(config.logging, "wandb_project") and config.logging.wandb_project
        ):
            try:
                import wandb

                # Initialize W&B with informative experiment name and config
                wandb.init(
                    project=config.logging.wandb_project,
                    name=self.exp_name,
                    config={
                        "mode": self.mode,
                        "env": self.env,
                        "use_custom_losses": self.use_custom_losses,
                        "learning_rate": self.train_cfg.learning_rate,
                        "batch_size": self.train_cfg.batch_size,
                        "epochs": self.train_cfg.num_train_epochs,
                        "lora_r": self.train_cfg.lora_r,
                        "lora_alpha": self.train_cfg.lora_alpha,
                        "objective_weights": {"ocr": 0.75, "field": 0.15, "reject": 0.10}
                        if self.use_custom_losses
                        else {"ocr": 1.0},
                    },
                    tags=[
                        f"m0_ocr",
                        f"{self.mode}",
                        f"{self.env}",
                        "custom_losses" if self.use_custom_losses else "ocr_only",
                    ],
                )
                logger.info(f"W&B initialized: {self.exp_name}")
            except Exception as e:
                logger.warning(f"Could not initialize W&B: {e}")

        # Initialize Custom M0SFTTrainer
        trainer = M0SFTTrainer(
            model=self.module.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=self.module.compute_metrics,
            callbacks=[
                GenerativeMetricsCallback(
                    self.module.processor, val_ds, use_custom_losses=self.use_custom_losses
                )
            ],
            use_custom_losses=self.use_custom_losses,
        )

        logger.info(
            f"Starting training with M0 Composite Objective (custom_losses={self.use_custom_losses})..."
        )
        trainer.train()

        self.save_adapter()

    def save_adapter(self):
        """Save LoRA adapter."""
        adapter_path = self.output_dir / "adapter"
        save_checkpoint(self.module.model, self.module.processor, adapter_path)

        # Save Metadata for Deployment Validation
        metadata = {
            "stage": "m0",
            "base_model": self.module.settings.name,
            "timestamp": self.timestamp,
            "env": self.env,
            "mode": self.mode,
            "custom_losses": self.use_custom_losses,
        }
        with open(adapter_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved training metadata to {adapter_path}/training_metadata.json")


@s3_sync_pipeline
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL using TRL")
    parser.add_argument("--env", type=str, default="dev", help="Environment (dev/prod)")
    parser.add_argument("--mode", type=str, default="lora", help="Mode (lora/qlora)")
    parser.add_argument(
        "--no-custom-losses",
        action="store_true",
        help="Disable custom loss functions (field + rejection). Use for sanity checking.",
    )
    args = parser.parse_args()

    os.environ["ENV"] = args.env

    use_custom_losses = not args.no_custom_losses

    trainer = M0Trainer(mode=args.mode, env=args.env, use_custom_losses=use_custom_losses)
    trainer.train()


if __name__ == "__main__":
    main()
