"""Global system configuration using YAML and Dataclasses."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict, Type, TypeVar, List
from dotenv import load_dotenv

# Load environment variables from the project root's parent (.env is in finetuning-llms/)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

T = TypeVar("T")


@dataclass
class PathConfig:
    project_root: Path
    data_dir: Path
    sources_dir: Path
    artifacts_dir: Path
    manifests_dir: Path
    logs_dir: Path
    cache_dir: Path


@dataclass
class ModelSettings:
    name: str = ""
    dtype: str = "float16"
    device: str = "auto"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class ModelConfig:
    dev_teacher: ModelSettings = field(default_factory=ModelSettings)
    prod_teacher: ModelSettings = field(default_factory=ModelSettings)
    fine_tune_base: ModelSettings = field(default_factory=ModelSettings)


@dataclass
class DataConfig:
    dtype: str = "float32"
    ignore_index: int = -100

    # Task and prompt paths
    prompt_files: Dict[str, str] = field(
        default_factory=lambda: {
            "prompt": "prompts/teacher_inference.yaml",
            "schema": "prompts/schema/teacher_inference.json",
        }
    )


@dataclass
class DataPrepConfig:
    max_pixels: int = 1024 * 1024
    max_side: int = 1024
    num_workers: int = 4


@dataclass
class GenerationConfig:
    batch_size: int = 1
    greedy: bool = False
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    out_seq_length: int = 512


@dataclass
class TrainingConfig:
    mode: str = "lora"
    experiment_name: Optional[str] = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_steps: int = -1
    num_train_epochs: int = 3
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    output_dir: str = "outputs/fine_tuned"
    max_seq_length: int = 2048


@dataclass
class HFExportConfig:
    enabled: bool = False
    repo_id: str = ""
    private: bool = True


@dataclass
class GGUFExportConfig:
    enabled: bool = False
    quantization: str = "Q4_K_M"


@dataclass
class OllamaExportConfig:
    enabled: bool = False
    model_name: str = ""


@dataclass
class ExportConfig:
    huggingface: HFExportConfig = field(default_factory=HFExportConfig)
    gguf: GGUFExportConfig = field(default_factory=GGUFExportConfig)
    ollama: OllamaExportConfig = field(default_factory=OllamaExportConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    wandb_project: str = "multimodal-receipt-ai"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))


@dataclass
class SystemConfig:
    """Global configuration for the multimodal receipt system."""

    env: str
    paths: PathConfig
    models: ModelConfig
    data: DataConfig
    data_prep: DataPrepConfig
    generation: GenerationConfig
    training: TrainingConfig
    logging: LoggingConfig
    export: ExportConfig = field(default_factory=ExportConfig)

    is_sagemaker: bool = field(default_factory=lambda: "SM_MODEL_DIR" in os.environ)

    @property
    def active_teacher(self) -> str:
        """Return the model ID for the current environment."""
        if self.env == "prod":
            return self.models.prod_teacher.name
        return self.models.dev_teacher.name

    @property
    def active_model_settings(self) -> ModelSettings:
        """Return the model settings for the current environment."""
        # For training, we might use fine_tune_base
        # But we'll keep the logic for teachers as is
        if self.env == "prod":
            return self.models.prod_teacher
        return self.models.dev_teacher

    @property
    def sources_dir(self) -> Path:
        return self.paths.sources_dir

    @property
    def artifacts_dir(self) -> Path:
        return self.paths.artifacts_dir

    @property
    def manifests_dir(self) -> Path:
        return self.paths.manifests_dir


def _merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def _create_dataclass(cls: Type[T], data: Any) -> T:
    """Recursively create dataclasses from dictionary data."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    init_data = {}

    for k, v in data.items():
        if k in field_types:
            ftype = field_types[k]
            # Handle Optional types (extract the actual type)
            if hasattr(ftype, "__origin__") and ftype.__origin__ is Optional:
                ftype = ftype.__args__[0]

            # Check if it's a dataclass
            if hasattr(ftype, "__dataclass_fields__"):
                init_data[k] = _create_dataclass(ftype, v)

            # Handle Dictionary types (Dict[str, str])
            elif hasattr(ftype, "__origin__") and ftype.__origin__ is dict:
                init_data[k] = v  # Assume a dictionary of primitives
            # Handle List types
            elif hasattr(ftype, "__origin__") and ftype.__origin__ is list:
                init_data[k] = v
            else:
                init_data[k] = v

    return cls(**init_data)


def load_config(config_file: Optional[str] = None) -> SystemConfig:
    """Load and merge configuration from a single task-specific YAML file."""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"

    # 1. Determine env and config file
    config_file = config_file or os.getenv("APP_CONFIG", "teacher_labels_generation.yaml")
    config_path = config_dir / config_file

    # Prioritize ENV as requested by user
    app_env = os.getenv("ENV", os.getenv("APP_ENV", "dev")).lower()

    if not config_path.exists():
        # Fallback to teacher_labels_generation.yaml as it's the primary task config
        config_path = config_dir / "teacher_labels_generation.yaml"
        if not config_path.exists():
            # Second fallback to default if it exists (for bootstrap)
            config_path = config_dir / "default.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # 2. Extract sections
    if "default" in full_config:
        config_dict = full_config["default"]
        if app_env in full_config:
            config_dict = _merge_dicts(config_dict, full_config[app_env])
    else:
        config_dict = full_config

    # 3. Handle Paths (Absolute conversion)
    path_data = config_dict.get("paths", {})
    for k, v in path_data.items():
        if isinstance(v, str):
            p = Path(v)
            if not p.is_absolute():
                path_data[k] = (project_root / p).resolve()

    # 4. Instantiate Dataclasses
    paths = PathConfig(project_root=project_root, **config_dict.get("paths", {}))
    models = _create_dataclass(ModelConfig, config_dict.get("models", {}))
    data_cfg = _create_dataclass(DataConfig, config_dict.get("data", {}))
    data_prep = _create_dataclass(DataPrepConfig, config_dict.get("data_prep", {}))
    generation = _create_dataclass(GenerationConfig, config_dict.get("generation", {}))
    training = _create_dataclass(TrainingConfig, config_dict.get("training", {}))
    logging = _create_dataclass(LoggingConfig, config_dict.get("logging", {}))
    export = _create_dataclass(ExportConfig, config_dict.get("export", {}))

    return SystemConfig(
        env=app_env,
        paths=paths,
        models=models,
        data=data_cfg,
        data_prep=data_prep,
        generation=generation,
        training=training,
        logging=logging,
        export=export,
    )


# Global instance
config = load_config()
