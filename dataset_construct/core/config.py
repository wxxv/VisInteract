"""
Configuration management for Vis-Interact Dataset Construction.
Configured for BIRD Mini-Dev SQLite dataset.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM API configuration for text-only tasks."""
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gemini-3-pro-preview"))
    temperature: float = 0.8
    
    # Thinking/reasoning parameters
    enable_thinking: bool = True  # Enable reasoning process for supported models
    thinking_mode: str = "auto"   # "auto", "qwen", "doubao", "custom"
    thinking_budget: int = 81920  # For Qwen models (thinking_budget)
    reasoning_effort: str = "high"  # For Doubao models: "low", "medium", "high"
    thinking_params: Optional[dict] = None  # Custom params for other models
    
    def __post_init__(self):
        if not self.api_key:
            import warnings
            warnings.warn("OPENAI_API_KEY environment variable is not set")
    
    def get_thinking_kwargs(self) -> dict:
        """Get model-specific thinking/reasoning parameters."""
        if not self.enable_thinking:
            return {}
        
        # Custom mode - use user-provided params
        if self.thinking_mode == "custom" and self.thinking_params:
            return self.thinking_params
        
        # Auto-detect based on model name
        model_lower = self.model.lower()
        
        # Qwen models - use extra_body
        if self.thinking_mode == "qwen" or (self.thinking_mode == "auto" and "qwen" in model_lower):
            return {
                "extra_body": {
                    "enable_thinking": True,
                    "thinking_budget": self.thinking_budget
                }
            }
        
        # Doubao models - use reasoning_effort
        elif self.thinking_mode == "doubao" or (self.thinking_mode == "auto" and "doubao" in model_lower):
            return {
                "reasoning_effort": self.reasoning_effort
            }
        
        # Default: no thinking params
        return {}


@dataclass
class VLMConfig:
    """VLM API configuration for vision tasks (image input)."""
    # VLM can use same API endpoint or different one
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: os.environ.get("VLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"))
    model: str = field(default_factory=lambda: os.environ.get("VLM_MODEL", "gemini-3-pro-preview"))
    temperature: float = 0.5
    
    # Image detail level for vision API
    image_detail: str = "auto"  # "auto", "low", "high" - controls image resolution/token usage
    
    # Thinking/reasoning parameters
    enable_thinking: bool = True  # Enable reasoning process for supported models
    thinking_mode: str = "auto"   # "auto", "qwen", "doubao", "custom"
    thinking_budget: int = 81920  # For Qwen models (thinking_budget)
    reasoning_effort: str = "medium"  # For Doubao models: "low", "medium", "high"
    thinking_params: Optional[dict] = None  # Custom params for other models
    
    def __post_init__(self):
        if not self.api_key:
            import warnings
            warnings.warn("VLM_API_KEY or OPENAI_API_KEY environment variable is not set")
    
    def get_thinking_kwargs(self) -> dict:
        """Get model-specific thinking/reasoning parameters."""
        if not self.enable_thinking:
            return {}
        
        # Custom mode - use user-provided params
        if self.thinking_mode == "custom" and self.thinking_params:
            return self.thinking_params
        
        # Auto-detect based on model name
        model_lower = self.model.lower()
        
        # Qwen models - use extra_body
        if self.thinking_mode == "qwen" or (self.thinking_mode == "auto" and "qwen" in model_lower):
            return {
                "extra_body": {
                    "enable_thinking": True,
                    "thinking_budget": self.thinking_budget
                }
            }
        
        # Doubao models - use reasoning_effort
        elif self.thinking_mode == "doubao" or (self.thinking_mode == "auto" and "doubao" in model_lower):
            return {
                "reasoning_effort": self.reasoning_effort
            }
        
        # Default: no thinking params
        return {}


@dataclass
class FeedbackConfig:
    """Configuration for iteration feedback loop."""
    max_sql_retries: int = 3           # Max SQL fix attempts
    max_code_retries: int = 3          # Max Altair code fix attempts
    max_total_iterations: int = 6      # Total iteration limit per sample
    enable_feedback_loop: bool = True  # Enable/disable feedback iteration
    stop_on_same_error: bool = True    # Stop if same error repeats


@dataclass
class ValidationConfig:
    """Configuration for data quality and chart validation."""
    # Data quality validation
    enable_data_quality_check: bool = True    # Enable LLM-based data quality validation
    min_data_diversity_score: float = 0.3     # Minimum data diversity score (0-1)
    
    # VLM chart validation
    enable_vlm_chart_check: bool = True       # Enable VLM-based chart readability validation
    min_chart_readability_score: float = 0.7  # Minimum chart readability score (0-1)
    
    # Retry settings
    max_validation_retries: int = 2           # Max retries when validation fails
    
    # Image settings
    chart_image_scale: float = 2.0            # Scale factor for rendered chart images
    save_validation_images: bool = True       # Save chart images during validation


@dataclass
class PipelineConfig:
    """Configuration for the sample generation pipeline."""
    num_candidates: int = 3            # Number of diverse candidates per instance
    min_data_rows: int = 5             # Minimum rows for visualization data
    max_data_rows: int = 100           # Maximum rows for visualization data
    enable_diversity_check: bool = True
    save_intermediate: bool = True     # Save intermediate results
    debug_mode: bool = field(default_factory=lambda: os.environ.get("VIS_INTERACT_DEBUG", "").lower() in ("1", "true", "yes"))  # Raise exceptions instead of catching
    
    # Parallel processing configuration
    # Instance-level parallel configuration (NEW)
    enable_instance_parallel: bool = True       # Enable instance-level parallelism
    instance_parallel_workers: int = 5          # Fixed number of parallel instances
    
    # Candidate-level parallel configuration
    enable_parallel: bool = True                # Enable parallel candidate processing
    initial_parallel_workers: int = 3           # Initial number of parallel workers (reduced from 5 to avoid burst)
    min_parallel_workers: int = 3               # Minimum workers (when rate limited)
    max_parallel_workers: int = 6               # Maximum workers (reduced from 10 for conservative approach)
    concurrency_recovery_threshold: int = 20    # Consecutive successes before increasing workers
    
    # Chart example execution configuration
    execute_chart_examples: bool = True         # Enable/disable chart example execution for multimodal input
    chart_execution_timeout: float = 5.0        # Timeout per chart execution (seconds)
    max_example_images: int = 5                 # Maximum number of example images to generate
    example_image_scale: float = 1.0            # Scale factor for example chart images


@dataclass
class PathConfig:
    """Path configuration for all resources."""
    # Base paths
    project_root: Path
    
    # BIRD Mini-Dev paths
    @property
    def minidev_dir(self) -> Path:
        return self.project_root / "minidev" / "MINIDEV"
    
    @property
    def minidev_json(self) -> Path:
        return self.minidev_dir / "mini_dev_sqlite.json"
    
    @property
    def dev_tables_json(self) -> Path:
        return self.minidev_dir / "dev_tables.json"
    
    @property
    def dev_databases_dir(self) -> Path:
        """Legacy path - kept for backward compatibility with schema loading."""
        return self.minidev_dir / "dev_databases"
    
    @property
    def local_databases_dir(self) -> Path:
        """Local databases directory - flat structure with {db_id}.sqlite files."""
        return self.project_root / "dataset_construct" / "databases"
    
    def get_relative_db_path(self, db_id: str) -> str:
        """Get relative database path for LLM prompts."""
        return f"./databases/{db_id}.sqlite"
    
    # Chart example resources (chart contracts from Vega-Altair official docs)
    @property
    def chart_example_dir(self) -> Path:
        return self.project_root / "chart_example"
    
    # Output paths
    @property
    def output_dir(self) -> Path:
        return self.project_root / "dataset_construct" / "output"
    
    @property
    def cache_dir(self) -> Path:
        return self.project_root / "dataset_construct" / "cache"
    
    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"
    
    @property
    def temp_images_dir(self) -> Path:
        """Temporary directory for validation chart images."""
        return self.output_dir / "temp_images"


class Config:
    """Main configuration class."""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Default: parent of dataset_construct directory
            # Note: __file__ is now in core/config.py, so we need to go up 2 levels
            project_root = Path(__file__).resolve().parent.parent.parent
        
        self.paths = PathConfig(project_root=project_root)
        self.llm = LLMConfig()
        self.vlm = VLMConfig()
        self.feedback = FeedbackConfig()
        self.pipeline = PipelineConfig()
        self.validation = ValidationConfig()
    
    def ensure_dirs(self):
        """Create output and cache directories if they don't exist."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.temp_images_dir.mkdir(parents=True, exist_ok=True)
        self.paths.local_databases_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate that required paths exist."""
        required_paths = [
            self.paths.minidev_json,
            self.paths.dev_tables_json,
            self.paths.dev_databases_dir,
        ]
        
        missing = [p for p in required_paths if not p.exists()]
        if missing:
            import warnings
            for p in missing:
                warnings.warn(f"Required path does not exist: {p}")
            return False
        return True


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(project_root: Optional[Path] = None) -> Config:
    """Initialize global config with custom project root."""
    global _config
    _config = Config(project_root=project_root)
    return _config
