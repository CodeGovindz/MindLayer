"""Configuration data model with default values."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_int_from_env(env_var: str, default: str) -> int:
    """Safely convert environment variable to int with proper error handling."""
    value = os.getenv(env_var, default)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{env_var} must be a valid integer")


@dataclass
class MemoryConfig:
    """Configuration settings for the Universal Memory Layer."""
    
    # Database settings
    database_path: str = "memory.db"
    vector_store_path: str = "vector_store.faiss"
    
    # Embedding settings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    
    # Context settings
    max_context_length: int = 4000
    default_recent_count: int = 3
    default_relevant_count: int = 5
    
    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    
    # Optional Hugging Face settings
    huggingface_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_cache_dir: Optional[str] = None
    
    # Control whether to load from environment variables
    _load_from_env: bool = field(default=True, init=False)
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Load from environment variables if no explicit values were provided
        if self._load_from_env:
            self._load_from_environment()
        self._process_paths()
        self.validate()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database settings
        if os.getenv("UML_DATABASE_PATH"):
            self.database_path = os.getenv("UML_DATABASE_PATH")
        if os.getenv("UML_VECTOR_STORE_PATH"):
            self.vector_store_path = os.getenv("UML_VECTOR_STORE_PATH")
        
        # Embedding settings
        if os.getenv("UML_EMBEDDING_PROVIDER"):
            self.embedding_provider = os.getenv("UML_EMBEDDING_PROVIDER")
        if os.getenv("UML_EMBEDDING_MODEL"):
            self.embedding_model = os.getenv("UML_EMBEDDING_MODEL")
        
        # Context settings
        if os.getenv("UML_MAX_CONTEXT_LENGTH"):
            self.max_context_length = _safe_int_from_env("UML_MAX_CONTEXT_LENGTH", str(self.max_context_length))
        if os.getenv("UML_DEFAULT_RECENT_COUNT"):
            self.default_recent_count = _safe_int_from_env("UML_DEFAULT_RECENT_COUNT", str(self.default_recent_count))
        if os.getenv("UML_DEFAULT_RELEVANT_COUNT"):
            self.default_relevant_count = _safe_int_from_env("UML_DEFAULT_RELEVANT_COUNT", str(self.default_relevant_count))
        
        # API Keys
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("GOOGLE_API_KEY"):
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if os.getenv("GROQ_API_KEY"):
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Hugging Face settings
        if os.getenv("UML_HF_MODEL_PATH"):
            self.huggingface_model_path = os.getenv("UML_HF_MODEL_PATH")
        if os.getenv("UML_HF_CACHE_DIR"):
            self.huggingface_cache_dir = os.getenv("UML_HF_CACHE_DIR")
    
    def _load_environment_overrides(self) -> None:
        """Load any additional environment variable overrides."""
        # Handle integer environment variables with error checking
        try:
            if os.getenv("UML_MAX_CONTEXT_LENGTH"):
                self.max_context_length = int(os.getenv("UML_MAX_CONTEXT_LENGTH"))
        except ValueError:
            raise ValueError("UML_MAX_CONTEXT_LENGTH must be a valid integer")
        
        try:
            if os.getenv("UML_DEFAULT_RECENT_COUNT"):
                self.default_recent_count = int(os.getenv("UML_DEFAULT_RECENT_COUNT"))
        except ValueError:
            raise ValueError("UML_DEFAULT_RECENT_COUNT must be a valid integer")
        
        try:
            if os.getenv("UML_DEFAULT_RELEVANT_COUNT"):
                self.default_relevant_count = int(os.getenv("UML_DEFAULT_RELEVANT_COUNT"))
        except ValueError:
            raise ValueError("UML_DEFAULT_RELEVANT_COUNT must be a valid integer")
    
    def _process_paths(self) -> None:
        """Process and validate file paths."""
        # Ensure database and vector store paths are absolute
        self.database_path = str(Path(self.database_path).resolve())
        self.vector_store_path = str(Path(self.vector_store_path).resolve())
        
        # Create parent directories if they don't exist
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).parent.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        self._validate_embedding_provider()
        self._validate_embedding_model()
        self._validate_context_settings()
        self._validate_paths()
    
    def _validate_embedding_provider(self) -> None:
        """Validate embedding provider setting."""
        valid_providers = ["openai", "huggingface"]
        if self.embedding_provider not in valid_providers:
            raise ValueError(f"Invalid embedding provider: {self.embedding_provider}. Must be one of: {valid_providers}")
    
    def _validate_embedding_model(self) -> None:
        """Validate embedding model setting."""
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("Embedding model must be a non-empty string")
    
    def _validate_context_settings(self) -> None:
        """Validate context-related settings."""
        if not isinstance(self.max_context_length, int) or self.max_context_length <= 0:
            raise ValueError("max_context_length must be a positive integer")
        
        if not isinstance(self.default_recent_count, int) or self.default_recent_count <= 0:
            raise ValueError("default_recent_count must be a positive integer")
        
        if not isinstance(self.default_relevant_count, int) or self.default_relevant_count <= 0:
            raise ValueError("default_relevant_count must be a positive integer")
        
        if self.max_context_length > 1000000:  # 1M character limit
            raise ValueError("max_context_length cannot exceed 1,000,000 characters")
    
    def _validate_paths(self) -> None:
        """Validate file paths."""
        # Get the original environment values to check for empty strings
        db_env_value = os.getenv("UML_DATABASE_PATH")
        vs_env_value = os.getenv("UML_VECTOR_STORE_PATH")
        
        # Check if environment variable was explicitly set to empty string
        if db_env_value is not None and not db_env_value.strip():
            raise ValueError("Database path must be a non-empty string")
        
        if vs_env_value is not None and not vs_env_value.strip():
            raise ValueError("Vector store path must be a non-empty string")
        
        # Standard validation for the processed paths
        if not isinstance(self.database_path, str) or not self.database_path.strip():
            raise ValueError("Database path must be a non-empty string")
        
        if not isinstance(self.vector_store_path, str) or not self.vector_store_path.strip():
            raise ValueError("Vector store path must be a non-empty string")
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for the given provider."""
        provider_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key
        }
        
        if provider not in provider_keys:
            raise ValueError(f"Unknown provider: {provider}")
        
        return bool(provider_keys[provider].strip())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], load_from_env: bool = False) -> 'MemoryConfig':
        """Create MemoryConfig from dictionary without environment variable loading.
        
        Args:
            config_dict: Dictionary containing configuration values.
            load_from_env: Whether to load from environment variables.
            
        Returns:
            MemoryConfig instance.
        """
        # Filter out any keys that aren't valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != '_load_from_env'}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Create instance
        instance = cls(**filtered_dict)
        instance._load_from_env = load_from_env
        
        # Re-run post_init with the correct environment loading setting
        if not load_from_env:
            instance._process_paths()
            instance.validate()
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive API keys)."""
        return {
            'database_path': self.database_path,
            'vector_store_path': self.vector_store_path,
            'embedding_provider': self.embedding_provider,
            'embedding_model': self.embedding_model,
            'max_context_length': self.max_context_length,
            'default_recent_count': self.default_recent_count,
            'default_relevant_count': self.default_relevant_count,
            'huggingface_model_path': self.huggingface_model_path,
            'huggingface_cache_dir': self.huggingface_cache_dir,
            # API keys are excluded for security
        }