"""Configuration loading and management system."""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .models.config import MemoryConfig


class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or validation."""
    pass


class ConfigLoader:
    """Handles loading configuration from files and environment variables."""
    
    DEFAULT_CONFIG_PATHS = [
        "uml_config.yaml",
        "uml_config.yml", 
        "uml_config.json",
        ".uml_config.yaml",
        ".uml_config.yml",
        ".uml_config.json",
        os.path.expanduser("~/.config/universal_memory_layer/config.yaml"),
        os.path.expanduser("~/.config/universal_memory_layer/config.yml"),
        os.path.expanduser("~/.config/universal_memory_layer/config.json"),
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Optional path to configuration file. If not provided,
                        will search for config files in default locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_data: Dict[str, Any] = {}
    
    def load_config(self) -> MemoryConfig:
        """Load configuration from file and environment variables.
        
        Returns:
            MemoryConfig instance with loaded configuration.
            
        Raises:
            ConfigurationError: If configuration loading fails.
        """
        try:
            # Load from file if available
            config_file_path = self._find_config_file()
            if config_file_path:
                self._config_data = self._load_config_file(config_file_path)
            
            # Create config with file data and environment overrides
            return self._create_memory_config()
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file to load.
        
        Returns:
            Path to configuration file or None if not found.
        """
        if self.config_path:
            if self.config_path.exists():
                return self.config_path
            else:
                raise ConfigurationError(f"Specified config file not found: {self.config_path}")
        
        # Search default locations
        for config_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(config_path)
            if path.exists():
                return path
        
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            Dictionary containing configuration data.
            
        Raises:
            ConfigurationError: If file loading fails.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    return json.load(f)
                elif config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
        
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file {config_path}: {str(e)}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file {config_path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file {config_path}: {str(e)}")
    
    def _create_memory_config(self) -> MemoryConfig:
        """Create MemoryConfig instance with loaded data and environment overrides.
        
        Returns:
            MemoryConfig instance.
        """
        # Start with file configuration data
        config_kwargs = {}
        
        # Map file config keys to MemoryConfig fields and their environment variables
        field_mapping = {
            'database_path': ('database_path', 'UML_DATABASE_PATH'),
            'vector_store_path': ('vector_store_path', 'UML_VECTOR_STORE_PATH'),
            'embedding_provider': ('embedding_provider', 'UML_EMBEDDING_PROVIDER'),
            'embedding_model': ('embedding_model', 'UML_EMBEDDING_MODEL'),
            'max_context_length': ('max_context_length', 'UML_MAX_CONTEXT_LENGTH'),
            'default_recent_count': ('default_recent_count', 'UML_DEFAULT_RECENT_COUNT'),
            'default_relevant_count': ('default_relevant_count', 'UML_DEFAULT_RELEVANT_COUNT'),
            'huggingface_model_path': ('huggingface_model_path', 'UML_HF_MODEL_PATH'),
            'huggingface_cache_dir': ('huggingface_cache_dir', 'UML_HF_CACHE_DIR'),
            'openai_api_key': ('openai_api_key', 'OPENAI_API_KEY'),
            'anthropic_api_key': ('anthropic_api_key', 'ANTHROPIC_API_KEY'),
            'google_api_key': ('google_api_key', 'GOOGLE_API_KEY'),
        }
        
        # Apply configuration with environment variable precedence
        for file_key, (config_field, env_var) in field_mapping.items():
            # Check environment variable first (highest precedence)
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Handle type conversion for integer fields
                if config_field in ['max_context_length', 'default_recent_count', 'default_relevant_count']:
                    try:
                        config_kwargs[config_field] = int(env_value)
                    except ValueError:
                        raise ConfigurationError(f"{env_var} must be a valid integer")
                else:
                    config_kwargs[config_field] = env_value
            # Otherwise use file configuration if available
            elif file_key in self._config_data:
                config_kwargs[config_field] = self._config_data[file_key]
        
        # Create config using from_dict to handle environment loading properly
        return MemoryConfig.from_dict(config_kwargs, load_from_env=True)
    
    def create_sample_config(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """Create a sample configuration file.
        
        Args:
            output_path: Path where to create the sample config file.
            format: Configuration file format ('yaml' or 'json').
            
        Raises:
            ConfigurationError: If sample config creation fails.
        """
        if format not in ['yaml', 'json']:
            raise ConfigurationError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")
        
        sample_config = {
            'database_path': 'memory.db',
            'vector_store_path': 'vector_store.faiss',
            'embedding_provider': 'openai',
            'embedding_model': 'text-embedding-ada-002',
            'max_context_length': 4000,
            'default_recent_count': 3,
            'default_relevant_count': 5,
            'huggingface_model_path': 'sentence-transformers/all-MiniLM-L6-v2',
            'huggingface_cache_dir': None,
            'openai_api_key': '${OPENAI_API_KEY}',
            'anthropic_api_key': '${ANTHROPIC_API_KEY}',
            'google_api_key': '${GOOGLE_API_KEY}',
        }
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(sample_config, f, indent=2)
                else:  # yaml
                    yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create sample config: {str(e)}")
    
    def validate_api_keys(self, config: MemoryConfig) -> Dict[str, bool]:
        """Validate API keys for different providers.
        
        Args:
            config: MemoryConfig instance to validate.
            
        Returns:
            Dictionary mapping provider names to validation status.
        """
        providers = ['openai', 'anthropic', 'google']
        validation_results = {}
        
        for provider in providers:
            try:
                validation_results[provider] = config.has_api_key(provider)
            except ValueError:
                validation_results[provider] = False
        
        return validation_results
    
    def get_missing_api_keys(self, config: MemoryConfig) -> list[str]:
        """Get list of providers with missing API keys.
        
        Args:
            config: MemoryConfig instance to check.
            
        Returns:
            List of provider names with missing API keys.
        """
        validation_results = self.validate_api_keys(config)
        return [provider for provider, has_key in validation_results.items() if not has_key]


def load_config(config_path: Optional[Union[str, Path]] = None) -> MemoryConfig:
    """Convenience function to load configuration.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        MemoryConfig instance with loaded configuration.
        
    Raises:
        ConfigurationError: If configuration loading fails.
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()


def create_sample_config(output_path: Union[str, Path], format: str = 'yaml') -> None:
    """Convenience function to create a sample configuration file.
    
    Args:
        output_path: Path where to create the sample config file.
        format: Configuration file format ('yaml' or 'json').
        
    Raises:
        ConfigurationError: If sample config creation fails.
    """
    loader = ConfigLoader()
    loader.create_sample_config(output_path, format)