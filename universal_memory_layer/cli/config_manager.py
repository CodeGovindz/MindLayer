"""CLI commands for configuration management."""

import click
import sys
from pathlib import Path
from typing import Optional

from ..config_loader import ConfigLoader, ConfigurationError, load_config, create_sample_config
from ..logging_config import setup_logging, get_logger
from ..errors import create_user_friendly_message

logger = get_logger(__name__)


@click.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.option('--path', '-p', type=click.Path(), help='Path to configuration file')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml', 
              help='Configuration file format')
def create(path: Optional[str], format: str):
    """Create a sample configuration file."""
    try:
        if not path:
            path = f"uml_config.{format}"
        
        create_sample_config(path, format)
        click.echo(f"✅ Sample configuration created at: {path}")
        click.echo(f"📝 Edit the file to customize your settings")
        click.echo(f"🔑 Don't forget to set your API keys!")
        
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error creating configuration: {error_msg}", err=True)
        sys.exit(1)


@config.command()
@click.option('--path', '-p', type=click.Path(exists=True), help='Path to configuration file')
def validate(path: Optional[str]):
    """Validate a configuration file."""
    try:
        config = load_config(path)
        click.echo("✅ Configuration is valid!")
        
        # Show configuration summary
        click.echo("\n📋 Configuration Summary:")
        click.echo(f"  Database: {config.database_path}")
        click.echo(f"  Vector Store: {config.vector_store_path}")
        click.echo(f"  Embedding Provider: {config.embedding_provider}")
        click.echo(f"  Embedding Model: {config.embedding_model}")
        click.echo(f"  Max Context Length: {config.max_context_length}")
        
        # Check API keys
        loader = ConfigLoader(path)
        missing_keys = loader.get_missing_api_keys(config)
        
        if missing_keys:
            click.echo(f"\n⚠️  Missing API keys for: {', '.join(missing_keys)}")
            click.echo("   Set these environment variables or add them to your config file:")
            for provider in missing_keys:
                env_var = {
                    'openai': 'OPENAI_API_KEY',
                    'anthropic': 'ANTHROPIC_API_KEY', 
                    'google': 'GOOGLE_API_KEY'
                }.get(provider, f'{provider.upper()}_API_KEY')
                click.echo(f"   - {env_var}")
        else:
            click.echo("\n🔑 All API keys are configured!")
            
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Configuration validation failed: {error_msg}", err=True)
        sys.exit(1)


@config.command()
@click.option('--path', '-p', type=click.Path(exists=True), help='Path to configuration file')
def show(path: Optional[str]):
    """Show current configuration (excluding sensitive data)."""
    try:
        config = load_config(path)
        
        click.echo("📋 Current Configuration:")
        config_dict = config.to_dict()
        
        for key, value in config_dict.items():
            # Format key for display
            display_key = key.replace('_', ' ').title()
            click.echo(f"  {display_key}: {value}")
            
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error loading configuration: {error_msg}", err=True)
        sys.exit(1)


@config.command()
@click.option('--level', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.option('--file', '-f', type=click.Path(), help='Log file path')
@click.option('--console/--no-console', default=True, help='Enable/disable console logging')
def setup_logging_cmd(level: str, file: Optional[str], console: bool):
    """Set up logging configuration."""
    try:
        setup_logging(
            log_level=level,
            log_file=file,
            console_output=console,
            file_output=file is not None
        )
        
        click.echo(f"✅ Logging configured:")
        click.echo(f"  Level: {level}")
        if console:
            click.echo(f"  Console: Enabled")
        if file:
            click.echo(f"  File: {file}")
            
        # Test the logging
        logger.info("Logging configuration test - this is an info message")
        logger.debug("This is a debug message (only visible if level is DEBUG)")
        
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error setting up logging: {error_msg}", err=True)
        sys.exit(1)


@config.command()
def check_env():
    """Check environment variables for API keys."""
    import os
    
    click.echo("🔍 Checking environment variables:")
    
    api_keys = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Google': 'GOOGLE_API_KEY'
    }
    
    found_keys = []
    missing_keys = []
    
    for provider, env_var in api_keys.items():
        value = os.getenv(env_var)
        if value:
            # Show partial key for security
            masked_key = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
            click.echo(f"  ✅ {provider}: {masked_key}")
            found_keys.append(provider)
        else:
            click.echo(f"  ❌ {provider}: Not set ({env_var})")
            missing_keys.append(provider)
    
    if found_keys:
        click.echo(f"\n🔑 Found API keys for: {', '.join(found_keys)}")
    
    if missing_keys:
        click.echo(f"\n⚠️  Missing API keys for: {', '.join(missing_keys)}")
        click.echo("   Set these environment variables to enable the corresponding providers.")
    else:
        click.echo("\n🎉 All API keys are configured!")


if __name__ == '__main__':
    config()