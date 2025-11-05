"""Main CLI entry point for Universal Memory Layer."""

import click
import sys
from typing import Optional

from .main import UniversalMemoryCLI
from .config_manager import config
from ..config_loader import load_config
from ..logging_config import setup_logging
from ..errors import create_user_friendly_message


@click.group()
@click.option('--config-file', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--log-level', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.option('--quiet', '-q', is_flag=True, help='Suppress console output')
@click.pass_context
def cli(ctx, config_file: Optional[str], log_level: str, quiet: bool):
    """Universal Memory Layer - Persistent conversation memory across LLMs."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    try:
        setup_logging(
            log_level=log_level,
            console_output=not quiet,
            file_output=True
        )
    except Exception as e:
        if not quiet:
            click.echo(f"Warning: Failed to set up logging: {e}", err=True)
    
    # Load configuration
    try:
        config = load_config(config_file)
        ctx.obj['config'] = config
        ctx.obj['config_file'] = config_file
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error loading configuration: {error_msg}", err=True)
        if not config_file:
            click.echo("💡 Try creating a config file with: uml config create", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', help='LLM model to use')
@click.pass_context
def chat(ctx, model: Optional[str]):
    """Start interactive chat session."""
    try:
        config = ctx.obj['config']
        cli_instance = UniversalMemoryCLI(config)
        
        if model:
            try:
                cli_instance.conversation_manager.switch_model(model)
            except Exception as e:
                click.echo(f"❌ Error switching to model '{model}': {e}", err=True)
                click.echo("Available models:")
                for available_model in cli_instance.conversation_manager.get_available_models():
                    click.echo(f"  - {available_model}")
                sys.exit(1)
        
        cli_instance.run()
        
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error starting chat: {error_msg}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def models(ctx):
    """List available LLM models."""
    try:
        config = ctx.obj['config']
        cli_instance = UniversalMemoryCLI(config)
        
        available_models = cli_instance.conversation_manager.get_available_models()
        current_model = cli_instance.conversation_manager.get_current_model()
        
        if not available_models:
            click.echo("❌ No models available. Please check your API key configuration.")
            click.echo("💡 Use 'uml config check-env' to verify your API keys.")
            sys.exit(1)
        
        click.echo("📋 Available Models:")
        for model in available_models:
            marker = " (current)" if model == current_model else ""
            click.echo(f"  - {model}{marker}")
            
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error listing models: {error_msg}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('message')
@click.option('--model', '-m', help='LLM model to use')
@click.option('--no-memory', is_flag=True, help='Disable memory/context retrieval')
@click.option('--context-strategy', type=click.Choice(['recent', 'relevant', 'hybrid']), 
              default='recent', help='Context retrieval strategy')
@click.pass_context
def ask(ctx, message: str, model: Optional[str], no_memory: bool, context_strategy: str):
    """Ask a single question without starting interactive session."""
    try:
        config = ctx.obj['config']
        cli_instance = UniversalMemoryCLI(config)
        
        if model:
            cli_instance.conversation_manager.switch_model(model)
        
        response = cli_instance.conversation_manager.chat(
            message=message,
            use_memory=not no_memory,
            context_strategy=context_strategy
        )
        
        click.echo(response)
        
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error processing question: {error_msg}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear_memory(ctx, confirm: bool):
    """Clear all stored conversation memory."""
    try:
        if not confirm:
            if not click.confirm("⚠️  This will delete all stored conversations. Continue?"):
                click.echo("Operation cancelled.")
                return
        
        config = ctx.obj['config']
        cli_instance = UniversalMemoryCLI(config)
        
        # Clear memory through the conversation manager
        cli_instance.conversation_manager.clear_memory()
        
        click.echo("✅ Memory cleared successfully.")
        
    except Exception as e:
        error_msg = create_user_friendly_message(e)
        click.echo(f"❌ Error clearing memory: {error_msg}", err=True)
        sys.exit(1)


# Add the config subcommand group
cli.add_command(config)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()