#!/usr/bin/env python3
"""Example usage of the Universal Memory Layer CLI."""

import os
import sys

# Add the project root to Python path for examples
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_memory_layer.cli.main import UniversalMemoryCLI
from universal_memory_layer.models.config import MemoryConfig

def example_cli_usage():
    """Demonstrate CLI usage programmatically."""
    
    print("Universal Memory Layer CLI Example")
    print("=" * 40)
    
    # Create CLI with custom config
    config = MemoryConfig()
    cli = UniversalMemoryCLI(config)
    
    try:
        # Initialize the CLI
        print("1. Initializing CLI...")
        cli.initialize()
        print("   ✓ CLI initialized successfully")
        
        # Check available models
        print("\n2. Checking available models...")
        models = cli.get_available_models()
        if models:
            print(f"   Available models: {', '.join(models)}")
        else:
            print("   No models available (API keys not configured)")
            print("   To use the CLI, set one or more of these environment variables:")
            print("   - OPENAI_API_KEY for ChatGPT/GPT-4")
            print("   - ANTHROPIC_API_KEY for Claude")
            print("   - GOOGLE_API_KEY for Gemini")
        
        # Display help
        print("\n3. Available CLI commands:")
        cli.display_help()
        
        print("\n4. CLI is ready for interactive use!")
        print("   Run: python -m universal_memory_layer.cli.main")
        
    except Exception as e:
        if "API key not found" in str(e):
            print("   ⚠️  CLI structure is working, but no API keys are configured")
            print("   To use the CLI, set one or more of these environment variables:")
            print("   - OPENAI_API_KEY for ChatGPT/GPT-4")
            print("   - ANTHROPIC_API_KEY for Claude")
            print("   - GOOGLE_API_KEY for Gemini")
            print("\n   CLI help and structure:")
            cli.display_help()
            return True  # This is expected behavior
        else:
            print(f"   ❌ Unexpected error: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = example_cli_usage()
    sys.exit(0 if success else 1)