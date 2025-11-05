# Universal Memory Layer CLI

The Universal Memory Layer CLI provides an interactive command-line interface for chatting with different AI models while maintaining conversation history and context across model switches.

## Features

- **Multi-Model Support**: Chat with ChatGPT, Claude, and Gemini
- **Persistent Memory**: Conversation history is stored and maintained across sessions
- **Model Switching**: Seamlessly switch between different AI models during conversation
- **Context Retrieval**: Intelligent context retrieval using recent or semantic search
- **Interactive Commands**: Rich set of commands for managing conversations and memory

## Installation

The CLI is part of the Universal Memory Layer package. Ensure you have the package installed:

```bash
pip install -e .
```

## Configuration

Set up API keys for the models you want to use:

```bash
# For OpenAI models (ChatGPT, GPT-4)
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-google-api-key"
```

Optional configuration:

```bash
# Database path (default: memory.db)
export UML_DATABASE_PATH="/path/to/memory.db"

# Embedding provider (default: openai)
export UML_EMBEDDING_PROVIDER="openai"  # or "huggingface"

# Logging level (default: WARNING)
export UML_LOG_LEVEL="INFO"
```

## Usage

### Starting the CLI

```bash
# Start with model selection
python -m universal_memory_layer.cli.main

# Start with a specific model
python -m universal_memory_layer.cli.main --model chatgpt

# Show help
python -m universal_memory_layer.cli.main --help
```

### Interactive Commands

Once in the CLI, you can use these commands:

#### Chat Commands
- `<message>` - Send a message to the current model
- Example: `Hello, how are you?`

#### Model Commands
- `/switch` - Switch to a different model
- `/models` - Show available models
- `/current` - Show current model

#### Memory Commands
- `/recent [n]` - Show last n messages (default: 5)
- `/search <query>` - Search for relevant messages
- `/stats` - Show memory statistics
- `/clear` - Clear all memory (requires confirmation)

#### Utility Commands
- `/help` - Show help message
- `/quit` or `/exit` - Exit the CLI

### Example Session

```
🌟 Welcome to Universal Memory Layer CLI!

📋 Available Models:
========================================
  1. chatgpt
  2. gpt-4
  3. claude
========================================

Select a model (1-3) or 'q' to quit: 1
✓ Selected model: chatgpt

🚀 Started conversation with chatgpt
📊 Memory contains 0 messages

============================================================
💬 Chat started! Type your message or '/help' for commands.
============================================================

[chatgpt] You: Hello! Can you help me understand machine learning?

🤖 chatgpt is thinking...

🤖 chatgpt:
----------------------------------------
Hello! I'd be happy to help you understand machine learning...
----------------------------------------

[chatgpt] You: /switch

Current model: chatgpt

📋 Available Models:
========================================
  1. chatgpt
  2. gpt-4
  3. claude
========================================

Select a model (1-3) or 'q' to quit: 3
✓ Selected model: claude
✅ Switched to claude

[claude] You: Can you continue explaining from where we left off?

🤖 claude is thinking...

🤖 claude:
----------------------------------------
Based on our previous conversation about machine learning...
----------------------------------------

[claude] You: /recent 3

📜 Last 3 messages:
==================================================
👤 [12:34:56] User 
   Hello! Can you help me understand machine learning?

🤖 [12:35:02] Assistant (chatgpt)
   Hello! I'd be happy to help you understand machine learning...

👤 [12:35:45] User 
   Can you continue explaining from where we left off?
==================================================

[claude] You: /quit
```

## Architecture

The CLI consists of several key components:

- **UniversalMemoryCLI**: Main CLI class handling user interaction
- **ConversationManager**: Orchestrates LLM clients and memory storage
- **MemoryStore**: Handles persistent storage and retrieval
- **LLM Clients**: Interfaces for different AI models

## Error Handling

The CLI handles various error conditions gracefully:

- **Missing API Keys**: Shows helpful configuration messages
- **Network Issues**: Retries with exponential backoff
- **Model Unavailability**: Falls back to available models
- **Memory Errors**: Continues operation with degraded functionality

## Troubleshooting

### No Models Available

If you see "No models available", ensure you have set the appropriate API keys:

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Memory Issues

If memory operations fail, check database permissions and disk space:

```bash
ls -la memory.db
df -h .
```

## Development

### Running Tests

```bash
# Run CLI tests
python -m pytest tests/cli/ -v

# Run all tests
python -m pytest -v
```

### Adding New Commands

To add new CLI commands:

1. Add the command handler to `process_command()` method
2. Update the help text in `display_help()` method
3. Add tests in `tests/cli/test_cli_main.py`

### Adding New Models

To add support for new AI models:

1. Create a new client class inheriting from `BaseLLMClient`
2. Add the client to `ConversationManager._initialize_clients()`
3. Update the CLI help and documentation