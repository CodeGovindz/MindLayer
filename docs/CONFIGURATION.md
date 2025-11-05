# Universal Memory Layer - Configuration Reference

This document provides comprehensive information about configuring the Universal Memory Layer.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [API Keys Setup](#api-keys-setup)
- [Storage Configuration](#storage-configuration)
- [Embedding Configuration](#embedding-configuration)
- [Performance Tuning](#performance-tuning)
- [Security Considerations](#security-considerations)
- [Configuration Examples](#configuration-examples)

## Configuration Overview

The Universal Memory Layer uses a hierarchical configuration system:

1. **Default values** - Built-in defaults for all settings
2. **Environment variables** - Override defaults with environment variables
3. **Configuration files** - Load settings from `.env` files
4. **Programmatic configuration** - Set values directly in code

### Configuration Priority

Settings are applied in this order (later overrides earlier):
1. Default values in `MemoryConfig`
2. Environment variables
3. Values passed to `MemoryConfig()` constructor

## Environment Variables

### API Keys

| Variable | Description | Required | Format |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for ChatGPT/GPT-4 | Optional* | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | Optional* | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google API key for Gemini | Optional* | `AIza...` |

*At least one API key is required for the system to function.

### Storage Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `UML_DATABASE_PATH` | Path to SQLite database file | `memory.db` | `./data/memory.db` |
| `UML_VECTOR_STORE_PATH` | Path to FAISS vector index | `vector_store.faiss` | `./data/vectors.faiss` |

### Embedding Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `UML_EMBEDDING_PROVIDER` | Embedding service provider | `openai` | `openai`, `huggingface` |
| `UML_EMBEDDING_MODEL` | Model name for embeddings | `text-embedding-ada-002` | See [models](#embedding-models) |
| `UML_HF_MODEL_PATH` | Hugging Face model path | `sentence-transformers/all-MiniLM-L6-v2` | Any HF model |
| `UML_HF_CACHE_DIR` | Hugging Face cache directory | `None` | `./hf_cache` |

### Context and Performance Settings

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `UML_MAX_CONTEXT_LENGTH` | Maximum context length (characters) | `4000` | `100-1000000` |
| `UML_DEFAULT_RECENT_COUNT` | Default number of recent messages | `3` | `1-100` |
| `UML_DEFAULT_RELEVANT_COUNT` | Default number of relevant messages | `5` | `1-100` |

### Logging Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `UML_LOG_LEVEL` | Logging verbosity level | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Configuration Files

### .env File

Create a `.env` file in your project directory:

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=AIza-your-google-key-here

# Storage Configuration
UML_DATABASE_PATH=./data/memory.db
UML_VECTOR_STORE_PATH=./data/vector_store.faiss

# Embedding Configuration
UML_EMBEDDING_PROVIDER=openai
UML_EMBEDDING_MODEL=text-embedding-ada-002

# Performance Settings
UML_MAX_CONTEXT_LENGTH=4000
UML_DEFAULT_RECENT_COUNT=3
UML_DEFAULT_RELEVANT_COUNT=5

# Logging
UML_LOG_LEVEL=INFO
```

### Loading .env Files

The system automatically loads `.env` files using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

from universal_memory_layer.conversation_manager import ConversationManager
manager = ConversationManager()  # Will use .env values
```

## API Keys Setup

### OpenAI API Key

1. **Get API Key:**
   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Set Environment Variable:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

3. **Verify Setup:**
```python
from universal_memory_layer.models.config import MemoryConfig
config = MemoryConfig()
print(f"OpenAI configured: {config.has_api_key('openai')}")
```

### Anthropic API Key

1. **Get API Key:**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Generate an API key
   - Copy the key (starts with `sk-ant-`)

2. **Set Environment Variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### Google API Key

1. **Get API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key
   - Copy the key (starts with `AIza`)

2. **Set Environment Variable:**
```bash
export GOOGLE_API_KEY="AIza-your-key-here"
```

### API Key Security

**Best Practices:**
- Never commit API keys to version control
- Use environment variables or secure key management
- Rotate keys regularly
- Monitor usage and set spending limits

**Secure Storage Options:**
```bash
# Option 1: .env file (add to .gitignore)
echo "OPENAI_API_KEY=sk-..." >> .env
echo ".env" >> .gitignore

# Option 2: System environment
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc

# Option 3: Secure key management (production)
# Use AWS Secrets Manager, Azure Key Vault, etc.
```

## Storage Configuration

### Database Configuration

The system uses SQLite for persistent storage:

```bash
# Default location
UML_DATABASE_PATH=memory.db

# Custom location
UML_DATABASE_PATH=/path/to/custom/memory.db

# Relative path
UML_DATABASE_PATH=./data/conversations.db
```

**Database Features:**
- Automatic schema creation
- Migration support
- Backup capabilities
- Transaction safety

### Vector Store Configuration

FAISS is used for vector similarity search:

```bash
# Default location
UML_VECTOR_STORE_PATH=vector_store.faiss

# Custom location
UML_VECTOR_STORE_PATH=/path/to/vectors.faiss

# Multiple files (FAISS creates .faiss and .pkl files)
UML_VECTOR_STORE_PATH=./data/embeddings
```

**Vector Store Features:**
- Automatic index creation
- Incremental updates
- Index optimization
- Persistence across sessions

### Storage Best Practices

1. **Use absolute paths in production:**
```python
import os
from universal_memory_layer.models.config import MemoryConfig

config = MemoryConfig(
    database_path=os.path.abspath("./data/memory.db"),
    vector_store_path=os.path.abspath("./data/vectors.faiss")
)
```

2. **Ensure directory permissions:**
```bash
mkdir -p ./data
chmod 755 ./data
```

3. **Regular backups:**
```python
from universal_memory_layer.storage.memory_store import MemoryStore

memory = MemoryStore()
memory.initialize()
memory.backup_database("./backups/memory_backup.db")
```

## Embedding Configuration

### Embedding Providers

#### OpenAI Embeddings

**Configuration:**
```bash
UML_EMBEDDING_PROVIDER=openai
UML_EMBEDDING_MODEL=text-embedding-ada-002
```

**Available Models:**
- `text-embedding-ada-002` (1536 dimensions, recommended)
- `text-embedding-3-small` (1536 dimensions, newer)
- `text-embedding-3-large` (3072 dimensions, highest quality)

**Pros:**
- High quality embeddings
- Fast API responses
- Consistent availability

**Cons:**
- Requires API calls (cost and latency)
- Needs internet connection

#### Hugging Face Embeddings

**Configuration:**
```bash
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
UML_HF_CACHE_DIR=./hf_cache
```

**Popular Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast, good quality)
- `sentence-transformers/all-mpnet-base-v2` (768 dim, high quality)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim, multilingual)

**Pros:**
- No API costs
- Works offline
- Many model options

**Cons:**
- Larger disk usage
- Initial model download
- Higher memory usage

### Embedding Model Selection

**For Speed:**
```bash
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
```

**For Quality:**
```bash
UML_EMBEDDING_PROVIDER=openai
UML_EMBEDDING_MODEL=text-embedding-3-large
```

**For Multilingual:**
```bash
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**For Offline Use:**
```bash
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-mpnet-base-v2
UML_HF_CACHE_DIR=./models
```

## Performance Tuning

### Context Length Optimization

**For Speed:**
```bash
UML_MAX_CONTEXT_LENGTH=2000      # Smaller context
UML_DEFAULT_RECENT_COUNT=2       # Fewer recent messages
UML_DEFAULT_RELEVANT_COUNT=3     # Fewer relevant messages
```

**For Quality:**
```bash
UML_MAX_CONTEXT_LENGTH=8000      # Larger context
UML_DEFAULT_RECENT_COUNT=5       # More recent messages
UML_DEFAULT_RELEVANT_COUNT=10    # More relevant messages
```

**For Memory Efficiency:**
```bash
UML_MAX_CONTEXT_LENGTH=1000      # Very small context
UML_DEFAULT_RECENT_COUNT=1       # Minimal recent
UML_DEFAULT_RELEVANT_COUNT=2     # Minimal relevant
```

### Embedding Performance

**Fast Configuration:**
```bash
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
```

**Quality Configuration:**
```bash
UML_EMBEDDING_PROVIDER=openai
UML_EMBEDDING_MODEL=text-embedding-3-large
```

### Database Performance

**Optimize SQLite:**
```python
from universal_memory_layer.models.config import MemoryConfig

# Use WAL mode for better concurrency
config = MemoryConfig(
    database_path="./memory.db?mode=rwc&cache=shared&_journal_mode=WAL"
)
```

## Security Considerations

### API Key Security

1. **Never hardcode API keys:**
```python
# DON'T DO THIS
config = MemoryConfig(openai_api_key="sk-hardcoded-key")

# DO THIS
config = MemoryConfig()  # Loads from environment
```

2. **Use secure storage:**
```bash
# Production: Use secret management
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id openai-key --query SecretString --output text)
```

3. **Restrict API key permissions:**
- Set usage limits
- Monitor usage patterns
- Use separate keys for different environments

### Data Security

1. **Encrypt sensitive data:**
```python
# For sensitive conversations, consider encryption
import os
from cryptography.fernet import Fernet

# Generate key (store securely)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt before storing
encrypted_content = cipher.encrypt(message.encode())
```

2. **Secure file permissions:**
```bash
chmod 600 memory.db          # Owner read/write only
chmod 600 vector_store.faiss # Owner read/write only
chmod 700 ./data             # Owner access only
```

3. **Regular cleanup:**
```python
# Implement data retention policies
from datetime import datetime, timedelta

def cleanup_old_data(days_to_keep=30):
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    # Remove messages older than cutoff
```

## Configuration Examples

### Development Configuration

```bash
# .env for development
OPENAI_API_KEY=sk-dev-key
UML_DATABASE_PATH=./dev_memory.db
UML_VECTOR_STORE_PATH=./dev_vectors.faiss
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
UML_LOG_LEVEL=DEBUG
UML_MAX_CONTEXT_LENGTH=2000
```

### Production Configuration

```bash
# .env for production
OPENAI_API_KEY=${OPENAI_API_KEY}  # From secret manager
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
GOOGLE_API_KEY=${GOOGLE_API_KEY}
UML_DATABASE_PATH=/var/lib/uml/memory.db
UML_VECTOR_STORE_PATH=/var/lib/uml/vectors.faiss
UML_EMBEDDING_PROVIDER=openai
UML_EMBEDDING_MODEL=text-embedding-3-small
UML_LOG_LEVEL=WARNING
UML_MAX_CONTEXT_LENGTH=4000
```

### High-Performance Configuration

```bash
# Optimized for speed
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
UML_HF_CACHE_DIR=/tmp/hf_cache
UML_MAX_CONTEXT_LENGTH=1500
UML_DEFAULT_RECENT_COUNT=2
UML_DEFAULT_RELEVANT_COUNT=3
UML_LOG_LEVEL=ERROR
```

### Multi-Language Configuration

```bash
# Support for multiple languages
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
UML_MAX_CONTEXT_LENGTH=6000
UML_DEFAULT_RELEVANT_COUNT=8
```

### Offline Configuration

```bash
# No internet required after initial setup
UML_EMBEDDING_PROVIDER=huggingface
UML_HF_MODEL_PATH=sentence-transformers/all-mpnet-base-v2
UML_HF_CACHE_DIR=./offline_models
# Don't set API keys - will work with local embeddings only
```

### Testing Configuration

```python
# Configuration for unit tests
from universal_memory_layer.models.config import MemoryConfig

test_config = MemoryConfig(
    database_path=":memory:",  # In-memory SQLite
    vector_store_path="./test_vectors.faiss",
    embedding_provider="huggingface",
    huggingface_model_path="sentence-transformers/all-MiniLM-L6-v2",
    max_context_length=1000,
    default_recent_count=2,
    default_relevant_count=2
)
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set environment variables
ENV UML_DATABASE_PATH=/data/memory.db
ENV UML_VECTOR_STORE_PATH=/data/vectors.faiss
ENV UML_EMBEDDING_PROVIDER=huggingface
ENV UML_LOG_LEVEL=INFO

# Create data directory
RUN mkdir -p /data

# Run application
CMD ["python", "-m", "universal_memory_layer.cli.main"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  uml:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - UML_DATABASE_PATH=/data/memory.db
      - UML_VECTOR_STORE_PATH=/data/vectors.faiss
      - UML_EMBEDDING_PROVIDER=openai
      - UML_LOG_LEVEL=INFO
    volumes:
      - ./data:/data
    stdin_open: true
    tty: true
```

## Configuration Validation

### Validate Configuration

```python
from universal_memory_layer.models.config import MemoryConfig

def validate_config():
    """Validate current configuration."""
    try:
        config = MemoryConfig()
        
        print("Configuration Validation:")
        print(f"✓ Database path: {config.database_path}")
        print(f"✓ Vector store path: {config.vector_store_path}")
        print(f"✓ Embedding provider: {config.embedding_provider}")
        print(f"✓ Max context length: {config.max_context_length}")
        
        # Check API keys
        providers = ['openai', 'anthropic', 'google']
        available_providers = [p for p in providers if config.has_api_key(p)]
        
        if available_providers:
            print(f"✓ Available providers: {', '.join(available_providers)}")
        else:
            print("⚠️  No API keys configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

# Run validation
validate_config()
```

### Configuration Debugging

```python
import os
from universal_memory_layer.models.config import MemoryConfig

def debug_config():
    """Debug configuration loading."""
    
    print("Environment Variables:")
    env_vars = [
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
        'UML_DATABASE_PATH', 'UML_VECTOR_STORE_PATH',
        'UML_EMBEDDING_PROVIDER', 'UML_EMBEDDING_MODEL',
        'UML_MAX_CONTEXT_LENGTH', 'UML_LOG_LEVEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if var.endswith('_KEY'):
            # Mask API keys
            display_value = f"{value[:8]}..." if value else "Not set"
        else:
            display_value = value or "Not set"
        print(f"  {var}: {display_value}")
    
    print("\nLoaded Configuration:")
    config = MemoryConfig()
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

# Run debugging
debug_config()
```