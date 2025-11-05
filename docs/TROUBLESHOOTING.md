# Universal Memory Layer - Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Universal Memory Layer.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [API Key Issues](#api-key-issues)
- [Database and Storage Issues](#database-and-storage-issues)
- [Embedding and Vector Search Issues](#embedding-and-vector-search-issues)
- [Performance Issues](#performance-issues)
- [CLI Issues](#cli-issues)
- [Network and Connectivity Issues](#network-and-connectivity-issues)
- [Error Messages Reference](#error-messages-reference)
- [Debugging Tips](#debugging-tips)

## Installation Issues

### Problem: `pip install` fails with dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**

1. **Use a virtual environment:**
```bash
python -m venv uml_env
source uml_env/bin/activate  # On Windows: uml_env\Scripts\activate
pip install -r requirements.txt
```

2. **Update pip and setuptools:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. **Install dependencies individually:**
```bash
pip install faiss-cpu numpy
pip install openai anthropic google-generativeai
pip install -r requirements.txt
```

### Problem: FAISS installation fails

**Symptoms:**
```
ERROR: Failed building wheel for faiss-cpu
```

**Solutions:**

1. **Use conda instead of pip:**
```bash
conda install -c conda-forge faiss-cpu
```

2. **Install pre-compiled wheel:**
```bash
pip install --find-links https://download.pytorch.org/whl/torch_stable.html faiss-cpu
```

3. **For Apple Silicon Macs:**
```bash
pip install faiss-cpu --no-cache-dir
```

### Problem: Import errors after installation

**Symptoms:**
```python
ImportError: No module named 'universal_memory_layer'
```

**Solutions:**

1. **Install in development mode:**
```bash
pip install -e .
```

2. **Check Python path:**
```python
import sys
sys.path.append('/path/to/universal-memory-layer')
```

3. **Verify installation:**
```bash
pip list | grep universal
python -c "import universal_memory_layer; print('OK')"
```

## Configuration Problems

### Problem: Environment variables not loading

**Symptoms:**
- CLI shows "No models available"
- API key errors despite setting environment variables

**Solutions:**

1. **Verify environment variables are set:**
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

2. **Use .env file:**
```bash
# Create .env file in project directory
cat > .env << EOF
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
EOF
```

3. **Set variables in Python:**
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

from universal_memory_layer.conversation_manager import ConversationManager
manager = ConversationManager()
```

4. **Check configuration loading:**
```python
from universal_memory_layer.models.config import MemoryConfig
config = MemoryConfig()
print(f"OpenAI key set: {config.has_api_key('openai')}")
print(f"Config: {config.to_dict()}")
```

### Problem: Invalid configuration values

**Symptoms:**
```
ValueError: max_context_length must be a positive integer
```

**Solutions:**

1. **Check environment variable types:**
```bash
# Wrong (string)
export UML_MAX_CONTEXT_LENGTH="4000"

# Correct (will be converted to int)
export UML_MAX_CONTEXT_LENGTH=4000
```

2. **Validate configuration:**
```python
from universal_memory_layer.models.config import MemoryConfig

try:
    config = MemoryConfig()
    config.validate()
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

3. **Reset to defaults:**
```bash
unset UML_MAX_CONTEXT_LENGTH
unset UML_DEFAULT_RECENT_COUNT
# etc.
```

## API Key Issues

### Problem: "API key not found" errors

**Symptoms:**
```
ConversationManagerError: OpenAI API key not found in configuration
```

**Solutions:**

1. **Verify API key format:**
```bash
# OpenAI keys start with 'sk-'
echo $OPENAI_API_KEY | grep '^sk-'

# Anthropic keys start with 'sk-ant-'
echo $ANTHROPIC_API_KEY | grep '^sk-ant-'

# Google keys start with 'AIza'
echo $GOOGLE_API_KEY | grep '^AIza'
```

2. **Test API keys directly:**
```python
import openai
openai.api_key = "your-key-here"

# Test with a simple request
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("OpenAI API key works")
except Exception as e:
    print(f"OpenAI API key error: {e}")
```

3. **Check key permissions:**
- Ensure API keys have necessary permissions
- Check usage limits and quotas
- Verify billing is set up correctly

### Problem: API quota exceeded

**Symptoms:**
```
openai.error.RateLimitError: You exceeded your current quota
```

**Solutions:**

1. **Check usage and billing:**
- Visit OpenAI/Anthropic/Google Cloud console
- Check current usage and limits
- Add billing information if needed

2. **Implement rate limiting:**
```python
import time
from universal_memory_layer.conversation_manager import ConversationManager

manager = ConversationManager()
manager.initialize()

# Add delays between requests
for message in messages:
    response = manager.chat(message)
    time.sleep(1)  # Wait 1 second between requests
```

3. **Use different models:**
```python
# Try less expensive models first
manager.switch_model("gpt-3.5-turbo")  # Instead of gpt-4
```

## Database and Storage Issues

### Problem: Database permission errors

**Symptoms:**
```
sqlite3.OperationalError: unable to open database file
```

**Solutions:**

1. **Check file permissions:**
```bash
ls -la memory.db
chmod 644 memory.db
```

2. **Check directory permissions:**
```bash
ls -la ./
mkdir -p data
export UML_DATABASE_PATH="./data/memory.db"
```

3. **Use absolute paths:**
```python
import os
from universal_memory_layer.models.config import MemoryConfig

config = MemoryConfig(
    database_path=os.path.abspath("./memory.db")
)
```

### Problem: Database corruption

**Symptoms:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. **Backup and recreate database:**
```bash
# Backup existing database
cp memory.db memory.db.backup

# Remove corrupted database
rm memory.db

# Restart application (will create new database)
python -m universal_memory_layer.cli.main
```

2. **Recover from backup:**
```bash
# If you have a backup
cp memory.db.backup memory.db

# Or export/import data
sqlite3 memory.db.backup ".dump" | sqlite3 memory_new.db
```

3. **Check disk space:**
```bash
df -h .
# Ensure sufficient disk space
```

### Problem: Vector store issues

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'vector_store.faiss'
```

**Solutions:**

1. **Let the system recreate the index:**
```python
from universal_memory_layer.storage.memory_store import MemoryStore
from universal_memory_layer.models.config import MemoryConfig

config = MemoryConfig()
memory = MemoryStore(config)
memory.initialize()

# Rebuild vector index
memory.rebuild_vector_index()
```

2. **Check vector store path:**
```bash
ls -la vector_store.faiss*
export UML_VECTOR_STORE_PATH="./data/vector_store.faiss"
```

3. **Clear and rebuild:**
```python
import os
from universal_memory_layer.storage.memory_store import MemoryStore

# Remove existing vector store
if os.path.exists("vector_store.faiss"):
    os.remove("vector_store.faiss")

# Initialize fresh
memory = MemoryStore()
memory.initialize()
```

## Embedding and Vector Search Issues

### Problem: Embedding generation fails

**Symptoms:**
```
EmbeddingError: Failed to generate embedding
```

**Solutions:**

1. **Check embedding provider configuration:**
```python
from universal_memory_layer.models.config import MemoryConfig

config = MemoryConfig()
print(f"Embedding provider: {config.embedding_provider}")
print(f"Embedding model: {config.embedding_model}")

if config.embedding_provider == "openai":
    print(f"OpenAI key available: {config.has_api_key('openai')}")
```

2. **Test embedding provider directly:**
```python
from universal_memory_layer.embeddings.openai_provider import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    api_key="your-openai-key",
    model="text-embedding-ada-002"
)

try:
    embedding = provider.generate_embedding("test text")
    print(f"Embedding generated: {len(embedding)} dimensions")
except Exception as e:
    print(f"Embedding error: {e}")
```

3. **Switch to Hugging Face embeddings:**
```bash
export UML_EMBEDDING_PROVIDER=huggingface
export UML_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Problem: Slow semantic search

**Symptoms:**
- Long delays when using `get_relevant()`
- High CPU usage during search

**Solutions:**

1. **Reduce search parameters:**
```python
# Use fewer results
relevant = memory.get_relevant("query", k=3)  # Instead of k=10

# Use recent strategy instead
recent = memory.get_recent(5)
```

2. **Optimize vector index:**
```python
memory.rebuild_vector_index()
memory.save_index()
```

3. **Use hybrid strategy sparingly:**
```python
# Hybrid is slower than recent or relevant alone
manager.chat("message", context_strategy="recent")  # Faster
```

### Problem: Hugging Face model download issues

**Symptoms:**
```
OSError: Can't load tokenizer for 'sentence-transformers/all-MiniLM-L6-v2'
```

**Solutions:**

1. **Check internet connection:**
```bash
ping huggingface.co
```

2. **Set cache directory:**
```bash
export UML_HF_CACHE_DIR="./hf_cache"
mkdir -p ./hf_cache
```

3. **Pre-download model:**
```python
from sentence_transformers import SentenceTransformer

# This will download the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

4. **Use different model:**
```bash
export UML_HF_MODEL_PATH="sentence-transformers/paraphrase-MiniLM-L6-v2"
```

## Performance Issues

### Problem: Slow response times

**Symptoms:**
- Long delays in chat responses
- High memory usage

**Solutions:**

1. **Reduce context length:**
```bash
export UML_MAX_CONTEXT_LENGTH=2000  # Instead of 4000
export UML_DEFAULT_RECENT_COUNT=2   # Instead of 3
```

2. **Use recent strategy:**
```python
# Faster than semantic search
response = manager.chat(
    "message",
    context_strategy="recent",
    context_count=3
)
```

3. **Disable memory for simple queries:**
```python
# Skip memory lookup for simple questions
response = manager.chat(
    "What's 2+2?",
    use_memory=False
)
```

4. **Profile performance:**
```python
import time

start = time.time()
response = manager.chat("test message")
end = time.time()

print(f"Response time: {end - start:.2f} seconds")
```

### Problem: High memory usage

**Symptoms:**
- System running out of RAM
- Slow performance with large conversation histories

**Solutions:**

1. **Limit conversation history:**
```python
# Periodically clear old messages
manager.clear_memory(confirm=True)
```

2. **Use smaller embedding models:**
```bash
export UML_EMBEDDING_PROVIDER=huggingface
export UML_HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
```

3. **Monitor memory usage:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")
```

## CLI Issues

### Problem: CLI doesn't start

**Symptoms:**
```bash
python -m universal_memory_layer.cli.main
# No output or immediate exit
```

**Solutions:**

1. **Check Python path:**
```bash
python -c "import universal_memory_layer; print('OK')"
```

2. **Run with verbose output:**
```bash
export UML_LOG_LEVEL=DEBUG
python -m universal_memory_layer.cli.main
```

3. **Check for import errors:**
```python
try:
    from universal_memory_layer.cli.main import main
    main()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### Problem: CLI commands not working

**Symptoms:**
- `/help` shows "Unknown command"
- Model switching fails

**Solutions:**

1. **Check command format:**
```bash
# Correct format (with forward slash)
/help
/switch
/models

# Not this
help
switch
```

2. **Verify CLI initialization:**
```bash
# Check if models are available
/models

# Check current status
/current
/stats
```

3. **Restart CLI:**
```bash
# Exit and restart
/quit
python -m universal_memory_layer.cli.main
```

## Network and Connectivity Issues

### Problem: API connection timeouts

**Symptoms:**
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool
```

**Solutions:**

1. **Check internet connection:**
```bash
ping api.openai.com
ping api.anthropic.com
```

2. **Configure proxy if needed:**
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

3. **Increase timeout:**
```python
# This would require modifying the client code
# or adding timeout configuration
```

### Problem: SSL certificate errors

**Symptoms:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions:**

1. **Update certificates:**
```bash
# On macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# On Ubuntu/Debian
sudo apt-get update && sudo apt-get install ca-certificates
```

2. **Check system time:**
```bash
date
# Ensure system time is correct
```

## Error Messages Reference

### Common Error Messages and Solutions

#### `ConversationManagerError: No model specified and no current model set`

**Cause:** No model selected or available models list is empty.

**Solution:**
```python
# Check available models
models = manager.get_available_models()
print(f"Available: {models}")

# Select a model
if models:
    manager.switch_model(models[0])
```

#### `MemoryStoreError: MemoryStore not initialized. Call initialize() first.`

**Cause:** Trying to use MemoryStore before initialization.

**Solution:**
```python
memory = MemoryStore()
memory.initialize()  # Must call this first
```

#### `ValueError: max_context_length must be a positive integer`

**Cause:** Invalid configuration value.

**Solution:**
```bash
export UML_MAX_CONTEXT_LENGTH=4000  # Must be positive integer
```

#### `FileNotFoundError: [Errno 2] No such file or directory: 'memory.db'`

**Cause:** Database file doesn't exist or wrong path.

**Solution:**
```python
# Let the system create the database
manager = ConversationManager()
manager.initialize()  # Will create database
```

## Debugging Tips

### Enable Debug Logging

```bash
export UML_LOG_LEVEL=DEBUG
python -m universal_memory_layer.cli.main
```

### Check Configuration

```python
from universal_memory_layer.models.config import MemoryConfig

config = MemoryConfig()
print("Configuration:")
for key, value in config.to_dict().items():
    print(f"  {key}: {value}")

print("\nAPI Keys:")
print(f"  OpenAI: {config.has_api_key('openai')}")
print(f"  Anthropic: {config.has_api_key('anthropic')}")
print(f"  Google: {config.has_api_key('google')}")
```

### Test Components Individually

```python
# Test memory store
from universal_memory_layer.storage.memory_store import MemoryStore

memory = MemoryStore()
memory.initialize()
print("Memory store: OK")

# Test conversation manager
from universal_memory_layer.conversation_manager import ConversationManager

manager = ConversationManager()
manager.initialize()
print(f"Available models: {manager.get_available_models()}")
```

### Check System Resources

```bash
# Check disk space
df -h .

# Check memory usage
free -h

# Check Python version
python --version

# Check installed packages
pip list | grep -E "(openai|anthropic|google|faiss|numpy)"
```

### Validate Installation

```python
#!/usr/bin/env python3
"""Validation script for Universal Memory Layer installation."""

def validate_installation():
    """Validate that UML is properly installed and configured."""
    
    print("Universal Memory Layer - Installation Validation")
    print("=" * 50)
    
    # Test imports
    try:
        from universal_memory_layer.conversation_manager import ConversationManager
        from universal_memory_layer.models.config import MemoryConfig
        from universal_memory_layer.storage.memory_store import MemoryStore
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test configuration
    try:
        config = MemoryConfig()
        print("✓ Configuration loaded")
        
        # Check API keys
        api_keys = {
            'OpenAI': config.has_api_key('openai'),
            'Anthropic': config.has_api_key('anthropic'),
            'Google': config.has_api_key('google')
        }
        
        for provider, has_key in api_keys.items():
            status = "✓" if has_key else "✗"
            print(f"{status} {provider} API key: {'Set' if has_key else 'Not set'}")
        
        if not any(api_keys.values()):
            print("⚠️  No API keys configured - CLI will have limited functionality")
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test memory store
    try:
        memory = MemoryStore(config)
        memory.initialize()
        print("✓ Memory store initialization successful")
    except Exception as e:
        print(f"✗ Memory store error: {e}")
        return False
    
    # Test conversation manager
    try:
        manager = ConversationManager(config)
        manager.initialize()
        models = manager.get_available_models()
        print(f"✓ Conversation manager initialized with {len(models)} models")
        
        if models:
            print(f"  Available models: {', '.join(models)}")
        else:
            print("  No models available (check API keys)")
        
    except Exception as e:
        print(f"✗ Conversation manager error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Installation validation completed successfully!")
    return True

if __name__ == "__main__":
    validate_installation()
```

### Getting Help

If you're still experiencing issues:

1. **Check the logs:** Look in the `logs/` directory for detailed error information
2. **Enable debug logging:** Set `UML_LOG_LEVEL=DEBUG`
3. **Run the validation script:** Use the script above to check your installation
4. **Check GitHub issues:** Look for similar problems in the project repository
5. **Create a minimal reproduction:** Isolate the problem to the smallest possible example

### Reporting Issues

When reporting issues, please include:

1. **System information:**
   - Operating system and version
   - Python version
   - Package versions (`pip list`)

2. **Configuration:**
   - Environment variables (without API keys)
   - Configuration file contents

3. **Error details:**
   - Full error message and stack trace
   - Steps to reproduce
   - Expected vs actual behavior

4. **Logs:**
   - Debug logs with `UML_LOG_LEVEL=DEBUG`
   - Any relevant log files