# Universal Memory Layer API Documentation

This document provides comprehensive API documentation for all public interfaces in the Universal Memory Layer.

## Table of Contents

- [ConversationManager](#conversationmanager)
- [MemoryStore](#memorystore)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [LLM Clients](#llm-clients)
- [Embedding Providers](#embedding-providers)
- [Error Handling](#error-handling)

## ConversationManager

The main orchestration class for managing LLM interactions with memory.

### Class: `ConversationManager`

**Location**: `universal_memory_layer.conversation_manager`

#### Constructor

```python
ConversationManager(config: Optional[MemoryConfig] = None)
```

**Parameters:**
- `config` (Optional[MemoryConfig]): Memory configuration. If None, uses default configuration.

**Example:**
```python
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

# With default configuration
manager = ConversationManager()

# With custom configuration
config = MemoryConfig(database_path="./custom.db")
manager = ConversationManager(config)
```

#### Methods

##### `initialize() -> None`

Initialize the conversation manager and memory store. Must be called before using other methods.

**Raises:**
- `ConversationManagerError`: If initialization fails

**Example:**
```python
manager = ConversationManager()
manager.initialize()
```

##### `chat(message: str, model: Optional[str] = None, use_memory: bool = True, context_strategy: Union[str, ContextStrategy] = ContextStrategy.RECENT, context_count: Optional[int] = None, store_interaction: bool = True) -> str`

Main chat method that coordinates LLM clients and memory.

**Parameters:**
- `message` (str): User message to send to the LLM
- `model` (Optional[str]): Model to use. If None, uses current model
- `use_memory` (bool): Whether to retrieve and use context from memory
- `context_strategy` (Union[str, ContextStrategy]): Strategy for context retrieval ('recent', 'relevant', 'hybrid')
- `context_count` (Optional[int]): Number of context messages to retrieve
- `store_interaction` (bool): Whether to store this interaction in memory

**Returns:**
- `str`: LLM response

**Raises:**
- `ConversationManagerError`: If chat operation fails

**Example:**
```python
# Basic chat
response = manager.chat("Hello, how are you?")

# Chat with specific model and context strategy
response = manager.chat(
    "Tell me about machine learning",
    model="chatgpt",
    context_strategy="relevant",
    context_count=5
)

# Chat without storing interaction
response = manager.chat(
    "What's the weather?",
    store_interaction=False
)
```

##### `get_context(query: str, strategy: Union[str, ContextStrategy] = ContextStrategy.RECENT, k: Optional[int] = None, recent_count: Optional[int] = None, relevant_count: Optional[int] = None) -> List[Message]`

Retrieve context messages using specified strategy.

**Parameters:**
- `query` (str): Query text for relevant context retrieval
- `strategy` (Union[str, ContextStrategy]): Context retrieval strategy
- `k` (Optional[int]): Number of messages to retrieve (overrides strategy-specific counts)
- `recent_count` (Optional[int]): Number of recent messages for recent/hybrid strategies
- `relevant_count` (Optional[int]): Number of relevant messages for relevant/hybrid strategies

**Returns:**
- `List[Message]`: List of Message objects ordered appropriately for the strategy

**Raises:**
- `ConversationManagerError`: If context retrieval fails

**Example:**
```python
# Get recent messages
recent = manager.get_context("", strategy="recent", k=5)

# Get relevant messages
relevant = manager.get_context("machine learning", strategy="relevant", k=3)

# Get hybrid context
hybrid = manager.get_context(
    "neural networks",
    strategy="hybrid",
    recent_count=3,
    relevant_count=5
)
```

##### `switch_model(model: str) -> None`

Switch to a different LLM model.

**Parameters:**
- `model` (str): Name of the model to switch to

**Raises:**
- `ConversationManagerError`: If model is not available

**Example:**
```python
manager.switch_model("claude")
manager.switch_model("gpt-4")
```

##### `get_available_models() -> List[str]`

Get list of available LLM models.

**Returns:**
- `List[str]`: List of available model names

**Example:**
```python
models = manager.get_available_models()
print(f"Available models: {models}")
```

##### `get_current_model() -> Optional[str]`

Get the currently selected model.

**Returns:**
- `Optional[str]`: Current model name or None if no model is selected

**Example:**
```python
current = manager.get_current_model()
print(f"Current model: {current}")
```

##### `clear_memory(confirm: bool = False) -> None`

Clear all stored conversation memory.

**Parameters:**
- `confirm` (bool): Must be True to actually clear data (safety measure)

**Raises:**
- `ConversationManagerError`: If clearing fails

**Example:**
```python
# Clear memory (requires confirmation)
manager.clear_memory(confirm=True)
```

##### `get_memory_stats() -> Dict[str, Any]`

Get statistics about the memory store.

**Returns:**
- `Dict[str, Any]`: Dictionary containing memory statistics

**Raises:**
- `ConversationManagerError`: If getting stats fails

**Example:**
```python
stats = manager.get_memory_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Available models: {stats['available_models']}")
```

#### Context Manager Support

ConversationManager supports context manager protocol:

```python
with ConversationManager() as manager:
    response = manager.chat("Hello!")
    print(response)
# Automatic cleanup on exit
```

## MemoryStore

Handles persistent storage and retrieval of conversation messages with semantic search capabilities.

### Class: `MemoryStore`

**Location**: `universal_memory_layer.storage.memory_store`

#### Constructor

```python
MemoryStore(config: Optional[MemoryConfig] = None)
```

**Parameters:**
- `config` (Optional[MemoryConfig]): Memory configuration. If None, uses default configuration.

#### Methods

##### `initialize() -> None`

Initialize all components of the memory store.

**Raises:**
- `MemoryStoreError`: If initialization fails

##### `store_message(content: str, role: str, model: str, metadata: Optional[Dict[str, Any]] = None, generate_embedding: bool = True) -> str`

Store a message with optional embedding generation.

**Parameters:**
- `content` (str): Message content
- `role` (str): Message role ('user' or 'assistant')
- `model` (str): Model name that generated/received the message
- `metadata` (Optional[Dict[str, Any]]): Optional metadata dictionary
- `generate_embedding` (bool): Whether to generate embeddings for semantic search

**Returns:**
- `str`: Message ID of the stored message

**Raises:**
- `MemoryStoreError`: If storing message fails

**Example:**
```python
message_id = memory.store_message(
    content="Hello world",
    role="user",
    model="chatgpt",
    metadata={"session": "test"},
    generate_embedding=True
)
```

##### `get_recent(n: int = None) -> List[Message]`

Retrieve recent messages in chronological order.

**Parameters:**
- `n` (int): Number of recent messages to retrieve. If None, uses config default.

**Returns:**
- `List[Message]`: List of Message objects ordered chronologically (oldest first)

**Raises:**
- `MemoryStoreError`: If retrieval fails

**Example:**
```python
recent_messages = memory.get_recent(10)
for msg in recent_messages:
    print(f"{msg.role}: {msg.content}")
```

##### `get_relevant(query: str, k: int = None, strategy: str = 'hybrid', score_threshold: float = 0.1) -> List[Message]`

Retrieve relevant messages using semantic search.

**Parameters:**
- `query` (str): Search query text
- `k` (int): Number of results to return. If None, uses config default.
- `strategy` (str): Search strategy ('semantic', 'recent', or 'hybrid')
- `score_threshold` (float): Minimum similarity score for semantic results

**Returns:**
- `List[Message]`: List of Message objects ordered by relevance

**Raises:**
- `MemoryStoreError`: If search fails

**Example:**
```python
relevant = memory.get_relevant(
    query="machine learning",
    k=5,
    strategy="semantic",
    score_threshold=0.2
)
```

##### `generate_embedding(text: str) -> List[float]`

Generate embedding for given text.

**Parameters:**
- `text` (str): Text to generate embedding for

**Returns:**
- `List[float]`: List of floats representing the embedding vector

**Raises:**
- `MemoryStoreError`: If embedding generation fails

##### `get_message_by_id(message_id: str) -> Optional[Message]`

Retrieve a specific message by ID.

**Parameters:**
- `message_id` (str): ID of the message to retrieve

**Returns:**
- `Optional[Message]`: Message object if found, None otherwise

**Raises:**
- `MemoryStoreError`: If retrieval fails

##### `get_stats() -> Dict[str, Any]`

Get comprehensive statistics about the memory store.

**Returns:**
- `Dict[str, Any]`: Dictionary containing various statistics

**Example:**
```python
stats = memory.get_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Embedding provider: {stats['embedding_provider']}")
```

## Data Models

### Message

**Location**: `universal_memory_layer.models.message`

```python
@dataclass
class Message:
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    model: str
    timestamp: datetime
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Example:**
```python
from universal_memory_layer.models.message import Message
from datetime import datetime

message = Message(
    id="msg_123",
    content="Hello world",
    role="user",
    model="chatgpt",
    timestamp=datetime.now(),
    metadata={"session": "test"}
)
```

### Conversation

**Location**: `universal_memory_layer.models.conversation`

```python
@dataclass
class Conversation:
    id: str
    title: str
    created_at: datetime
    last_updated: datetime
    messages: List[Message] = field(default_factory=list)
```

### MemoryConfig

**Location**: `universal_memory_layer.models.config`

Configuration model with environment variable support.

```python
@dataclass
class MemoryConfig:
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
    
    # Hugging Face settings
    huggingface_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_cache_dir: Optional[str] = None
```

#### Methods

##### `has_api_key(provider: str) -> bool`

Check if API key is available for the given provider.

**Parameters:**
- `provider` (str): Provider name ('openai', 'anthropic', 'google')

**Returns:**
- `bool`: True if API key is available and non-empty

**Example:**
```python
config = MemoryConfig()
if config.has_api_key("openai"):
    print("OpenAI API key is configured")
```

##### `to_dict() -> Dict[str, Any]`

Convert configuration to dictionary (excluding sensitive API keys).

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

##### `from_dict(config_dict: Dict[str, Any], load_from_env: bool = False) -> MemoryConfig`

Create MemoryConfig from dictionary.

**Parameters:**
- `config_dict` (Dict[str, Any]): Dictionary containing configuration values
- `load_from_env` (bool): Whether to load from environment variables

**Returns:**
- `MemoryConfig`: MemoryConfig instance

## Configuration

### Environment Variables

All configuration can be set via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | "" |
| `ANTHROPIC_API_KEY` | Anthropic API key | "" |
| `GOOGLE_API_KEY` | Google API key | "" |
| `UML_DATABASE_PATH` | SQLite database path | "memory.db" |
| `UML_VECTOR_STORE_PATH` | FAISS index path | "vector_store.faiss" |
| `UML_EMBEDDING_PROVIDER` | Embedding provider | "openai" |
| `UML_EMBEDDING_MODEL` | Embedding model name | "text-embedding-ada-002" |
| `UML_MAX_CONTEXT_LENGTH` | Max context length | 4000 |
| `UML_DEFAULT_RECENT_COUNT` | Default recent count | 3 |
| `UML_DEFAULT_RELEVANT_COUNT` | Default relevant count | 5 |
| `UML_HF_MODEL_PATH` | Hugging Face model path | "sentence-transformers/all-MiniLM-L6-v2" |
| `UML_HF_CACHE_DIR` | Hugging Face cache directory | None |
| `UML_LOG_LEVEL` | Logging level | "WARNING" |

## LLM Clients

### Base Interface

**Location**: `universal_memory_layer.clients.base`

```python
class BaseLLMClient:
    def chat(self, message: str, context: List[str] = None) -> str:
        """Send message with context to LLM and return response"""
        pass
    
    def format_prompt(self, message: str, context: List[str]) -> str:
        """Format message and context into model-specific prompt"""
        pass
```

### Available Clients

#### ChatGPTClient

**Location**: `universal_memory_layer.clients.chatgpt_client`

```python
ChatGPTClient(api_key: str, model: str = "gpt-3.5-turbo")
```

#### ClaudeClient

**Location**: `universal_memory_layer.clients.claude_client`

```python
ClaudeClient(api_key: str, model: str = "claude-3-sonnet-20240229")
```

#### GeminiClient

**Location**: `universal_memory_layer.clients.gemini_client`

```python
GeminiClient(api_key: str, model: str = "gemini-pro")
```

## Embedding Providers

### Base Interface

**Location**: `universal_memory_layer.embeddings.base`

```python
class BaseEmbeddingProvider:
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text"""
        pass
    
    def get_dimension(self) -> int:
        """Return embedding dimension for vector store setup"""
        pass
```

### Available Providers

#### OpenAIEmbeddingProvider

**Location**: `universal_memory_layer.embeddings.openai_provider`

```python
OpenAIEmbeddingProvider(api_key: str, model: str = "text-embedding-ada-002")
```

#### HuggingFaceEmbeddingProvider

**Location**: `universal_memory_layer.embeddings.huggingface_provider`

```python
HuggingFaceEmbeddingProvider(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Optional[str] = None
)
```

## Error Handling

### Exception Hierarchy

```
UMLError (base exception)
├── ConversationError
│   └── ConversationManagerError
├── StorageError
│   └── MemoryStoreError
├── LLMClientError
├── EmbeddingError
└── CLIError
```

### Error Handling Utilities

**Location**: `universal_memory_layer.errors`

#### `handle_errors` Decorator

```python
@handle_errors(
    error_mapping={
        SpecificError: GeneralError,
        Exception: UMLError
    }
)
def my_function():
    # Function implementation
    pass
```

#### `safe_execute` Function

```python
result = safe_execute(
    func=risky_function,
    args=(arg1, arg2),
    kwargs={'key': 'value'},
    default_return=None,
    error_message="Operation failed"
)
```

### Common Error Scenarios

#### API Key Issues
```python
try:
    manager = ConversationManager()
    manager.initialize()
except ConversationManagerError as e:
    if "API key not found" in str(e):
        print("Please set your API keys in environment variables")
```

#### Storage Issues
```python
try:
    memory.store_message("test", "user", "chatgpt")
except MemoryStoreError as e:
    if "database" in str(e).lower():
        print("Database error - check file permissions")
```

#### Network Issues
```python
try:
    response = manager.chat("Hello")
except ConversationManagerError as e:
    if "timeout" in str(e).lower():
        print("Network timeout - check your connection")
```

## Usage Patterns

### Context Manager Pattern

```python
# Recommended pattern for resource management
with ConversationManager() as manager:
    response = manager.chat("Hello!")
    print(response)
```

### Error Handling Pattern

```python
from universal_memory_layer.errors import ConversationManagerError

try:
    manager = ConversationManager()
    manager.initialize()
    response = manager.chat("Hello!")
except ConversationManagerError as e:
    print(f"Conversation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Configuration Pattern

```python
# Load configuration from environment
config = MemoryConfig()  # Automatically loads from env vars

# Or create custom configuration
config = MemoryConfig(
    database_path="./custom.db",
    embedding_provider="huggingface",
    max_context_length=8000
)

manager = ConversationManager(config)
```

### Batch Processing Pattern

```python
# Process multiple messages efficiently
messages = ["Hello", "How are you?", "Tell me a joke"]

with ConversationManager() as manager:
    for msg in messages:
        response = manager.chat(msg)
        print(f"Q: {msg}")
        print(f"A: {response}\n")
```