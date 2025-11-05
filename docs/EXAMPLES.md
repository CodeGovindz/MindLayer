# Universal Memory Layer - Usage Examples

This document provides comprehensive examples of how to use the Universal Memory Layer in various scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [CLI Examples](#cli-examples)
- [Programmatic Usage](#programmatic-usage)
- [Advanced Scenarios](#advanced-scenarios)
- [Integration Examples](#integration-examples)
- [Performance Optimization](#performance-optimization)

## Basic Usage

### Quick Start Example

```python
from universal_memory_layer.conversation_manager import ConversationManager

# Initialize and start chatting
manager = ConversationManager()
manager.initialize()

# Simple conversation
response = manager.chat("Hello, how are you?")
print(response)

# Continue conversation with context
response = manager.chat("What did we just talk about?")
print(response)
```

### CLI Quick Start

```bash
# Start interactive CLI
python -m universal_memory_layer.cli.main

# In the CLI:
# 1. Select a model (e.g., ChatGPT)
# 2. Start chatting:
Hello, how are you?

# Switch models while maintaining context:
/switch
# Select Claude
What did we just discuss?
```

## CLI Examples

### Basic CLI Commands

```bash
# Start CLI
python -m universal_memory_layer.cli.main

# Available commands:
/help              # Show all commands
/models            # List available models
/current           # Show current model
/switch            # Switch to different model
/recent 5          # Show last 5 messages
/search "topic"    # Search for messages about topic
/stats             # Show memory statistics
/clear             # Clear all memory (with confirmation)
/quit              # Exit CLI
```

### CLI Session Example

```bash
$ python -m universal_memory_layer.cli.main

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

[chatgpt] You: Hello! I'm working on a machine learning project.

🤖 chatgpt is thinking...

🤖 chatgpt:
----------------------------------------
Hello! That's exciting! I'd be happy to help with your machine learning project. What kind of project are you working on? Are you focusing on a particular domain like computer vision, natural language processing, or something else? And what stage are you at - just getting started, working on data preprocessing, model selection, or something else?
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

[claude] You: What were we just discussing?

🤖 claude is thinking...

🤖 claude:
----------------------------------------
We were just discussing your machine learning project! You mentioned that you're working on one, and I had asked you about what kind of project it is, what domain you're focusing on (like computer vision or NLP), and what stage you're currently at in the development process. I was trying to understand more details so I could provide better assistance.
----------------------------------------
```

### CLI Memory Management

```bash
# View recent conversation history
[chatgpt] You: /recent 10

📜 Last 3 messages:
==================================================
👤 [14:23:15] User 
   Hello! I'm working on a machine learning project.

🤖 [14:23:18] Assistant (chatgpt)
   Hello! That's exciting! I'd be happy to help with your machine learning project...

👤 [14:24:02] User 
   What were we just discussing?
==================================================

# Search for specific topics
[claude] You: /search "machine learning"

🔍 Search results for 'machine learning' (2 found):
==================================================
👤 [14:23:15] User 
   Hello! I'm working on a machine learning project.

🤖 [14:23:18] Assistant (chatgpt)
   Hello! That's exciting! I'd be happy to help with your machine learning project. What kind of project are you working on?...
==================================================

# View memory statistics
[claude] You: /stats

📊 Memory Statistics:
==============================
Total messages: 4
User messages: 2
Assistant messages: 2
Available models: chatgpt, gpt-4, claude
Current model: claude
==============================
```

## Programmatic Usage

### Basic Conversation Flow

```python
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

# Initialize with default configuration
manager = ConversationManager()
manager.initialize()

# Check available models
models = manager.get_available_models()
print(f"Available models: {models}")

# Start conversation with specific model
if "chatgpt" in models:
    manager.switch_model("chatgpt")
    
    # Send messages
    response1 = manager.chat("Hello! I'm learning about Python.")
    print(f"Response 1: {response1}")
    
    response2 = manager.chat("Can you help me with list comprehensions?")
    print(f"Response 2: {response2}")
    
    # Switch models while maintaining context
    if "claude" in models:
        manager.switch_model("claude")
        response3 = manager.chat("What were we discussing about Python?")
        print(f"Response 3: {response3}")
```

### Custom Configuration

```python
from universal_memory_layer.models.config import MemoryConfig
from universal_memory_layer.conversation_manager import ConversationManager

# Create custom configuration
config = MemoryConfig(
    database_path="./project_memory.db",
    embedding_provider="huggingface",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    max_context_length=8000,
    default_recent_count=5,
    default_relevant_count=10
)

# Initialize with custom config
manager = ConversationManager(config)
manager.initialize()

# Use the manager
response = manager.chat("Tell me about neural networks")
print(response)
```

### Context Strategy Examples

```python
# Different context strategies
manager = ConversationManager()
manager.initialize()

# Store some conversation history first
manager.chat("I'm interested in machine learning")
manager.chat("Specifically deep learning and neural networks")
manager.chat("I'm also curious about natural language processing")

# Use recent context (chronological)
response = manager.chat(
    "What should I learn first?",
    context_strategy="recent",
    context_count=3
)

# Use relevant context (semantic search)
response = manager.chat(
    "Tell me more about neural networks",
    context_strategy="relevant",
    context_count=5
)

# Use hybrid context (combines recent and relevant)
response = manager.chat(
    "How do I get started with NLP?",
    context_strategy="hybrid",
    context_count=7
)
```

### Memory Store Direct Usage

```python
from universal_memory_layer.storage.memory_store import MemoryStore
from universal_memory_layer.models.config import MemoryConfig

# Initialize memory store directly
config = MemoryConfig()
memory = MemoryStore(config)
memory.initialize()

# Store messages manually
message_id1 = memory.store_message(
    content="What is machine learning?",
    role="user",
    model="chatgpt",
    metadata={"session": "tutorial", "topic": "ml_basics"}
)

message_id2 = memory.store_message(
    content="Machine learning is a subset of artificial intelligence...",
    role="assistant",
    model="chatgpt",
    metadata={"session": "tutorial", "topic": "ml_basics"}
)

# Retrieve messages
recent_messages = memory.get_recent(5)
for msg in recent_messages:
    print(f"{msg.role}: {msg.content[:50]}...")

# Search for relevant messages
relevant = memory.get_relevant("neural networks", k=3)
for msg in relevant:
    print(f"Relevant: {msg.content[:50]}...")

# Get specific message
message = memory.get_message_by_id(message_id1)
if message:
    print(f"Found message: {message.content}")
```

## Advanced Scenarios

### Multi-Session Management

```python
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id: str, model: str = "chatgpt"):
        """Create a new conversation session."""
        config = MemoryConfig(
            database_path=f"./session_{session_id}.db"
        )
        manager = ConversationManager(config)
        manager.initialize()
        
        if model in manager.get_available_models():
            manager.switch_model(model)
        
        self.sessions[session_id] = manager
        return manager
    
    def get_session(self, session_id: str):
        """Get existing session."""
        return self.sessions.get(session_id)
    
    def chat_in_session(self, session_id: str, message: str):
        """Send message in specific session."""
        session = self.get_session(session_id)
        if session:
            return session.chat(message)
        return None

# Usage
session_mgr = SessionManager()

# Create different sessions for different topics
ml_session = session_mgr.create_session("machine_learning", "chatgpt")
web_session = session_mgr.create_session("web_development", "claude")

# Chat in different sessions
ml_response = session_mgr.chat_in_session("machine_learning", "Explain neural networks")
web_response = session_mgr.chat_in_session("web_development", "How do I build a REST API?")

print(f"ML Response: {ml_response}")
print(f"Web Response: {web_response}")
```

### Conversation Analysis

```python
from universal_memory_layer.conversation_manager import ConversationManager
from collections import Counter
import re

def analyze_conversation(manager: ConversationManager):
    """Analyze conversation patterns and topics."""
    
    # Get all messages
    all_messages = manager.get_context("", strategy="recent", k=1000)
    
    # Analyze by model
    model_usage = Counter(msg.model for msg in all_messages if msg.role == "assistant")
    
    # Analyze topics (simple keyword extraction)
    user_messages = [msg.content for msg in all_messages if msg.role == "user"]
    all_text = " ".join(user_messages).lower()
    
    # Extract common words (simple approach)
    words = re.findall(r'\b\w+\b', all_text)
    common_words = Counter(words).most_common(10)
    
    # Conversation statistics
    stats = {
        "total_messages": len(all_messages),
        "user_messages": len([m for m in all_messages if m.role == "user"]),
        "assistant_messages": len([m for m in all_messages if m.role == "assistant"]),
        "model_usage": dict(model_usage),
        "common_topics": common_words,
        "conversation_length_days": (
            max(msg.timestamp for msg in all_messages) - 
            min(msg.timestamp for msg in all_messages)
        ).days if all_messages else 0
    }
    
    return stats

# Usage
manager = ConversationManager()
manager.initialize()

# Have some conversations...
manager.chat("Tell me about Python programming")
manager.chat("How do I use list comprehensions?")
manager.switch_model("claude")
manager.chat("What are the best practices for code organization?")

# Analyze the conversation
analysis = analyze_conversation(manager)
print(f"Conversation Analysis: {analysis}")
```

### Batch Processing

```python
from universal_memory_layer.conversation_manager import ConversationManager
import json
from typing import List, Dict

def process_questions_batch(questions: List[str], model: str = "chatgpt") -> List[Dict]:
    """Process a batch of questions and return structured results."""
    
    manager = ConversationManager()
    manager.initialize()
    
    if model in manager.get_available_models():
        manager.switch_model(model)
    
    results = []
    
    for i, question in enumerate(questions):
        try:
            response = manager.chat(question)
            result = {
                "question_id": i,
                "question": question,
                "response": response,
                "model": manager.get_current_model(),
                "success": True
            }
        except Exception as e:
            result = {
                "question_id": i,
                "question": question,
                "response": None,
                "model": manager.get_current_model(),
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
        print(f"Processed question {i+1}/{len(questions)}")
    
    return results

# Usage
questions = [
    "What is machine learning?",
    "Explain neural networks in simple terms",
    "How do I start learning Python?",
    "What are the best practices for data science?",
    "Explain the difference between supervised and unsupervised learning"
]

results = process_questions_batch(questions, model="chatgpt")

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Processed {len(results)} questions")
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

app = Flask(__name__)

# Initialize conversation manager
config = MemoryConfig(database_path="./web_app_memory.db")
manager = ConversationManager(config)
manager.initialize()

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for web application."""
    try:
        data = request.json
        message = data.get('message', '')
        model = data.get('model', 'chatgpt')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Switch model if requested
        if model in manager.get_available_models():
            manager.switch_model(model)
        
        # Get response
        response = manager.chat(message)
        
        return jsonify({
            'response': response,
            'model': manager.get_current_model(),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models."""
    return jsonify({
        'models': manager.get_available_models(),
        'current': manager.get_current_model()
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    try:
        count = request.args.get('count', 10, type=int)
        messages = manager.get_context("", strategy="recent", k=count)
        
        history = [
            {
                'content': msg.content,
                'role': msg.role,
                'model': msg.model,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in messages
        ]
        
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Discord Bot Integration

```python
import discord
from discord.ext import commands
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize conversation manager
config = MemoryConfig(database_path="./discord_bot_memory.db")
manager = ConversationManager(config)

@bot.event
async def on_ready():
    """Initialize when bot is ready."""
    manager.initialize()
    print(f'{bot.user} has connected to Discord!')
    print(f'Available models: {manager.get_available_models()}')

@bot.command(name='chat')
async def chat_command(ctx, *, message):
    """Chat with AI using current model."""
    try:
        response = manager.chat(message)
        
        # Discord has message length limits
        if len(response) > 2000:
            response = response[:1997] + "..."
        
        await ctx.send(response)
        
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

@bot.command(name='switch')
async def switch_model(ctx, model_name):
    """Switch to different AI model."""
    try:
        available = manager.get_available_models()
        
        if model_name not in available:
            await ctx.send(f"Model '{model_name}' not available. Available: {', '.join(available)}")
            return
        
        manager.switch_model(model_name)
        await ctx.send(f"Switched to {model_name}")
        
    except Exception as e:
        await ctx.send(f"Error switching model: {str(e)}")

@bot.command(name='models')
async def list_models(ctx):
    """List available models."""
    models = manager.get_available_models()
    current = manager.get_current_model()
    
    model_list = "\n".join([f"{'→ ' if m == current else '  '}{m}" for m in models])
    await ctx.send(f"Available models:\n```\n{model_list}\n```")

@bot.command(name='history')
async def show_history(ctx, count: int = 5):
    """Show recent conversation history."""
    try:
        messages = manager.get_context("", strategy="recent", k=count)
        
        if not messages:
            await ctx.send("No conversation history found.")
            return
        
        history = []
        for msg in messages[-5:]:  # Show last 5 to fit in Discord
            role_emoji = "👤" if msg.role == "user" else "🤖"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            history.append(f"{role_emoji} {content}")
        
        await ctx.send("Recent history:\n" + "\n".join(history))
        
    except Exception as e:
        await ctx.send(f"Error getting history: {str(e)}")

# Run bot (you'll need to set DISCORD_TOKEN environment variable)
# bot.run(os.getenv('DISCORD_TOKEN'))
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig
import pandas as pd
from IPython.display import display, Markdown

# Initialize for notebook use
config = MemoryConfig(database_path="./notebook_memory.db")
manager = ConversationManager(config)
manager.initialize()

print("Universal Memory Layer initialized for Jupyter!")
print(f"Available models: {manager.get_available_models()}")

# Cell 2: Helper functions
def chat(message, model=None, show_context=False):
    """Enhanced chat function for notebooks."""
    if model:
        manager.switch_model(model)
    
    current_model = manager.get_current_model()
    
    if show_context:
        context = manager.get_context(message, strategy="relevant", k=3)
        if context:
            print("Context being used:")
            for msg in context:
                print(f"  {msg.role}: {msg.content[:50]}...")
            print()
    
    response = manager.chat(message)
    
    # Display nicely formatted response
    display(Markdown(f"**{current_model}:** {response}"))
    
    return response

def show_conversation_stats():
    """Show conversation statistics."""
    stats = manager.get_memory_stats()
    
    df = pd.DataFrame([
        ["Total Messages", stats.get('total_messages', 0)],
        ["Current Model", stats.get('current_model', 'None')],
        ["Available Models", ', '.join(stats.get('available_models', []))],
    ], columns=['Metric', 'Value'])
    
    display(df)

# Cell 3: Usage examples
# Start a conversation about data science
chat("I'm working on a data science project with customer churn prediction")

# Switch models and continue
chat("What algorithms would you recommend?", model="claude")

# Show conversation statistics
show_conversation_stats()

# Cell 4: Advanced analysis
def analyze_conversation_topics():
    """Analyze conversation topics using the memory store."""
    messages = manager.get_context("", strategy="recent", k=100)
    
    # Create DataFrame for analysis
    df = pd.DataFrame([
        {
            'role': msg.role,
            'model': msg.model,
            'content_length': len(msg.content),
            'timestamp': msg.timestamp,
            'content_preview': msg.content[:100]
        }
        for msg in messages
    ])
    
    if not df.empty:
        print("Conversation Analysis:")
        print(f"Total messages: {len(df)}")
        print(f"Average message length: {df['content_length'].mean():.1f} characters")
        print("\nMessages by role:")
        print(df['role'].value_counts())
        print("\nMessages by model:")
        print(df['model'].value_counts())
        
        display(df.head())
    
    return df

# Run analysis
conversation_df = analyze_conversation_topics()
```

## Performance Optimization

### Efficient Context Management

```python
from universal_memory_layer.conversation_manager import ConversationManager
from universal_memory_layer.models.config import MemoryConfig

# Optimize for performance
config = MemoryConfig(
    max_context_length=2000,  # Smaller context for faster processing
    default_recent_count=3,   # Fewer recent messages
    default_relevant_count=3, # Fewer relevant messages
    embedding_provider="huggingface"  # Local embeddings for speed
)

manager = ConversationManager(config)
manager.initialize()

# Use recent strategy for speed when semantic search isn't needed
fast_response = manager.chat(
    "Quick question: what's 2+2?",
    context_strategy="recent",
    context_count=1,
    use_memory=False  # Skip memory for simple queries
)

# Use relevant strategy only when needed
contextual_response = manager.chat(
    "Based on our previous discussion about machine learning, what should I learn next?",
    context_strategy="relevant",
    context_count=5
)
```

### Memory Management

```python
from universal_memory_layer.storage.memory_store import MemoryStore
from universal_memory_layer.models.config import MemoryConfig

# Initialize memory store
config = MemoryConfig()
memory = MemoryStore(config)
memory.initialize()

# Periodic cleanup function
def cleanup_old_messages(days_to_keep=30):
    """Remove messages older than specified days."""
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # Get all messages (this is a simplified example)
    all_messages = memory.get_recent(10000)  # Get a large number
    
    old_messages = [msg for msg in all_messages if msg.timestamp < cutoff_date]
    
    print(f"Found {len(old_messages)} messages older than {days_to_keep} days")
    
    # In a real implementation, you'd want a proper delete method
    # This is just for demonstration
    return len(old_messages)

# Optimize vector store
def optimize_vector_store():
    """Rebuild vector index for better performance."""
    try:
        memory.rebuild_vector_index()
        print("Vector index optimized")
    except Exception as e:
        print(f"Optimization failed: {e}")

# Usage
old_count = cleanup_old_messages(30)
optimize_vector_store()
```

### Batch Operations

```python
from universal_memory_layer.storage.memory_store import MemoryStore
from universal_memory_layer.models.config import MemoryConfig

def batch_store_messages(messages_data):
    """Efficiently store multiple messages."""
    config = MemoryConfig()
    memory = MemoryStore(config)
    memory.initialize()
    
    stored_ids = []
    
    # Store messages in batch (disable embedding generation for speed)
    for msg_data in messages_data:
        message_id = memory.store_message(
            content=msg_data['content'],
            role=msg_data['role'],
            model=msg_data['model'],
            metadata=msg_data.get('metadata', {}),
            generate_embedding=False  # Skip embeddings for speed
        )
        stored_ids.append(message_id)
    
    # Generate embeddings in batch after storing
    print(f"Stored {len(stored_ids)} messages")
    
    # Optionally rebuild vector index once at the end
    memory.rebuild_vector_index()
    
    return stored_ids

# Usage
messages = [
    {'content': 'Hello', 'role': 'user', 'model': 'chatgpt'},
    {'content': 'Hi there!', 'role': 'assistant', 'model': 'chatgpt'},
    {'content': 'How are you?', 'role': 'user', 'model': 'chatgpt'},
]

batch_store_messages(messages)
```

These examples demonstrate the flexibility and power of the Universal Memory Layer across different use cases, from simple CLI interactions to complex web applications and performance-optimized scenarios.