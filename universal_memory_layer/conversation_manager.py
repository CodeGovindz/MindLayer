"""ConversationManager orchestration layer for managing LLM interactions with memory."""

from typing import List, Optional, Dict, Any, Union
from enum import Enum

from .models.message import Message
from .models.config import MemoryConfig
from .storage.memory_store import MemoryStore, MemoryStoreError
from .clients.base import BaseLLMClient
from .clients.chatgpt_client import ChatGPTClient
from .clients.claude_client import ClaudeClient
from .clients.gemini_client import GeminiClient
from .clients.groq_client import GroqClient
from .logging_config import get_logger
from .errors import (
    ConversationError, 
    LLMClientError, 
    StorageError, 
    handle_errors, 
    safe_execute,
    create_user_friendly_message
)

logger = get_logger(__name__)


class ContextStrategy(Enum):
    """Available context retrieval strategies."""
    RECENT = "recent"
    RELEVANT = "relevant"
    HYBRID = "hybrid"


# Keep backward compatibility
class ConversationManagerError(ConversationError):
    """Exception raised when ConversationManager operations fail."""
    pass


class ConversationManager:
    """
    Orchestrates interactions between LLM clients and memory storage.
    
    Manages context retrieval strategies, model switching, and conversation flow
    while maintaining persistent memory across different LLM providers.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize ConversationManager with configuration.
        
        Args:
            config: Memory configuration. If None, uses default configuration.
        """
        self.config = config or MemoryConfig()
        self.memory_store = MemoryStore(self.config)
        
        # Available LLM clients
        self._clients: Dict[str, BaseLLMClient] = {}
        self._current_model: Optional[str] = None
        
        # Initialize available clients based on API keys
        self._initialize_clients()
        
        logger.info("ConversationManager initialized")
    
    def initialize(self) -> None:
        """
        Initialize the conversation manager and memory store.
        
        Raises:
            ConversationManagerError: If initialization fails
        """
        try:
            self.memory_store.initialize()
            logger.info("ConversationManager initialization completed")
        except MemoryStoreError as e:
            logger.error(f"Failed to initialize memory store: {e}")
            raise ConversationManagerError(f"Failed to initialize ConversationManager: {e}")
    
    def _initialize_clients(self) -> None:
        """Initialize available LLM clients based on API keys."""
        # Initialize ChatGPT client if API key is available
        if self.config.has_api_key("openai"):
            try:
                self._clients["chatgpt"] = ChatGPTClient(
                    api_key=self.config.openai_api_key,
                    model="gpt-3.5-turbo"
                )
                self._clients["gpt-4"] = ChatGPTClient(
                    api_key=self.config.openai_api_key,
                    model="gpt-4"
                )
                self._clients["gpt-5-nano"] = ChatGPTClient(
                    api_key=self.config.openai_api_key,
                    model="gpt-5-nano"
                )
                logger.debug("ChatGPT clients initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGPT client: {e}")
        
        # Initialize Claude client if API key is available
        if self.config.has_api_key("anthropic"):
            try:
                self._clients["claude"] = ClaudeClient(
                    api_key=self.config.anthropic_api_key
                )
                logger.debug("Claude client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
        
        # Initialize Gemini client if API key is available
        if self.config.has_api_key("google"):
            try:
                self._clients["gemini"] = GeminiClient(
                    api_key=self.config.google_api_key,
                    model="gemini-pro"
                )
                self._clients["gemini-2.0-flash"] = GeminiClient(
                    api_key=self.config.google_api_key,
                    model="gemini-2.0-flash"
                )
                logger.debug("Gemini clients initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        
        # Initialize Groq client if API key is available
        if self.config.has_api_key("groq"):
            try:
                self._clients["llama-3.1-8b"] = GroqClient(
                    api_key=self.config.groq_api_key,
                    model="llama-3.1-8b-instant"  # Using available Groq model
                )
                self._clients["mixtral-8x7b"] = GroqClient(
                    api_key=self.config.groq_api_key,
                    model="mixtral-8x7b-32768"  # Using available Groq model
                )
                logger.debug("Groq clients initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # Set default model if any clients are available
        if self._clients and not self._current_model:
            self._current_model = list(self._clients.keys())[0]
            logger.info(f"Default model set to: {self._current_model}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available LLM models.
        
        Returns:
            List of available model names
        """
        return list(self._clients.keys())
    
    def get_current_model(self) -> Optional[str]:
        """
        Get the currently selected model.
        
        Returns:
            Current model name or None if no model is selected
        """
        return self._current_model
    
    def switch_model(self, model: str) -> None:
        """
        Switch to a different LLM model.
        
        Args:
            model: Name of the model to switch to
            
        Raises:
            ConversationManagerError: If model is not available
        """
        if model not in self._clients:
            available = ", ".join(self._clients.keys())
            raise ConversationManagerError(
                f"Model '{model}' not available. Available models: {available}"
            )
        
        old_model = self._current_model
        self._current_model = model
        logger.info(f"Switched model from {old_model} to {model}")
    
    def get_context(
        self,
        query: str,
        strategy: Union[str, ContextStrategy] = ContextStrategy.RECENT,
        k: Optional[int] = None,
        recent_count: Optional[int] = None,
        relevant_count: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve context messages using specified strategy.
        
        Args:
            query: Query text for relevant context retrieval
            strategy: Context retrieval strategy ('recent', 'relevant', or 'hybrid')
            k: Number of messages to retrieve (overrides strategy-specific counts)
            recent_count: Number of recent messages for recent/hybrid strategies
            relevant_count: Number of relevant messages for relevant/hybrid strategies
            
        Returns:
            List of Message objects ordered appropriately for the strategy
            
        Raises:
            ConversationManagerError: If context retrieval fails
        """
        try:
            # Convert string to enum if needed
            if isinstance(strategy, str):
                strategy = ContextStrategy(strategy.lower())
            
            # Set default counts
            if recent_count is None:
                recent_count = self.config.default_recent_count
            if relevant_count is None:
                relevant_count = self.config.default_relevant_count
            
            if strategy == ContextStrategy.RECENT:
                count = k or recent_count
                messages = self.memory_store.get_recent(count)
                logger.debug(f"Retrieved {len(messages)} recent messages")
                return messages
            
            elif strategy == ContextStrategy.RELEVANT:
                count = k or relevant_count
                messages = self.memory_store.get_relevant(query, count)
                logger.debug(f"Retrieved {len(messages)} relevant messages")
                return messages
            
            elif strategy == ContextStrategy.HYBRID:
                # Get both recent and relevant messages
                recent_messages = self.memory_store.get_recent(recent_count)
                relevant_messages = self.memory_store.get_relevant(query, relevant_count)
                
                # Combine and deduplicate by message ID
                seen_ids = set()
                combined_messages = []
                
                # Add recent messages first (to maintain chronological order)
                for msg in recent_messages:
                    if msg.id not in seen_ids:
                        combined_messages.append(msg)
                        seen_ids.add(msg.id)
                
                # Add relevant messages that aren't already included
                for msg in relevant_messages:
                    if msg.id not in seen_ids:
                        combined_messages.append(msg)
                        seen_ids.add(msg.id)
                
                # If k is specified, limit the total number of messages
                if k is not None:
                    combined_messages = combined_messages[:k]
                
                logger.debug(f"Retrieved {len(combined_messages)} hybrid messages")
                return combined_messages
            
            else:
                raise ConversationManagerError(f"Unknown context strategy: {strategy}")
                
        except MemoryStoreError as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise ConversationManagerError(f"Failed to retrieve context: {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving context: {e}")
            raise ConversationManagerError(f"Failed to retrieve context: {e}")
    
    def format_context(self, messages: List[Message], model: Optional[str] = None) -> List[str]:
        """
        Format context messages for specific model.
        
        Args:
            messages: List of Message objects to format
            model: Target model name. If None, uses current model.
            
        Returns:
            List of formatted context strings
            
        Raises:
            ConversationManagerError: If formatting fails
        """
        if not messages:
            return []
        
        target_model = model or self._current_model
        if not target_model:
            raise ConversationManagerError("No model specified and no current model set")
        
        try:
            formatted_context = []
            
            for message in messages:
                # Format based on target model characteristics
                if target_model.startswith("gpt") or target_model == "chatgpt":
                    # OpenAI models prefer clear role indicators
                    role_prefix = "User" if message.role == "user" else "Assistant"
                    formatted_msg = f"{role_prefix}: {message.content}"
                
                elif target_model == "claude":
                    # Claude prefers conversational format
                    if message.role == "user":
                        formatted_msg = f"Human: {message.content}"
                    else:
                        formatted_msg = f"Assistant: {message.content}"
                
                elif target_model == "gemini":
                    # Gemini works well with simple role prefixes
                    role_prefix = "User" if message.role == "user" else "Model"
                    formatted_msg = f"{role_prefix}: {message.content}"
                
                else:
                    # Default formatting for unknown models
                    role_prefix = message.role.capitalize()
                    formatted_msg = f"{role_prefix}: {message.content}"
                
                formatted_context.append(formatted_msg)
            
            logger.debug(f"Formatted {len(formatted_context)} context messages for {target_model}")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Failed to format context: {e}")
            raise ConversationManagerError(f"Failed to format context: {e}")
    
    def _calculate_context_length(self, context: List[str]) -> int:
        """
        Calculate approximate token length of context.
        
        Args:
            context: List of context strings
            
        Returns:
            Approximate character count (rough token estimation)
        """
        return sum(len(ctx) for ctx in context)
    
    def _trim_context_to_limit(self, context: List[str]) -> List[str]:
        """
        Trim context to fit within max_context_length.
        
        Args:
            context: List of context strings
            
        Returns:
            Trimmed context list
        """
        if not context:
            return context
        
        total_length = self._calculate_context_length(context)
        
        if total_length <= self.config.max_context_length:
            return context
        
        # Remove messages from the beginning until we fit within the limit
        trimmed_context = context[:]
        while trimmed_context and self._calculate_context_length(trimmed_context) > self.config.max_context_length:
            trimmed_context.pop(0)
        
        logger.debug(f"Trimmed context from {len(context)} to {len(trimmed_context)} messages")
        return trimmed_context
    
    def clear_memory(self, confirm: bool = False) -> None:
        """
        Clear all stored conversation memory.
        
        Args:
            confirm: Must be True to actually clear data (safety measure)
            
        Raises:
            ConversationManagerError: If clearing fails
        """
        try:
            self.memory_store.clear_memory(confirm=confirm)
            logger.info("Conversation memory cleared")
        except MemoryStoreError as e:
            logger.error(f"Failed to clear memory: {e}")
            raise ConversationManagerError(f"Failed to clear memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary containing memory statistics
            
        Raises:
            ConversationManagerError: If getting stats fails
        """
        try:
            stats = self.memory_store.get_stats()
            stats['available_models'] = self.get_available_models()
            stats['current_model'] = self.get_current_model()
            return stats
        except MemoryStoreError as e:
            logger.error(f"Failed to get memory stats: {e}")
            raise ConversationManagerError(f"Failed to get memory stats: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    @handle_errors(
        error_mapping={
            MemoryStoreError: StorageError,
            Exception: ConversationError
        }
    )
    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        use_memory: bool = True,
        context_strategy: Union[str, ContextStrategy] = ContextStrategy.RECENT,
        context_count: Optional[int] = None,
        store_interaction: bool = True
    ) -> str:
        """
        Main chat method that coordinates LLM clients and memory.
        
        Args:
            message: User message to send to the LLM
            model: Model to use. If None, uses current model.
            use_memory: Whether to retrieve and use context from memory
            context_strategy: Strategy for context retrieval ('recent', 'relevant', 'hybrid')
            context_count: Number of context messages to retrieve
            store_interaction: Whether to store this interaction in memory
            
        Returns:
            LLM response as a string
            
        Raises:
            ConversationManagerError: If chat operation fails
        """
        try:
            # Determine which model to use
            target_model = model or self._current_model
            if not target_model:
                raise ConversationManagerError("No model specified and no current model set")
            
            if target_model not in self._clients:
                available = ", ".join(self._clients.keys())
                raise ConversationManagerError(
                    f"Model '{target_model}' not available. Available models: {available}"
                )
            
            # Get LLM client
            client = self._clients[target_model]
            
            # Retrieve context if requested
            context_messages = []
            formatted_context = []
            
            if use_memory:
                try:
                    context_messages = self.get_context(
                        query=message,
                        strategy=context_strategy,
                        k=context_count
                    )
                    formatted_context = self.format_context(context_messages, target_model)
                    
                    # Trim context to fit within limits
                    formatted_context = self._trim_context_to_limit(formatted_context)
                    
                    logger.debug(f"Using {len(formatted_context)} context messages for {target_model}")
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve context, proceeding without: {e}")
                    formatted_context = []
            
            # Store user message if requested
            if store_interaction:
                try:
                    self.memory_store.store_message(
                        content=message,
                        role="user",
                        model=target_model,
                        metadata={"context_strategy": str(context_strategy)}
                    )
                    logger.debug("Stored user message in memory")
                except Exception as e:
                    logger.warning(f"Failed to store user message: {e}")
            
            # Send message to LLM with context
            try:
                response = client.chat(message, formatted_context)
                logger.debug(f"Received response from {target_model}")
            except Exception as e:
                logger.error(f"LLM client error for {target_model}: {e}")
                raise ConversationManagerError(f"Failed to get response from {target_model}: {e}")
            
            # Store assistant response if requested
            if store_interaction:
                try:
                    self.memory_store.store_message(
                        content=response,
                        role="assistant",
                        model=target_model,
                        metadata={
                            "context_strategy": str(context_strategy),
                            "context_count": len(formatted_context)
                        }
                    )
                    logger.debug("Stored assistant response in memory")
                except Exception as e:
                    logger.warning(f"Failed to store assistant response: {e}")
            
            return response
            
        except ConversationManagerError:
            # Re-raise ConversationManagerError as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat method: {e}")
            raise ConversationManagerError(f"Chat operation failed: {e}")
    
    def start_conversation(
        self,
        model: Optional[str] = None,
        context_strategy: Union[str, ContextStrategy] = ContextStrategy.RECENT
    ) -> Dict[str, Any]:
        """
        Start a new conversation session.
        
        Args:
            model: Model to use for the conversation
            context_strategy: Default context strategy for the conversation
            
        Returns:
            Dictionary with conversation session information
            
        Raises:
            ConversationManagerError: If starting conversation fails
        """
        try:
            # Set model if provided
            if model:
                self.switch_model(model)
            
            # Get current state
            current_model = self.get_current_model()
            if not current_model:
                raise ConversationManagerError("No model available for conversation")
            
            session_info = {
                "model": current_model,
                "context_strategy": str(context_strategy),
                "available_models": self.get_available_models(),
                "memory_stats": self.get_memory_stats()
            }
            
            logger.info(f"Started conversation with {current_model}")
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise ConversationManagerError(f"Failed to start conversation: {e}")
    
    def continue_conversation(
        self,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation with a message.
        
        Args:
            message: User message
            **kwargs: Additional arguments passed to chat method
            
        Returns:
            Dictionary with response and conversation metadata
            
        Raises:
            ConversationManagerError: If continuing conversation fails
        """
        try:
            # Get response using chat method
            response = self.chat(message, **kwargs)
            
            # Return structured response with metadata
            result = {
                "response": response,
                "model": self.get_current_model(),
                "context_strategy": kwargs.get("context_strategy", ContextStrategy.RECENT),
                "used_memory": kwargs.get("use_memory", True),
                "stored_interaction": kwargs.get("store_interaction", True)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to continue conversation: {e}")
            raise ConversationManagerError(f"Failed to continue conversation: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Memory store handles its own cleanup
        pass