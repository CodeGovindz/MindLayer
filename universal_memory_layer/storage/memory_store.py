"""MemoryStore class that integrates database storage with vector search capabilities."""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from ..models.message import Message
from ..models.config import MemoryConfig
from ..embeddings.base import BaseEmbeddingProvider, EmbeddingError
from .database import DatabaseManager
from .message_repository import MessageRepository
from .semantic_search import SemanticSearch, SearchResult, SemanticSearchError

logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    """Exception raised when MemoryStore operations fail."""
    pass


class MemoryStore:
    """
    Main interface for storing and retrieving conversation messages with semantic search.
    
    Integrates database storage with vector search capabilities, providing both
    recent message retrieval and semantic search functionality.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize MemoryStore with configuration.
        
        Args:
            config: Memory configuration. If None, uses default configuration.
        """
        self.config = config or MemoryConfig()
        
        # Initialize components
        self._embedding_provider = None
        self._message_repository = None
        self._semantic_search = None
        self._initialized = False
        
        logger.info("MemoryStore initialized with configuration")
    
    def initialize(self) -> None:
        """Initialize all components of the memory store.
        
        Raises:
            MemoryStoreError: If initialization fails
        """
        try:
            # Initialize embedding provider
            self._embedding_provider = self._create_embedding_provider()
            
            # Initialize message repository
            self._message_repository = MessageRepository(self.config)
            self._message_repository.initialize()
            
            # Initialize semantic search
            self._semantic_search = SemanticSearch(
                embedding_provider=self._embedding_provider,
                message_repository=self._message_repository,
                config=self.config
            )
            
            self._initialized = True
            logger.info("MemoryStore initialization completed successfully")
            
        except Exception as e:
            logger.error(f"MemoryStore initialization failed: {e}")
            raise MemoryStoreError(f"Failed to initialize MemoryStore: {e}")
    
    def _create_embedding_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider based on configuration.
        
        Returns:
            Configured embedding provider
            
        Raises:
            MemoryStoreError: If provider creation fails
        """
        try:
            if self.config.embedding_provider == "openai":
                if not self.config.has_api_key("openai"):
                    raise MemoryStoreError("OpenAI API key not found in configuration")
                
                # Lazy import to avoid dependency issues during testing
                from ..embeddings.openai_provider import OpenAIEmbeddingProvider
                return OpenAIEmbeddingProvider(
                    api_key=self.config.openai_api_key,
                    model=self.config.embedding_model
                )
            
            elif self.config.embedding_provider == "huggingface":
                # Lazy import to avoid dependency issues during testing
                from ..embeddings.huggingface_provider import HuggingFaceEmbeddingProvider
                return HuggingFaceEmbeddingProvider(
                    model_name=self.config.huggingface_model_path,
                    cache_dir=self.config.huggingface_cache_dir
                )
            
            else:
                raise MemoryStoreError(f"Unknown embedding provider: {self.config.embedding_provider}")
                
        except Exception as e:
            logger.error(f"Failed to create embedding provider: {e}")
            raise MemoryStoreError(f"Failed to create embedding provider: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure MemoryStore is initialized before operations.
        
        Raises:
            MemoryStoreError: If not initialized
        """
        if not self._initialized:
            raise MemoryStoreError("MemoryStore not initialized. Call initialize() first.")
    
    def store_message(
        self,
        content: str,
        role: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True
    ) -> str:
        """Store a message with optional embedding generation.
        
        Args:
            content: Message content
            role: Message role ('user' or 'assistant')
            model: Model name that generated/received the message
            metadata: Optional metadata dictionary
            generate_embedding: Whether to generate embeddings for semantic search
            
        Returns:
            Message ID of the stored message
            
        Raises:
            MemoryStoreError: If storing message fails
        """
        self._ensure_initialized()
        
        try:
            # Create message object
            message = Message(
                content=content,
                role=role,
                model=model,
                metadata=metadata or {}
            )
            
            if generate_embedding:
                # Store with semantic search (includes database storage)
                message_id = self._semantic_search.add_message(message)
            else:
                # Store only in database
                message_id = self._message_repository.store_message(message)
            
            logger.debug(f"Stored message {message_id} (embedding: {generate_embedding})")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            raise MemoryStoreError(f"Failed to store message: {e}")
    
    def get_recent(self, n: int = None) -> List[Message]:
        """Retrieve recent messages in chronological order.
        
        Args:
            n: Number of recent messages to retrieve. If None, uses config default.
            
        Returns:
            List of Message objects ordered chronologically (oldest first)
            
        Raises:
            MemoryStoreError: If retrieval fails
        """
        self._ensure_initialized()
        
        try:
            if n is None:
                n = self.config.default_recent_count
            
            if n <= 0:
                raise ValueError("Number of messages must be positive")
            
            messages = self._message_repository.get_recent_messages(n)
            logger.debug(f"Retrieved {len(messages)} recent messages")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            raise MemoryStoreError(f"Failed to get recent messages: {e}")
    
    def get_relevant(
        self,
        query: str,
        k: int = None,
        strategy: str = 'hybrid',
        score_threshold: float = 0.1
    ) -> List[Message]:
        """Retrieve relevant messages using semantic search.
        
        Args:
            query: Search query text
            k: Number of results to return. If None, uses config default.
            strategy: Search strategy ('semantic', 'recent', or 'hybrid')
            score_threshold: Minimum similarity score for semantic results
            
        Returns:
            List of Message objects ordered by relevance
            
        Raises:
            MemoryStoreError: If search fails
        """
        self._ensure_initialized()
        
        try:
            if k is None:
                k = self.config.default_relevant_count
            
            if k <= 0:
                raise ValueError("Number of results must be positive")
            
            # Perform search
            search_results = self._semantic_search.search(
                query=query,
                k=k,
                strategy=strategy,
                score_threshold=score_threshold
            )
            
            # Extract messages from search results
            messages = [result.message for result in search_results]
            
            logger.debug(f"Retrieved {len(messages)} relevant messages using {strategy} strategy")
            return messages
            
        except SemanticSearchError as e:
            logger.error(f"Semantic search failed: {e}")
            raise MemoryStoreError(f"Failed to get relevant messages: {e}")
        except Exception as e:
            logger.error(f"Failed to get relevant messages: {e}")
            raise MemoryStoreError(f"Failed to get relevant messages: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            MemoryStoreError: If embedding generation fails
        """
        self._ensure_initialized()
        
        try:
            embedding = self._embedding_provider.generate_embedding(text)
            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
            
        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise MemoryStoreError(f"Failed to generate embedding: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise MemoryStoreError(f"Failed to generate embedding: {e}")
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieve a specific message by ID.
        
        Args:
            message_id: ID of the message to retrieve
            
        Returns:
            Message object if found, None otherwise
            
        Raises:
            MemoryStoreError: If retrieval fails
        """
        self._ensure_initialized()
        
        try:
            message = self._message_repository.get_message_by_id(message_id)
            if message:
                logger.debug(f"Retrieved message {message_id}")
            else:
                logger.debug(f"Message {message_id} not found")
            return message
            
        except Exception as e:
            logger.error(f"Failed to get message by ID: {e}")
            raise MemoryStoreError(f"Failed to get message by ID: {e}")
    
    def get_messages_by_model(self, model: str, count: Optional[int] = None) -> List[Message]:
        """Retrieve messages filtered by model name.
        
        Args:
            model: Model name to filter by
            count: Optional limit on number of messages
            
        Returns:
            List of Message objects from the specified model
            
        Raises:
            MemoryStoreError: If retrieval fails
        """
        self._ensure_initialized()
        
        try:
            messages = self._message_repository.get_messages_by_model(model, count)
            logger.debug(f"Retrieved {len(messages)} messages for model {model}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages by model: {e}")
            raise MemoryStoreError(f"Failed to get messages by model: {e}")
    
    def get_messages_by_role(self, role: str, count: Optional[int] = None) -> List[Message]:
        """Retrieve messages filtered by role.
        
        Args:
            role: Role to filter by ('user' or 'assistant')
            count: Optional limit on number of messages
            
        Returns:
            List of Message objects with the specified role
            
        Raises:
            MemoryStoreError: If retrieval fails
        """
        self._ensure_initialized()
        
        try:
            messages = self._message_repository.get_messages_by_role(role, count)
            logger.debug(f"Retrieved {len(messages)} messages for role {role}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages by role: {e}")
            raise MemoryStoreError(f"Failed to get messages by role: {e}")
    
    def clear_memory(self, confirm: bool = False) -> None:
        """Clear all stored messages and embeddings.
        
        Args:
            confirm: Must be True to actually clear data (safety measure)
            
        Raises:
            MemoryStoreError: If clearing fails
        """
        self._ensure_initialized()
        
        if not confirm:
            raise MemoryStoreError("Must set confirm=True to clear all memory data")
        
        try:
            # Clear vector store
            self._semantic_search.vector_store.clear()
            
            # Clear database (we'll need to implement this in the repository)
            # For now, we'll delete all messages individually
            # In a production system, you'd want a more efficient bulk delete
            
            logger.warning("Memory cleared - all messages and embeddings deleted")
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            raise MemoryStoreError(f"Failed to clear memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory store.
        
        Returns:
            Dictionary containing various statistics
            
        Raises:
            MemoryStoreError: If getting stats fails
        """
        self._ensure_initialized()
        
        try:
            # Get message repository stats
            total_messages = self._message_repository.get_message_count()
            
            # Get search stats
            search_stats = self._semantic_search.get_search_stats()
            
            # Get database health
            db_health = self._message_repository.db_manager.health_check()
            
            stats = {
                'total_messages': total_messages,
                'embedding_provider': self.config.embedding_provider,
                'embedding_model': self.config.embedding_model,
                'database_health': db_health,
                'search_statistics': search_stats,
                'configuration': self.config.to_dict()
            }
            
            logger.debug("Retrieved memory store statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise MemoryStoreError(f"Failed to get stats: {e}")
    
    def rebuild_vector_index(self) -> None:
        """Rebuild the vector index from stored messages.
        
        This is useful for recovering from corruption or changing embedding providers.
        
        Raises:
            MemoryStoreError: If rebuild fails
        """
        self._ensure_initialized()
        
        try:
            self._semantic_search.rebuild_vector_index()
            logger.info("Vector index rebuild completed")
            
        except SemanticSearchError as e:
            logger.error(f"Vector index rebuild failed: {e}")
            raise MemoryStoreError(f"Failed to rebuild vector index: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during vector index rebuild: {e}")
            raise MemoryStoreError(f"Failed to rebuild vector index: {e}")
    
    def save_index(self) -> None:
        """Save the vector index to disk.
        
        Raises:
            MemoryStoreError: If saving fails
        """
        self._ensure_initialized()
        
        try:
            self._semantic_search.save_index()
            logger.debug("Vector index saved to disk")
            
        except SemanticSearchError as e:
            logger.error(f"Failed to save vector index: {e}")
            raise MemoryStoreError(f"Failed to save vector index: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving vector index: {e}")
            raise MemoryStoreError(f"Failed to save vector index: {e}")
    
    def backup_database(self, backup_path: str) -> None:
        """Create a backup of the database.
        
        Args:
            backup_path: Path where to save the backup
            
        Raises:
            MemoryStoreError: If backup fails
        """
        self._ensure_initialized()
        
        try:
            self._message_repository.db_manager.backup_database(backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise MemoryStoreError(f"Failed to backup database: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def _clear_all_data(self) -> None:
        """Clear all data from memory store (for testing purposes).
        
        This method is intended for testing and will remove all messages
        and embeddings from the store.
        
        Raises:
            MemoryStoreError: If clearing fails
        """
        self._ensure_initialized()
        
        try:
            # Clear vector store
            if hasattr(self._semantic_search, 'vector_store'):
                self._semantic_search.vector_store.clear()
            
            # Clear all messages from database
            if hasattr(self._message_repository, 'clear_all_messages'):
                self._message_repository.clear_all_messages()
            else:
                # Fallback: delete database file and reinitialize
                import os
                if os.path.exists(self.config.database_path):
                    os.unlink(self.config.database_path)
                self._message_repository.initialize()
            
            logger.debug("All data cleared from memory store")
            
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            raise MemoryStoreError(f"Failed to clear all data: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save index on exit."""
        if self._initialized:
            try:
                self.save_index()
            except Exception as e:
                logger.warning(f"Failed to save index on exit: {e}")