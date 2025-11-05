"""Semantic search functionality with fallback to recent messages."""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..models.message import Message
from ..models.config import MemoryConfig
from ..embeddings.base import BaseEmbeddingProvider, EmbeddingError
from .vector_store import VectorStore, VectorStoreError
from .message_repository import MessageRepository

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with message and relevance score."""
    message: Message
    score: float
    search_type: str  # 'semantic', 'recent', or 'hybrid'


class SemanticSearchError(Exception):
    """Exception raised when semantic search operations fail."""
    pass


class SemanticSearch:
    """Semantic search with fallback to recent messages."""
    
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        message_repository: MessageRepository,
        config: MemoryConfig
    ):
        """Initialize semantic search.
        
        Args:
            embedding_provider: Provider for generating embeddings
            message_repository: Repository for message storage/retrieval
            config: Memory configuration
        """
        self.embedding_provider = embedding_provider
        self.message_repository = message_repository
        self.config = config
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_provider=embedding_provider,
            index_path=config.vector_store_path,
            metadata_path=config.vector_store_path.replace('.faiss', '_metadata.pkl')
        )
        
        self._search_stats = {
            'semantic_searches': 0,
            'fallback_searches': 0,
            'failed_searches': 0
        }
    
    def add_message(self, message: Message) -> str:
        """Add a message to both database and vector store.
        
        Args:
            message: Message to add
            
        Returns:
            Message ID
            
        Raises:
            SemanticSearchError: If adding message fails
        """
        try:
            # Store in database first
            message_id = self.message_repository.store_message(message)
            
            # Try to add to vector store
            try:
                embedding_id = self.vector_store.add_message(message)
                
                # Update message with embedding ID
                self.message_repository.update_message_embedding_id(message_id, embedding_id)
                
                logger.debug(f"Added message {message_id} to both database and vector store")
                
            except VectorStoreError as e:
                logger.warning(f"Failed to add message to vector store, but stored in database: {e}")
                # Message is still stored in database, so this is not a complete failure
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise SemanticSearchError(f"Failed to add message: {e}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        strategy: str = 'hybrid',
        score_threshold: float = 0.1,
        fallback_count: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for relevant messages with fallback strategies.
        
        Args:
            query: Search query text
            k: Number of results to return
            strategy: Search strategy ('semantic', 'recent', or 'hybrid')
            score_threshold: Minimum similarity score for semantic results
            fallback_count: Number of recent messages for fallback (defaults to k)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            SemanticSearchError: If search fails completely
        """
        if fallback_count is None:
            fallback_count = k
        
        try:
            if strategy == 'semantic':
                return self._semantic_search(query, k, score_threshold, fallback_count)
            elif strategy == 'recent':
                return self._recent_search(fallback_count)
            elif strategy == 'hybrid':
                return self._hybrid_search(query, k, score_threshold, fallback_count)
            else:
                raise ValueError(f"Unknown search strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self._search_stats['failed_searches'] += 1
            raise SemanticSearchError(f"Search failed: {e}")
    
    def _semantic_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        fallback_count: int
    ) -> List[SearchResult]:
        """Perform semantic search with fallback to recent messages.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score
            fallback_count: Number of recent messages for fallback
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Try semantic search first
            vector_results = self.vector_store.search(query, k, score_threshold)
            
            if vector_results:
                # Convert vector results to SearchResult objects
                search_results = []
                for metadata, score in vector_results:
                    if not metadata.get('removed', False):  # Skip removed messages
                        message = self._metadata_to_message(metadata)
                        if message:
                            search_results.append(SearchResult(
                                message=message,
                                score=score,
                                search_type='semantic'
                            ))
                
                if search_results:
                    self._search_stats['semantic_searches'] += 1
                    logger.debug(f"Semantic search returned {len(search_results)} results")
                    return search_results
            
            # Fallback to recent messages
            logger.debug("Semantic search returned no results, falling back to recent messages")
            return self._recent_search_fallback(fallback_count)
            
        except (VectorStoreError, EmbeddingError) as e:
            logger.warning(f"Semantic search failed, falling back to recent messages: {e}")
            return self._recent_search_fallback(fallback_count)
    
    def _recent_search(self, count: int) -> List[SearchResult]:
        """Perform recent message search.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            recent_messages = self.message_repository.get_recent_messages(count)
            
            results = []
            for i, message in enumerate(recent_messages):
                # Assign decreasing scores based on recency (most recent = highest score)
                score = 1.0 - (i / len(recent_messages)) * 0.5  # Score range: 0.5 to 1.0
                results.append(SearchResult(
                    message=message,
                    score=score,
                    search_type='recent'
                ))
            
            logger.debug(f"Recent search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Recent search failed: {e}")
            raise SemanticSearchError(f"Recent search failed: {e}")
    
    def _recent_search_fallback(self, count: int) -> List[SearchResult]:
        """Fallback to recent messages search.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of SearchResult objects
        """
        self._search_stats['fallback_searches'] += 1
        return self._recent_search(count)
    
    def _hybrid_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        fallback_count: int
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and recent results.
        
        Args:
            query: Search query text
            k: Total number of results to return
            score_threshold: Minimum similarity score for semantic results
            fallback_count: Number of recent messages for fallback
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Get semantic results (up to k/2)
            semantic_k = max(1, k // 2)
            semantic_results = []
            
            try:
                vector_results = self.vector_store.search(query, semantic_k, score_threshold)
                
                for metadata, score in vector_results:
                    if not metadata.get('removed', False):
                        message = self._metadata_to_message(metadata)
                        if message:
                            semantic_results.append(SearchResult(
                                message=message,
                                score=score,
                                search_type='semantic'
                            ))
            
            except (VectorStoreError, EmbeddingError) as e:
                logger.warning(f"Semantic part of hybrid search failed: {e}")
            
            # Get recent results (remaining slots)
            recent_k = k - len(semantic_results)
            recent_results = []
            
            if recent_k > 0:
                try:
                    recent_messages = self.message_repository.get_recent_messages(recent_k * 2)  # Get more to filter duplicates
                    
                    # Filter out messages already in semantic results
                    semantic_ids = {result.message.id for result in semantic_results}
                    
                    for i, message in enumerate(recent_messages):
                        if message.id not in semantic_ids and len(recent_results) < recent_k:
                            score = 0.8 - (i / len(recent_messages)) * 0.3  # Score range: 0.5 to 0.8
                            recent_results.append(SearchResult(
                                message=message,
                                score=score,
                                search_type='recent'
                            ))
                
                except Exception as e:
                    logger.warning(f"Recent part of hybrid search failed: {e}")
            
            # Combine and sort by score
            all_results = semantic_results + recent_results
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Mark as hybrid search type
            for result in all_results:
                result.search_type = 'hybrid'
            
            if semantic_results:
                self._search_stats['semantic_searches'] += 1
            else:
                self._search_stats['fallback_searches'] += 1
            
            logger.debug(f"Hybrid search returned {len(all_results)} results ({len(semantic_results)} semantic, {len(recent_results)} recent)")
            return all_results[:k]  # Return only requested number
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Final fallback to recent search
            return self._recent_search_fallback(fallback_count)
    
    def _metadata_to_message(self, metadata: Dict[str, Any]) -> Optional[Message]:
        """Convert vector store metadata to Message object.
        
        Args:
            metadata: Metadata dictionary from vector store
            
        Returns:
            Message object or None if conversion fails
        """
        try:
            # Try to get full message from database first (more reliable)
            message_id = metadata.get('message_id')
            if message_id:
                message = self.message_repository.get_message_by_id(message_id)
                if message:
                    return message
            
            # Fallback to reconstructing from metadata
            from datetime import datetime
            return Message(
                id=metadata['message_id'],
                content=metadata['content'],
                role=metadata['role'],
                model=metadata['model'],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                embedding_id=metadata.get('embedding_id'),
                metadata=metadata.get('metadata', {})
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert metadata to message: {e}")
            return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        return {
            'search_stats': self._search_stats.copy(),
            'vector_store_stats': vector_stats,
            'total_searches': sum(self._search_stats.values())
        }
    
    def rebuild_vector_index(self) -> None:
        """Rebuild the vector index from database messages.
        
        This is useful for recovering from vector store corruption or
        when switching embedding providers.
        """
        try:
            logger.info("Starting vector index rebuild from database")
            
            # Clear existing vector store
            self.vector_store.clear()
            
            # Get all messages from database
            # We'll process in batches to avoid memory issues
            batch_size = 100
            processed = 0
            
            # Get total count for progress tracking
            total_messages = self.message_repository.get_message_count()
            logger.info(f"Rebuilding index for {total_messages} messages")
            
            # Process messages in batches
            offset = 0
            while True:
                # Get batch of messages (we'll use recent messages as a proxy for pagination)
                # In a real implementation, you'd want proper pagination in the repository
                try:
                    messages = self.message_repository.get_recent_messages(batch_size)
                    if not messages:
                        break
                    
                    # Process each message
                    for message in messages:
                        try:
                            embedding_id = self.vector_store.add_message(message)
                            self.message_repository.update_message_embedding_id(message.id, embedding_id)
                            processed += 1
                            
                            if processed % 10 == 0:
                                logger.debug(f"Processed {processed}/{total_messages} messages")
                                
                        except Exception as e:
                            logger.warning(f"Failed to process message {message.id}: {e}")
                    
                    offset += batch_size
                    
                    # Break if we got fewer messages than requested (end of data)
                    if len(messages) < batch_size:
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to get message batch at offset {offset}: {e}")
                    break
            
            # Save the rebuilt index
            self.vector_store.save_index()
            
            logger.info(f"Vector index rebuild completed. Processed {processed} messages")
            
        except Exception as e:
            logger.error(f"Vector index rebuild failed: {e}")
            raise SemanticSearchError(f"Vector index rebuild failed: {e}")
    
    def save_index(self) -> None:
        """Save the vector index to disk."""
        try:
            self.vector_store.save_index()
            logger.debug("Vector index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            raise SemanticSearchError(f"Failed to save vector index: {e}")