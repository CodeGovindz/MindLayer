"""FAISS-based vector store for semantic search."""

import os
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..models.message import Message
from ..embeddings.base import BaseEmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStoreError(Exception):
    """Exception raised when vector store operations fail."""
    pass


class VectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        index_path: str = "vector_store.faiss",
        metadata_path: str = "vector_metadata.pkl"
    ):
        """Initialize vector store.
        
        Args:
            embedding_provider: Provider for generating embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata mapping
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu package is required for vector store functionality. "
                "Install with: pip install faiss-cpu"
            )
        
        self.embedding_provider = embedding_provider
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        self._index = None
        self._metadata = {}  # Maps vector index to message metadata
        self._dimension = None
        
        # Try to load existing index
        self._load_index()
    
    @property
    def index(self) -> 'faiss.Index':
        """Get or create FAISS index."""
        if self._index is None:
            self._create_index()
        return self._index
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._dimension = self.embedding_provider.get_dimension()
        return self._dimension
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        try:
            # Use IndexFlatIP for cosine similarity (after normalization)
            # This is good for most embedding use cases
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
        except Exception as e:
            raise VectorStoreError(f"Failed to create FAISS index: {e}")
    
    def _load_index(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # Load FAISS index
                self._index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self._metadata = pickle.load(f)
                logger.info(f"Loaded {len(self._metadata)} metadata entries")
                
                # Set dimension from loaded index
                self._dimension = self._index.d
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                self._index = None
                self._metadata = {}
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            if self._index is not None:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
                
                # Save FAISS index
                faiss.write_index(self._index, self.index_path)
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self._metadata, f)
                
                logger.info(f"Saved vector store with {self._index.ntotal} vectors")
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {e}")
    
    def add_message(self, message: Message) -> str:
        """Add a message to the vector store.
        
        Args:
            message: Message to add
            
        Returns:
            Embedding ID for the stored vector
            
        Raises:
            VectorStoreError: If adding message fails
        """
        try:
            # Generate embedding for message content
            embedding = self.embedding_provider.generate_embedding(message.content)
            
            # Convert to numpy array and normalize for cosine similarity
            embedding_array = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            
            # Add to index
            current_count = self.index.ntotal
            self.index.add(embedding_array)
            
            # Store metadata
            embedding_id = f"emb_{current_count}"
            self._metadata[current_count] = {
                'message_id': message.id,
                'embedding_id': embedding_id,
                'content': message.content,
                'role': message.role,
                'model': message.model,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            }
            
            logger.debug(f"Added message {message.id} to vector store with embedding ID {embedding_id}")
            return embedding_id
            
        except EmbeddingError as e:
            raise VectorStoreError(f"Failed to generate embedding for message: {e}")
        except Exception as e:
            raise VectorStoreError(f"Failed to add message to vector store: {e}")
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar messages.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of tuples containing (message_metadata, similarity_score)
            
        Raises:
            VectorStoreError: If search fails
        """
        if self._index is None or self._index.ntotal == 0:
            logger.debug("Vector store is empty, returning no results")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_provider.generate_embedding(query)
            
            # Convert to numpy array and normalize
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            k = min(k, self._index.ntotal)  # Don't search for more than available
            scores, indices = self._index.search(query_array, k)
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if score < score_threshold:
                    continue
                
                if idx in self._metadata:
                    results.append((self._metadata[idx], float(score)))
            
            logger.debug(f"Found {len(results)} similar messages for query")
            return results
            
        except EmbeddingError as e:
            raise VectorStoreError(f"Failed to generate embedding for query: {e}")
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")
    
    def remove_message(self, embedding_id: str) -> bool:
        """Remove a message from the vector store.
        
        Note: FAISS doesn't support efficient removal, so this marks as removed
        in metadata but doesn't actually remove from index.
        
        Args:
            embedding_id: ID of embedding to remove
            
        Returns:
            True if message was found and marked as removed
        """
        try:
            # Find the index for this embedding_id
            for idx, metadata in self._metadata.items():
                if metadata.get('embedding_id') == embedding_id:
                    # Mark as removed instead of actually removing
                    metadata['removed'] = True
                    logger.debug(f"Marked embedding {embedding_id} as removed")
                    return True
            
            logger.debug(f"Embedding {embedding_id} not found for removal")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove embedding {embedding_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        active_count = sum(1 for meta in self._metadata.values() if not meta.get('removed', False))
        
        return {
            'total_vectors': self._index.ntotal if self._index else 0,
            'active_vectors': active_count,
            'removed_vectors': len(self._metadata) - active_count,
            'dimension': self.dimension,
            'index_path': self.index_path,
            'metadata_path': self.metadata_path
        }
    
    def rebuild_index(self) -> None:
        """Rebuild the index to remove deleted entries.
        
        This creates a new index with only active (non-removed) vectors.
        """
        if not self._metadata:
            logger.info("No metadata to rebuild from")
            return
        
        try:
            logger.info("Starting index rebuild to remove deleted entries")
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            new_metadata = {}
            
            # Re-add only active vectors
            active_vectors = []
            active_metas = []
            
            for idx, metadata in self._metadata.items():
                if not metadata.get('removed', False):
                    # We need to regenerate embeddings since we can't extract from FAISS
                    try:
                        embedding = self.embedding_provider.generate_embedding(metadata['content'])
                        active_vectors.append(embedding)
                        active_metas.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to regenerate embedding for message {metadata.get('message_id')}: {e}")
            
            if active_vectors:
                # Add all vectors at once
                vectors_array = np.array(active_vectors, dtype=np.float32)
                faiss.normalize_L2(vectors_array)
                new_index.add(vectors_array)
                
                # Update metadata with new indices
                for i, metadata in enumerate(active_metas):
                    new_metadata[i] = metadata
            
            # Replace old index and metadata
            self._index = new_index
            self._metadata = new_metadata
            
            logger.info(f"Index rebuilt with {len(new_metadata)} active vectors")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to rebuild index: {e}")
    
    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self._index = None
        self._metadata = {}
        
        # Remove files if they exist
        for path in [self.index_path, self.metadata_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")
        
        logger.info("Vector store cleared")