"""Base embedding provider interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension for vector store setup.
        
        Returns:
            Integer representing the embedding dimension
        """
        pass


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass