"""Base interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding provider implementations."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return embedding dimension for vector store setup.
        
        Returns:
            The dimension of embeddings produced by this provider
        """
        pass
    
    def validate_text(self, text: str) -> None:
        """
        Validate input text before generating embeddings.
        
        Args:
            text: The text to validate
            
        Raises:
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        if len(text) > 8000:  # Common limit for embedding models
            raise ValueError("Text is too long for embedding generation")