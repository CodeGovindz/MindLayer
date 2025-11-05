"""Hugging Face embedding provider implementation."""

import os
from typing import List, Optional
import logging

from .base import BaseEmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    # Create dummy classes for type hints when not available
    class SentenceTransformer:
        pass


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Hugging Face embedding provider using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize Hugging Face embedding provider.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache downloaded models. If None, uses default
            device: Device to run model on ('cpu', 'cuda', etc.). If None, auto-detects
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "sentence-transformers and torch packages are required for Hugging Face embeddings. "
                "Install with: pip install sentence-transformers torch"
            )
        
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/sentence_transformers")
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._model = None
        self._dimension = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                logger.info(f"Loading Hugging Face model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device
                )
                logger.info(f"Model loaded successfully on device: {self.device}")
            except Exception as e:
                raise EmbeddingError(f"Failed to load Hugging Face model {self.model_name}: {e}")
        
        return self._model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text using Hugging Face model.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text length {len(text)}")
            return embedding
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate Hugging Face embedding: {e}")
    
    def get_dimension(self) -> int:
        """Return embedding dimension for the loaded model.
        
        Returns:
            Integer representing the embedding dimension
        """
        if self._dimension is None:
            try:
                # Get dimension from model
                self._dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Could not determine dimension for model {self.model_name}: {e}")
                # Fallback: generate a test embedding to get dimension
                try:
                    test_embedding = self.generate_embedding("test")
                    self._dimension = len(test_embedding)
                except Exception:
                    # Last resort fallback
                    self._dimension = 384  # Common dimension for many sentence-transformers models
        
        return self._dimension
    
    def download_model(self) -> None:
        """Explicitly download and cache the model.
        
        This can be useful for pre-downloading models or checking if download works.
        """
        try:
            logger.info(f"Downloading model {self.model_name} to {self.cache_dir}")
            # Accessing the model property will trigger download if needed
            _ = self.model
            logger.info("Model download completed successfully")
        except Exception as e:
            raise EmbeddingError(f"Failed to download model {self.model_name}: {e}")
    
    def clear_cache(self) -> None:
        """Clear the model from memory to free up resources."""
        if self._model is not None:
            logger.info("Clearing Hugging Face model from memory")
            del self._model
            self._model = None
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()