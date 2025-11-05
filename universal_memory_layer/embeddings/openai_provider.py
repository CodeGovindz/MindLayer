"""OpenAI embedding provider implementation."""

import time
from typing import List, Optional
import logging

from .base import BaseEmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var
            model: OpenAI embedding model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI embeddings")
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = openai.OpenAI()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text using OpenAI API.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails after retries
        """
        if not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                if not response.data:
                    raise EmbeddingError("No embedding data returned from OpenAI API")
                
                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding of dimension {len(embedding)} for text length {len(text)}")
                return embedding
                
            except openai.RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(f"Rate limit exceeded after {self.max_retries} retries: {e}")
            
            except openai.AuthenticationError as e:
                raise EmbeddingError(f"OpenAI authentication failed: {e}")
            
            except openai.APIError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API error, retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(f"OpenAI API error after {self.max_retries} retries: {e}")
            
            except Exception as e:
                raise EmbeddingError(f"Unexpected error generating OpenAI embedding: {e}")
    
    def get_dimension(self) -> int:
        """Return embedding dimension for the configured model.
        
        Returns:
            Integer representing the embedding dimension
        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        if self.model in model_dimensions:
            return model_dimensions[self.model]
        
        # For unknown models, we'll need to make a test call
        # This is a fallback that should rarely be used
        try:
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Could not determine dimension for model {self.model}: {e}")
            # Default to ada-002 dimension as fallback
            return 1536