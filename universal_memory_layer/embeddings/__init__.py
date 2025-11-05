"""Embedding providers for the Universal Memory Layer."""

from .base import BaseEmbeddingProvider, EmbeddingError
from .openai_provider import OpenAIEmbeddingProvider

__all__ = [
    'BaseEmbeddingProvider',
    'EmbeddingError',
    'OpenAIEmbeddingProvider',
]

# Import HuggingFace provider only if dependencies are available
try:
    from .huggingface_provider import HuggingFaceEmbeddingProvider
    __all__.append('HuggingFaceEmbeddingProvider')
except (ImportError, TypeError, AttributeError) as e:
    # HuggingFace provider not available due to missing dependencies or compatibility issues
    # This is expected in environments without sentence-transformers or with numpy compatibility issues
    pass