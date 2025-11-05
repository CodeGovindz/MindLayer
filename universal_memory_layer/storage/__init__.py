"""Storage components for the Universal Memory Layer."""

from .database import DatabaseManager
from .message_repository import MessageRepository
from .memory_store import MemoryStore, MemoryStoreError

# Import vector store components only if dependencies are available
try:
    from .vector_store import VectorStore, VectorStoreError
    from .semantic_search import SemanticSearch, SearchResult, SemanticSearchError
    VECTOR_STORE_AVAILABLE = True
except (ImportError, TypeError, AttributeError):
    # Vector store not available due to missing dependencies
    VECTOR_STORE_AVAILABLE = False

__all__ = [
    "DatabaseManager", 
    "MessageRepository",
    "MemoryStore",
    "MemoryStoreError"
]

if VECTOR_STORE_AVAILABLE:
    __all__.extend([
        "VectorStore",
        "VectorStoreError", 
        "SemanticSearch",
        "SearchResult",
        "SemanticSearchError"
    ])