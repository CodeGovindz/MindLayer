"""Error handling and custom exceptions for Universal Memory Layer."""

import functools
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from .logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class UMLError(Exception):
    """Base exception for Universal Memory Layer errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize UML error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class ConfigurationError(UMLError):
    """Raised when there's a configuration-related error."""
    pass


class StorageError(UMLError):
    """Raised when there's a storage-related error."""
    pass


class DatabaseError(StorageError):
    """Raised when there's a database-related error."""
    pass


class VectorStoreError(StorageError):
    """Raised when there's a vector store-related error."""
    pass


class EmbeddingError(UMLError):
    """Raised when there's an embedding-related error."""
    pass


class LLMClientError(UMLError):
    """Raised when there's an LLM client-related error."""
    pass


class APIError(LLMClientError):
    """Raised when there's an API-related error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, provider: Optional[str] = None, **kwargs):
        """Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code if applicable
            provider: LLM provider name
            **kwargs: Additional error details
        """
        details = kwargs
        if status_code:
            details['status_code'] = status_code
        if provider:
            details['provider'] = provider
        
        super().__init__(message, details=details)
        self.status_code = status_code
        self.provider = provider


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, provider: Optional[str] = None, **kwargs):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            provider: LLM provider name
            **kwargs: Additional error details
        """
        super().__init__(message, provider=provider, **kwargs)
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.retry_after = retry_after
        
        # Add retry_after to details
        if retry_after:
            self.details['retry_after'] = retry_after


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        """Initialize authentication error.
        
        Args:
            message: Error message
            provider: LLM provider name
        """
        super().__init__(message, status_code=401, provider=provider)
        self.error_code = "AUTHENTICATION_FAILED"


class ValidationError(UMLError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field name that failed validation
            value: Invalid value
        """
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        
        super().__init__(message, error_code="VALIDATION_FAILED", details=details)
        self.field = field
        self.value = value


class ConversationError(UMLError):
    """Raised when there's a conversation management error."""
    pass


def handle_errors(
    default_return: Any = None,
    reraise: bool = True,
    log_errors: bool = True,
    error_mapping: Optional[Dict[Type[Exception], Type[UMLError]]] = None
) -> Callable[[F], F]:
    """Decorator for comprehensive error handling.
    
    Args:
        default_return: Default value to return on error (if not reraising)
        reraise: Whether to reraise exceptions after handling
        log_errors: Whether to log errors
        error_mapping: Mapping of exception types to UML error types
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UMLError:
                # UML errors are already properly formatted, just reraise
                if log_errors:
                    logger.error(f"UML error in {func.__name__}: {traceback.format_exc()}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # Convert other exceptions to UML errors if mapping provided
                if error_mapping and type(e) in error_mapping:
                    uml_error_class = error_mapping[type(e)]
                    # Handle different error class constructors
                    try:
                        uml_error = uml_error_class(
                            f"Error in {func.__name__}: {str(e)}",
                            details={'original_error': str(e), 'function': func.__name__}
                        )
                    except TypeError:
                        # Fallback for errors that don't accept details parameter
                        uml_error = uml_error_class(f"Error in {func.__name__}: {str(e)}")
                    if log_errors:
                        logger.error(f"Converted error in {func.__name__}: {traceback.format_exc()}")
                    if reraise:
                        raise uml_error from e
                    return default_return
                else:
                    # Log and optionally reraise original exception
                    if log_errors:
                        logger.error(f"Unhandled error in {func.__name__}: {traceback.format_exc()}")
                    if reraise:
                        raise
                    return default_return
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    error_message: Optional[str] = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        log_errors: Whether to log errors
        error_message: Custom error message prefix
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            prefix = error_message or f"Error executing {func.__name__}"
            logger.error(f"{prefix}: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        return default_return


def create_user_friendly_message(error: Exception) -> str:
    """Create user-friendly error message from exception.
    
    Args:
        error: Exception to convert
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, AuthenticationError):
        provider = error.provider or "the LLM provider"
        return f"Authentication failed for {provider}. Please check your API key configuration."
    
    elif isinstance(error, RateLimitError):
        provider = error.provider or "the API"
        retry_msg = f" Please wait {error.retry_after} seconds before retrying." if error.retry_after else ""
        return f"Rate limit exceeded for {provider}.{retry_msg}"
    
    elif isinstance(error, APIError):
        provider = error.provider or "the API"
        return f"API error from {provider}: {error.message}"
    
    elif isinstance(error, ConfigurationError):
        return f"Configuration error: {error.message}"
    
    elif isinstance(error, DatabaseError):
        return f"Database error: {error.message}"
    
    elif isinstance(error, VectorStoreError):
        return f"Vector store error: {error.message}"
    
    elif isinstance(error, EmbeddingError):
        return f"Embedding error: {error.message}"
    
    elif isinstance(error, ValidationError):
        field_info = f" (field: {error.field})" if error.field else ""
        return f"Validation error{field_info}: {error.message}"
    
    elif isinstance(error, UMLError):
        return error.message
    
    else:
        return f"An unexpected error occurred: {str(error)}"


def log_and_raise(
    error_class: Type[UMLError],
    message: str,
    logger_instance: Optional[Any] = None,
    **kwargs
) -> None:
    """Log an error and raise the specified exception.
    
    Args:
        error_class: Exception class to raise
        message: Error message
        logger_instance: Logger to use (defaults to module logger)
        **kwargs: Additional arguments for the exception
    """
    log = logger_instance or logger
    log.error(message)
    raise error_class(message, **kwargs)


# Import sqlite3 for error mapping
try:
    import sqlite3
    SQLITE3_AVAILABLE = True
except ImportError:
    SQLITE3_AVAILABLE = False

# Common error mappings for different modules
STORAGE_ERROR_MAPPING = {
    FileNotFoundError: StorageError,
    PermissionError: StorageError,
    OSError: StorageError,
}

if SQLITE3_AVAILABLE:
    STORAGE_ERROR_MAPPING[sqlite3.Error] = DatabaseError

API_ERROR_MAPPING = {
    ConnectionError: APIError,
    TimeoutError: APIError,
    ValueError: ValidationError,
}