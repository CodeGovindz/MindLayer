"""Base interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM client implementations."""
    
    def __init__(self, api_key: str):
        """Initialize the LLM client with API key."""
        self.api_key = api_key
    
    @abstractmethod
    def chat(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Send message with context to LLM and return response.
        
        Args:
            message: The user message to send
            context: Optional list of context messages to prepend
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If the API call fails
        """
        pass
    
    @abstractmethod
    def format_prompt(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Format message and context into model-specific prompt.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted prompt string for the specific model
        """
        pass
    
    def handle_error(self, error: Exception) -> str:
        """
        Handle API errors gracefully.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        return f"An error occurred while communicating with the LLM: {str(error)}"