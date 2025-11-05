"""Message data model for conversation storage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import json


@dataclass
class Message:
    """Represents a single message in a conversation."""
    
    content: str
    role: str  # 'user' or 'assistant'
    model: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message data after initialization."""
        self._validate_role()
        self._validate_content()
        self._validate_model()
        self._validate_id()
        self._validate_metadata()
    
    def _validate_role(self) -> None:
        """Validate the role field."""
        if not isinstance(self.role, str):
            raise TypeError(f"Role must be a string, got: {type(self.role)}")
        
        if self.role not in ['user', 'assistant']:
            raise ValueError(f"Role must be 'user' or 'assistant', got: {self.role}")
    
    def _validate_content(self) -> None:
        """Validate the content field."""
        if not isinstance(self.content, str):
            raise TypeError(f"Content must be a string, got: {type(self.content)}")
        
        if not self.content.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        
        if len(self.content) > 100000:  # 100KB limit
            raise ValueError("Message content exceeds maximum length of 100,000 characters")
    
    def _validate_model(self) -> None:
        """Validate the model field."""
        if not isinstance(self.model, str):
            raise TypeError(f"Model must be a string, got: {type(self.model)}")
        
        if not self.model.strip():
            raise ValueError("Model name cannot be empty or whitespace only")
    
    def _validate_id(self) -> None:
        """Validate the id field."""
        if not isinstance(self.id, str):
            raise TypeError(f"ID must be a string, got: {type(self.id)}")
        
        if not self.id.strip():
            raise ValueError("Message ID cannot be empty")
    
    def _validate_metadata(self) -> None:
        """Validate the metadata field."""
        if not isinstance(self.metadata, dict):
            raise TypeError(f"Metadata must be a dictionary, got: {type(self.metadata)}")
        
        # Ensure metadata is JSON serializable
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON serializable: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'role': self.role,
            'model': self.model,
            'timestamp': self.timestamp.isoformat(),
            'embedding_id': self.embedding_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        # Parse timestamp back to datetime
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            id=data['id'],
            content=data['content'],
            role=data['role'],
            model=data['model'],
            timestamp=timestamp,
            embedding_id=data.get('embedding_id'),
            metadata=data.get('metadata', {})
        )