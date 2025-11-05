"""Conversation data model for grouping related messages."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import uuid

from .message import Message


@dataclass
class Conversation:
    """Represents a conversation containing multiple messages."""
    
    title: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate conversation data after initialization."""
        self._validate_title()
        self._validate_id()
        self._validate_messages()
    
    def _validate_title(self) -> None:
        """Validate the title field."""
        if not isinstance(self.title, str):
            raise TypeError(f"Title must be a string, got: {type(self.title)}")
        
        if not self.title.strip():
            raise ValueError("Conversation title cannot be empty or whitespace only")
        
        if len(self.title) > 200:
            raise ValueError("Conversation title cannot exceed 200 characters")
    
    def _validate_id(self) -> None:
        """Validate the id field."""
        if not isinstance(self.id, str):
            raise TypeError(f"ID must be a string, got: {type(self.id)}")
        
        if not self.id.strip():
            raise ValueError("Conversation ID cannot be empty")
    
    def _validate_messages(self) -> None:
        """Validate the messages field."""
        if not isinstance(self.messages, list):
            raise TypeError(f"Messages must be a list, got: {type(self.messages)}")
        
        for i, message in enumerate(self.messages):
            if not isinstance(message, Message):
                raise TypeError(f"Message at index {i} must be a Message instance, got: {type(message)}")
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation and update timestamp."""
        if not isinstance(message, Message):
            raise TypeError(f"Expected Message instance, got: {type(message)}")
        
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_message_count(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.messages)
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages from the conversation."""
        if count <= 0:
            raise ValueError("Count must be positive")
        
        return self.messages[-count:] if self.messages else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'messages': [message.to_dict() for message in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary."""
        # Parse timestamps back to datetime
        created_at = datetime.fromisoformat(data['created_at'])
        last_updated = datetime.fromisoformat(data['last_updated'])
        
        # Convert message dictionaries back to Message objects
        messages = [Message.from_dict(msg_data) for msg_data in data.get('messages', [])]
        
        return cls(
            id=data['id'],
            title=data['title'],
            created_at=created_at,
            last_updated=last_updated,
            messages=messages
        )