"""Message repository for database operations."""

import sqlite3
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from ..models.message import Message
from ..models.config import MemoryConfig
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class MessageRepository:
    """Repository for message storage and retrieval operations."""
    
    def __init__(self, config: MemoryConfig):
        """Initialize message repository with database manager."""
        self.config = config
        self.db_manager = DatabaseManager(config)
        
    def initialize(self) -> None:
        """Initialize the repository by setting up the database."""
        self.db_manager.initialize()
    
    def store_message(self, message: Message) -> str:
        """
        Store a message in the database.
        
        Args:
            message: Message object to store
            
        Returns:
            str: The ID of the stored message
            
        Raises:
            ValueError: If message validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate message before storing
            if not isinstance(message, Message):
                raise ValueError(f"Expected Message instance, got {type(message)}")
            
            with self.db_manager.get_connection() as conn:
                # Convert metadata to JSON string
                metadata_json = json.dumps(message.metadata) if message.metadata else None
                
                # Insert message into database
                conn.execute("""
                    INSERT INTO messages (
                        id, content, role, model, timestamp, embedding_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.content,
                    message.role,
                    message.model,
                    message.timestamp.isoformat(),
                    message.embedding_id,
                    metadata_json
                ))
                
                conn.commit()
                
            logger.debug(f"Stored message {message.id} from {message.role} using {message.model}")
            return message.id
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"Message {message.id} already exists in database")
                raise ValueError(f"Message with ID {message.id} already exists")
            else:
                logger.error(f"Database integrity error storing message: {e}")
                raise
        except Exception as e:
            # Handle case where message might not be a valid Message object
            message_id = getattr(message, 'id', 'unknown') if hasattr(message, 'id') else 'unknown'
            logger.error(f"Failed to store message {message_id}: {e}")
            raise
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Retrieve a message by its ID.
        
        Args:
            message_id: ID of the message to retrieve
            
        Returns:
            Message object if found, None otherwise
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, content, role, model, timestamp, embedding_id, metadata
                    FROM messages
                    WHERE id = ?
                """, (message_id,))
                
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_message(row)
                else:
                    logger.debug(f"Message {message_id} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve message {message_id}: {e}")
            raise
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """
        Retrieve the most recent messages in chronological order.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of Message objects ordered by timestamp (oldest first)
        """
        try:
            if count <= 0:
                raise ValueError("Count must be positive")
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, content, role, model, timestamp, embedding_id, metadata
                    FROM messages
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (count,))
                
                rows = cursor.fetchall()
                
                # Convert rows to messages and reverse to get chronological order
                messages = [self._row_to_message(row) for row in reversed(rows)]
                
            logger.debug(f"Retrieved {len(messages)} recent messages")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent messages: {e}")
            raise
    
    def get_messages_by_model(self, model: str, count: Optional[int] = None) -> List[Message]:
        """
        Retrieve messages filtered by model name.
        
        Args:
            model: Name of the model to filter by
            count: Optional limit on number of messages to retrieve
            
        Returns:
            List of Message objects ordered by timestamp (newest first)
        """
        try:
            with self.db_manager.get_connection() as conn:
                if count is not None:
                    if count <= 0:
                        raise ValueError("Count must be positive")
                    
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE model = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (model, count))
                else:
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE model = ?
                        ORDER BY timestamp DESC
                    """, (model,))
                
                rows = cursor.fetchall()
                messages = [self._row_to_message(row) for row in rows]
                
            logger.debug(f"Retrieved {len(messages)} messages for model {model}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve messages for model {model}: {e}")
            raise
    
    def get_messages_by_role(self, role: str, count: Optional[int] = None) -> List[Message]:
        """
        Retrieve messages filtered by role.
        
        Args:
            role: Role to filter by ('user' or 'assistant')
            count: Optional limit on number of messages to retrieve
            
        Returns:
            List of Message objects ordered by timestamp (newest first)
        """
        try:
            if role not in ['user', 'assistant']:
                raise ValueError(f"Role must be 'user' or 'assistant', got: {role}")
            
            with self.db_manager.get_connection() as conn:
                if count is not None:
                    if count <= 0:
                        raise ValueError("Count must be positive")
                    
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE role = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (role, count))
                else:
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE role = ?
                        ORDER BY timestamp DESC
                    """, (role,))
                
                rows = cursor.fetchall()
                messages = [self._row_to_message(row) for row in rows]
                
            logger.debug(f"Retrieved {len(messages)} messages for role {role}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve messages for role {role}: {e}")
            raise
    
    def get_messages_in_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        count: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve messages within a date range.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            count: Optional limit on number of messages to retrieve
            
        Returns:
            List of Message objects ordered by timestamp (newest first)
        """
        try:
            if start_date > end_date:
                raise ValueError("Start date must be before or equal to end date")
            
            with self.db_manager.get_connection() as conn:
                if count is not None:
                    if count <= 0:
                        raise ValueError("Count must be positive")
                    
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (start_date.isoformat(), end_date.isoformat(), count))
                else:
                    cursor = conn.execute("""
                        SELECT id, content, role, model, timestamp, embedding_id, metadata
                        FROM messages
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                    """, (start_date.isoformat(), end_date.isoformat()))
                
                rows = cursor.fetchall()
                messages = [self._row_to_message(row) for row in rows]
                
            logger.debug(f"Retrieved {len(messages)} messages in date range")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve messages in date range: {e}")
            raise
    
    def update_message_embedding_id(self, message_id: str, embedding_id: str) -> bool:
        """
        Update the embedding ID for a message.
        
        Args:
            message_id: ID of the message to update
            embedding_id: New embedding ID to set
            
        Returns:
            bool: True if message was updated, False if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE messages
                    SET embedding_id = ?
                    WHERE id = ?
                """, (embedding_id, message_id))
                
                conn.commit()
                
                updated = cursor.rowcount > 0
                if updated:
                    logger.debug(f"Updated embedding ID for message {message_id}")
                else:
                    logger.warning(f"Message {message_id} not found for embedding update")
                
                return updated
                
        except Exception as e:
            logger.error(f"Failed to update embedding ID for message {message_id}: {e}")
            raise
    
    def delete_message(self, message_id: str) -> bool:
        """
        Delete a message from the database.
        
        Args:
            message_id: ID of the message to delete
            
        Returns:
            bool: True if message was deleted, False if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM messages
                    WHERE id = ?
                """, (message_id,))
                
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted message {message_id}")
                else:
                    logger.warning(f"Message {message_id} not found for deletion")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete message {message_id}: {e}")
            raise
    
    def get_message_count(self) -> int:
        """
        Get the total number of messages in the database.
        
        Returns:
            int: Total message count
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                count = cursor.fetchone()[0]
                
            logger.debug(f"Total message count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to get message count: {e}")
            raise
    
    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """
        Convert database row to Message object.
        
        Args:
            row: Database row from messages table
            
        Returns:
            Message object
        """
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(row['timestamp'])
            
            # Parse metadata JSON
            metadata = {}
            if row['metadata']:
                metadata = json.loads(row['metadata'])
            
            return Message(
                id=row['id'],
                content=row['content'],
                role=row['role'],
                model=row['model'],
                timestamp=timestamp,
                embedding_id=row['embedding_id'],
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert row to message: {e}")
            raise ValueError(f"Invalid message data in database: {e}")