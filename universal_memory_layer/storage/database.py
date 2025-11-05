"""SQLite database setup and management for Universal Memory Layer."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json
from datetime import datetime

from ..models.config import MemoryConfig


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for conversation storage."""
    
    def __init__(self, config: MemoryConfig):
        """Initialize database manager with configuration."""
        self.config = config
        self.db_path = config.database_path
        self._connection: Optional[sqlite3.Connection] = None
        
    def initialize(self) -> None:
        """Initialize database with schema creation and migrations."""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database and tables
            with self.get_connection() as conn:
                self._create_schema(conn)
                self._run_migrations(conn)
                
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling and cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # 30 second timeout
                check_same_thread=False
            )
            
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            yield conn
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected error in database connection: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema with all required tables."""
        
        # Create messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                model TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_id TEXT,
                metadata JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create message_conversations junction table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_conversations (
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                PRIMARY KEY (message_id, conversation_id),
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        
        # Create schema_version table for migrations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Create indexes for performance
        self._create_indexes(conn)
        
        conn.commit()
        logger.info("Database schema created successfully")
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for improved query performance."""
        
        indexes = [
            # Messages table indexes
            ("idx_messages_timestamp", "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)"),
            ("idx_messages_role", "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)"),
            ("idx_messages_model", "CREATE INDEX IF NOT EXISTS idx_messages_model ON messages(model)"),
            ("idx_messages_embedding_id", "CREATE INDEX IF NOT EXISTS idx_messages_embedding_id ON messages(embedding_id)"),
            
            # Conversations table indexes
            ("idx_conversations_last_updated", "CREATE INDEX IF NOT EXISTS idx_conversations_last_updated ON conversations(last_updated DESC)"),
            ("idx_conversations_created_at", "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC)"),
            
            # Junction table indexes
            ("idx_message_conversations_conv_id", "CREATE INDEX IF NOT EXISTS idx_message_conversations_conv_id ON message_conversations(conversation_id)"),
            ("idx_message_conversations_seq", "CREATE INDEX IF NOT EXISTS idx_message_conversations_seq ON message_conversations(conversation_id, sequence_number)"),
        ]
        
        for index_name, index_sql in indexes:
            try:
                conn.execute(index_sql)
                logger.debug(f"Created index: {index_name}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to create index {index_name}: {e}")
    
    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations to update schema."""
        
        # Get current schema version
        current_version = self._get_schema_version(conn)
        
        # Define migrations
        migrations = [
            (1, "Initial schema", self._migration_v1),
            # Future migrations can be added here
        ]
        
        # Apply migrations
        for version, description, migration_func in migrations:
            if version > current_version:
                try:
                    logger.info(f"Applying migration v{version}: {description}")
                    migration_func(conn)
                    self._set_schema_version(conn, version, description)
                    conn.commit()
                    logger.info(f"Migration v{version} applied successfully")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Migration v{version} failed: {e}")
                    raise
    
    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version."""
        try:
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0
        except sqlite3.Error:
            # Table doesn't exist yet
            return 0
    
    def _set_schema_version(self, conn: sqlite3.Connection, version: int, description: str) -> None:
        """Set schema version after successful migration."""
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (version, description)
        )
    
    def _migration_v1(self, conn: sqlite3.Connection) -> None:
        """Initial migration - schema already created in _create_schema."""
        # This migration is a no-op since the schema is created in _create_schema
        # Future migrations would modify the schema here
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_connection() as conn:
                # Test basic connectivity
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
                
                # Get table counts
                tables_info = {}
                for table in ['messages', 'conversations', 'message_conversations']:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    tables_info[table] = count
                
                # Get database size
                cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                return {
                    'status': 'healthy',
                    'database_path': self.db_path,
                    'database_size_bytes': db_size,
                    'tables': tables_info,
                    'schema_version': self._get_schema_version(conn)
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_path': self.db_path
            }
    
    def backup_database(self, backup_path: str) -> None:
        """Create a backup of the database."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.get_connection() as source_conn:
                with sqlite3.connect(str(backup_path)) as backup_conn:
                    source_conn.backup(backup_conn)
            
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def vacuum_database(self) -> None:
        """Optimize database by running VACUUM command."""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            
            logger.info("Database vacuum completed successfully")
            
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            raise