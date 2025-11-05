"""Logging configuration and setup for Universal Memory Layer."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union


class UMLLogger:
    """Universal Memory Layer logging configuration."""
    
    DEFAULT_LOG_LEVEL = logging.INFO
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_FILE = "uml.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup_logging(
        cls,
        log_level: Optional[Union[str, int]] = None,
        log_file: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True,
        format_string: Optional[str] = None
    ) -> None:
        """Set up logging configuration for the entire application.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Log file name (default: uml.log)
            log_dir: Directory for log files (default: logs)
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            format_string: Custom log format string
        """
        if cls._configured:
            return
        
        # Determine log level
        if log_level is None:
            log_level = os.getenv("UML_LOG_LEVEL", cls.DEFAULT_LOG_LEVEL)
        
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), cls.DEFAULT_LOG_LEVEL)
        
        # Set up format
        if format_string is None:
            format_string = os.getenv("UML_LOG_FORMAT", cls.DEFAULT_LOG_FORMAT)
        
        formatter = logging.Formatter(format_string)
        
        # Configure root logger
        root_logger = logging.getLogger("universal_memory_layer")
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            if log_dir is None:
                log_dir = os.getenv("UML_LOG_DIR", cls.DEFAULT_LOG_DIR)
            if log_file is None:
                log_file = os.getenv("UML_LOG_FILE", cls.DEFAULT_LOG_FILE)
            
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_path = log_dir / log_file
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=cls.MAX_LOG_SIZE,
                backupCount=cls.BACKUP_COUNT
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
        
        cls._configured = True
        
        # Log initial setup message
        logger = cls.get_logger("logging_config")
        logger.info(f"Logging configured - Level: {logging.getLevelName(log_level)}")
        if file_output:
            logger.info(f"Log file: {log_path}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for the given name.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        full_name = f"universal_memory_layer.{name}"
        
        if full_name not in cls._loggers:
            logger = logging.getLogger(full_name)
            cls._loggers[full_name] = logger
        
        return cls._loggers[full_name]
    
    @classmethod
    def set_level(cls, level: Union[str, int]) -> None:
        """Change the logging level for all UML loggers.
        
        Args:
            level: New logging level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger("universal_memory_layer")
        root_logger.setLevel(level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def reset_logging(cls) -> None:
        """Reset logging configuration (mainly for testing)."""
        root_logger = logging.getLogger("universal_memory_layer")
        
        # Close all handlers before clearing
        for handler in root_logger.handlers:
            handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        root_logger.handlers.clear()
        cls._loggers.clear()
        cls._configured = False


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger.
    
    Args:
        name: Logger name (typically __name__ or module name)
        
    Returns:
        Logger instance
    """
    # Extract just the module name if full path is provided
    if name.startswith("universal_memory_layer."):
        name = name.replace("universal_memory_layer.", "")
    
    return UMLLogger.get_logger(name)


def setup_logging(**kwargs) -> None:
    """Convenience function to set up logging.
    
    Args:
        **kwargs: Arguments passed to UMLLogger.setup_logging()
    """
    UMLLogger.setup_logging(**kwargs)