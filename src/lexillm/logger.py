"""
Logging module for LexiLLM

This module provides standardized logging functionality for the entire package,
replacing ad-hoc print statements with structured logging.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Optional

from .config import LOGGING_CONFIG

# Define log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class LexiLLMLogger:
    """
    Centralized logger for the LexiLLM package.
    
    This class provides standardized logging functionality with file 
    and console handlers, as well as configurable log levels.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for the logger."""
        if cls._instance is None:
            cls._instance = super(LexiLLMLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with configured handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger("lexillm")
        self.logger.setLevel(LOG_LEVELS.get(LOGGING_CONFIG["log_level"], logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(LOGGING_CONFIG["log_format"])
        
        # Create console handler if enabled
        if LOGGING_CONFIG["console_logging"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Create file handler
        log_file = LOGGING_CONFIG["log_file"]
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Avoid propagation to root logger
        self.logger.propagate = False
        
        # Log initialization
        self.logger.info("LexiLLM logger initialized")
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


# Global functions for easy access
def get_logger() -> logging.Logger:
    """Get the LexiLLM logger instance."""
    return LexiLLMLogger().get_logger()

def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    get_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    """Log an info message."""
    get_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    get_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """Log an error message."""
    get_logger().error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    get_logger().critical(msg, *args, **kwargs)

def exception(msg: str, *args, exc_info=True, **kwargs):
    """Log an exception message with traceback."""
    get_logger().exception(msg, *args, exc_info=exc_info, **kwargs)
