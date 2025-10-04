"""
Logging utilities for the Databricks PDF upload application.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from config import Config


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(name: str = __name__, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with appropriate handlers and formatting.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Get log level from config or parameter
    log_level = level or Config.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = CustomFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler for errors (optional)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = logging.FileHandler(
        f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.WARNING)
    
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_function_call(func):
    """
    Decorator to log function calls and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        # Log function start
        logger.debug(f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Completed {func.__name__} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper


class ErrorHandler:
    """Centralized error handling utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_databricks_error(self, error: Exception, context: str = "") -> dict:
        """
        Handle Databricks-specific errors.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Standardized error response
        """
        error_msg = str(error)
        
        # Common Databricks error patterns
        if "401" in error_msg or "Unauthorized" in error_msg:
            user_msg = "Authentication failed. Please check your access token."
            self.logger.error(f"Databricks auth error {context}: {error_msg}")
            
        elif "403" in error_msg or "Forbidden" in error_msg:
            user_msg = "Access denied. Please check your permissions."
            self.logger.error(f"Databricks permission error {context}: {error_msg}")
            
        elif "404" in error_msg or "Not Found" in error_msg:
            user_msg = "Resource not found. Please check the workspace path."
            self.logger.error(f"Databricks resource error {context}: {error_msg}")
            
        elif "timeout" in error_msg.lower():
            user_msg = "Request timed out. Please try again."
            self.logger.error(f"Databricks timeout error {context}: {error_msg}")
            
        else:
            user_msg = f"Databricks error: {error_msg}"
            self.logger.error(f"Databricks general error {context}: {error_msg}")
        
        return {
            'success': False,
            'error': user_msg,
            'error_type': 'databricks_error',
            'context': context
        }
    
    def handle_pdf_error(self, error: Exception, context: str = "") -> dict:
        """
        Handle PDF processing errors.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Standardized error response
        """
        error_msg = str(error)
        
        if "PdfReadError" in error_msg or "Invalid PDF" in error_msg:
            user_msg = "Invalid PDF file. The file may be corrupted or not a valid PDF."
            self.logger.error(f"PDF read error {context}: {error_msg}")
            
        elif "encrypted" in error_msg.lower():
            user_msg = "PDF is password protected. Please provide an unencrypted PDF."
            self.logger.error(f"PDF encryption error {context}: {error_msg}")
            
        elif "size" in error_msg.lower():
            user_msg = "PDF file is too large. Please use a smaller file."
            self.logger.error(f"PDF size error {context}: {error_msg}")
            
        else:
            user_msg = f"PDF processing error: {error_msg}"
            self.logger.error(f"PDF general error {context}: {error_msg}")
        
        return {
            'success': False,
            'error': user_msg,
            'error_type': 'pdf_error',
            'context': context
        }
    
    def handle_general_error(self, error: Exception, context: str = "") -> dict:
        """
        Handle general application errors.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Standardized error response
        """
        error_msg = str(error)
        self.logger.error(f"General error {context}: {error_msg}")
        
        return {
            'success': False,
            'error': f"An unexpected error occurred: {error_msg}",
            'error_type': 'general_error',
            'context': context
        }


def create_error_handler(logger_name: str = __name__) -> ErrorHandler:
    """
    Create an error handler with a logger.
    
    Args:
        logger_name: Name for the logger
        
    Returns:
        ErrorHandler instance
    """
    logger = setup_logger(logger_name)
    return ErrorHandler(logger)


# Global logger instance
app_logger = setup_logger('databricks_pdf_app')


def log_upload_attempt(filename: str, file_size: int, user_info: dict = None):
    """Log PDF upload attempt."""
    app_logger.info(
        f"PDF upload attempt - File: {filename}, Size: {file_size/1024/1024:.2f}MB, "
        f"User: {user_info.get('user', 'unknown') if user_info else 'unknown'}"
    )


def log_upload_success(filename: str, workspace_path: str, processing_time: float):
    """Log successful PDF upload."""
    app_logger.info(
        f"PDF upload success - File: {filename}, Path: {workspace_path}, "
        f"Time: {processing_time:.2f}s"
    )


def log_upload_failure(filename: str, error: str, processing_time: float):
    """Log failed PDF upload."""
    app_logger.error(
        f"PDF upload failed - File: {filename}, Error: {error}, "
        f"Time: {processing_time:.2f}s"
    )
