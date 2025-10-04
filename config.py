"""
Configuration management for the Databricks PDF upload application.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for managing application settings."""
    
    # Databricks Configuration
    DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
    DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
    DATABRICKS_CLUSTER_ID = os.getenv('DATABRICKS_CLUSTER_ID')

    # Workspace Paths
    DATABRICKS_UPLOAD_PATH = os.getenv('DATABRICKS_UPLOAD_PATH', '/Workspace/Shared/pdf_uploads')
    DATABRICKS_NOTEBOOK_PATH = os.getenv('DATABRICKS_NOTEBOOK_PATH', '/Workspace/Shared/pdf_processing')

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 1500))

    # Databricks AI Configuration
    DATABRICKS_AI_MODEL = os.getenv('DATABRICKS_AI_MODEL', 'databricks-gpt-oss-120b')
    DATABRICKS_AI_MAX_TOKENS = int(os.getenv('DATABRICKS_AI_MAX_TOKENS', 8192))
    DATABRICKS_AI_TEMPERATURE = float(os.getenv('DATABRICKS_AI_TEMPERATURE', 0.7))

    # Application Configuration
    APP_TITLE = os.getenv('APP_TITLE', 'PDF Upload to Databricks')
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 50))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate the configuration settings.
        
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required Databricks settings
        if not cls.DATABRICKS_HOST:
            validation_result['errors'].append('DATABRICKS_HOST is required')
        elif not cls.DATABRICKS_HOST.startswith('https://'):
            validation_result['warnings'].append('DATABRICKS_HOST should start with https://')
        
        if not cls.DATABRICKS_TOKEN:
            validation_result['errors'].append('DATABRICKS_TOKEN is required')
        
        # Check file size limits
        if cls.MAX_FILE_SIZE_MB <= 0:
            validation_result['errors'].append('MAX_FILE_SIZE_MB must be greater than 0')
        elif cls.MAX_FILE_SIZE_MB > 100:
            validation_result['warnings'].append('MAX_FILE_SIZE_MB is quite large (>100MB)')
        
        # Check workspace paths
        if not cls.DATABRICKS_UPLOAD_PATH.startswith('/'):
            validation_result['errors'].append('DATABRICKS_UPLOAD_PATH must start with /')
        
        if not cls.DATABRICKS_NOTEBOOK_PATH.startswith('/'):
            validation_result['errors'].append('DATABRICKS_NOTEBOOK_PATH must start with /')
        
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Dict with configuration summary
        """
        return {
            'databricks_host': cls.DATABRICKS_HOST,
            'databricks_token_set': bool(cls.DATABRICKS_TOKEN),
            'upload_path': cls.DATABRICKS_UPLOAD_PATH,
            'notebook_path': cls.DATABRICKS_NOTEBOOK_PATH,
            'max_file_size_mb': cls.MAX_FILE_SIZE_MB,
            'app_title': cls.APP_TITLE,
            'log_level': cls.LOG_LEVEL
        }
    
    @classmethod
    def create_env_template(cls, filepath: str = '.env.example') -> None:
        """
        Create an environment template file.
        
        Args:
            filepath: Path to create the template file
        """
        template_content = """# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_CLUSTER_ID=your-cluster-id

# Optional: Specific workspace paths
DATABRICKS_UPLOAD_PATH=/Workspace/Shared/pdf_uploads
DATABRICKS_NOTEBOOK_PATH=/Workspace/Shared/pdf_processing

# Application Configuration
APP_TITLE=PDF Upload to Databricks
MAX_FILE_SIZE_MB=50

# Logging Configuration
LOG_LEVEL=INFO
"""
        
        with open(filepath, 'w') as f:
            f.write(template_content)


def get_databricks_config() -> Dict[str, Optional[str]]:
    """
    Get Databricks-specific configuration.
    
    Returns:
        Dict with Databricks configuration
    """
    return {
        'host': Config.DATABRICKS_HOST,
        'token': Config.DATABRICKS_TOKEN,
        'cluster_id': Config.DATABRICKS_CLUSTER_ID,
        'upload_path': Config.DATABRICKS_UPLOAD_PATH,
        'notebook_path': Config.DATABRICKS_NOTEBOOK_PATH
    }


def get_app_config() -> Dict[str, Any]:
    """
    Get application-specific configuration.
    
    Returns:
        Dict with application configuration
    """
    return {
        'title': Config.APP_TITLE,
        'max_file_size_mb': Config.MAX_FILE_SIZE_MB,
        'log_level': Config.LOG_LEVEL
    }


# Validate configuration on import
_validation = Config.validate_config()
if not _validation['is_valid']:
    import warnings
    warnings.warn(f"Configuration validation failed: {_validation['errors']}")
