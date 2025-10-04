"""
PDF Manager for handling uploaded PDFs and their content for querying.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64

# Add current directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.databricks_client import DatabricksClient

logger = logging.getLogger(__name__)


class PDFManager:
    """Manager for handling PDF files and their content for querying."""
    
    def __init__(self, databricks_client: DatabricksClient):
        """
        Initialize PDF Manager.
        
        Args:
            databricks_client: Databricks client instance
        """
        self.databricks_client = databricks_client
        self.upload_base_path = os.getenv('DATABRICKS_UPLOAD_PATH', '/Workspace/Shared/pdf_uploads')
        
        # Cache for PDF content to avoid re-downloading
        self.pdf_content_cache = {}
    
    def list_available_pdfs(self) -> List[Dict[str, Any]]:
        """
        List all available PDF files in the workspace.
        
        Returns:
            List of PDF file information
        """
        try:
            files = self.databricks_client.list_workspace_files(self.upload_base_path)
            pdf_files = []
            
            for file_info in files:
                if file_info['path'].lower().endswith('.pdf'):
                    # Extract filename from path
                    filename = os.path.basename(file_info['path'])
                    
                    pdf_info = {
                        'filename': filename,
                        'workspace_path': file_info['path'],
                        'display_name': filename.replace('.pdf', ''),
                        'object_type': file_info.get('object_type', 'FILE'),
                        'cached': file_info['path'] in self.pdf_content_cache
                    }
                    pdf_files.append(pdf_info)
            
            # Sort by filename
            pdf_files.sort(key=lambda x: x['filename'])
            return pdf_files
            
        except Exception as e:
            logger.error(f"Failed to list PDFs: {str(e)}")
            return []
    
    def get_pdf_content(self, workspace_path: str, use_cache: bool = True) -> Optional[bytes]:
        """
        Get PDF content from workspace.
        
        Args:
            workspace_path: Path to PDF in workspace
            use_cache: Whether to use cached content
            
        Returns:
            PDF content as bytes or None if failed
        """
        # Check cache first
        if use_cache and workspace_path in self.pdf_content_cache:
            logger.info(f"Using cached content for {workspace_path}")
            return self.pdf_content_cache[workspace_path]
        
        try:
            # Download PDF content from workspace using export API
            logger.info(f"Downloading PDF content from {workspace_path}")

            # Use the workspace export API to get file content
            content = self.databricks_client.export_workspace_file(workspace_path)

            if content:
                # Cache the downloaded content
                if use_cache:
                    self.pdf_content_cache[workspace_path] = content
                logger.info(f"Successfully downloaded PDF content from {workspace_path} ({len(content)} bytes)")
                return content
            else:
                logger.warning(f"No content returned from workspace export: {workspace_path}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to get PDF content from {workspace_path}: {str(e)}")
            return None
    
    def cache_pdf_content(self, workspace_path: str, content: bytes):
        """
        Cache PDF content for faster access.
        
        Args:
            workspace_path: Path to PDF in workspace
            content: PDF content as bytes
        """
        self.pdf_content_cache[workspace_path] = content
        logger.info(f"Cached PDF content for {workspace_path} ({len(content)} bytes)")
    
    def get_cached_pdf_content(self, workspace_path: str) -> Optional[bytes]:
        """
        Get cached PDF content.
        
        Args:
            workspace_path: Path to PDF in workspace
            
        Returns:
            Cached PDF content or None
        """
        return self.pdf_content_cache.get(workspace_path)
    
    def clear_cache(self):
        """Clear all cached PDF content."""
        self.pdf_content_cache.clear()
        logger.info("Cleared PDF content cache")
    
    def get_pdf_info(self, workspace_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a PDF.
        
        Args:
            workspace_path: Path to PDF in workspace
            
        Returns:
            Dict with PDF information
        """
        filename = os.path.basename(workspace_path)
        
        info = {
            'filename': filename,
            'workspace_path': workspace_path,
            'display_name': filename.replace('.pdf', ''),
            'cached': workspace_path in self.pdf_content_cache,
            'cache_size': len(self.pdf_content_cache.get(workspace_path, b''))
        }
        
        # Try to get additional info from cached content
        if workspace_path in self.pdf_content_cache:
            content = self.pdf_content_cache[workspace_path]
            info['file_size'] = len(content)
            info['file_size_mb'] = len(content) / 1024 / 1024
        
        return info


class ConversationManager:
    """Manager for handling conversation contexts and history."""
    
    def __init__(self):
        """Initialize conversation manager."""
        self.conversations = {}
        self.active_conversations = {}
    
    def create_conversation(self, pdf_path: str, user_id: str = "default") -> str:
        """
        Create a new conversation for a PDF.
        
        Args:
            pdf_path: Path to the PDF being discussed
            user_id: User identifier
            
        Returns:
            Conversation ID
        """
        conversation_id = f"{user_id}_{pdf_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.conversations[conversation_id] = {
            'pdf_path': pdf_path,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'messages': [],
            'last_activity': datetime.now().isoformat()
        }
        
        self.active_conversations[user_id] = conversation_id
        logger.info(f"Created conversation {conversation_id} for PDF {pdf_path}")
        
        return conversation_id
    
    def get_active_conversation(self, user_id: str = "default") -> Optional[str]:
        """
        Get the active conversation ID for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Active conversation ID or None
        """
        return self.active_conversations.get(user_id)
    
    def set_active_conversation(self, conversation_id: str, user_id: str = "default"):
        """
        Set the active conversation for a user.
        
        Args:
            conversation_id: Conversation ID to set as active
            user_id: User identifier
        """
        if conversation_id in self.conversations:
            self.active_conversations[user_id] = conversation_id
            self.conversations[conversation_id]['last_activity'] = datetime.now().isoformat()
    
    def add_message(self, conversation_id: str, question: str, answer: str):
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            question: User question
            answer: System answer
        """
        if conversation_id in self.conversations:
            message = {
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversations[conversation_id]['messages'].append(message)
            self.conversations[conversation_id]['last_activity'] = datetime.now().isoformat()
            
            # Keep only last 20 messages
            if len(self.conversations[conversation_id]['messages']) > 20:
                self.conversations[conversation_id]['messages'] = \
                    self.conversations[conversation_id]['messages'][-20:]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of messages
        """
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]['messages']
        return []
    
    def get_conversation_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation information.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation info or None
        """
        return self.conversations.get(conversation_id)
    
    def list_conversations(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        List all conversations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation summaries
        """
        user_conversations = []
        
        for conv_id, conv_data in self.conversations.items():
            if conv_data['user_id'] == user_id:
                summary = {
                    'conversation_id': conv_id,
                    'pdf_path': conv_data['pdf_path'],
                    'created_at': conv_data['created_at'],
                    'last_activity': conv_data['last_activity'],
                    'message_count': len(conv_data['messages']),
                    'is_active': self.active_conversations.get(user_id) == conv_id
                }
                user_conversations.append(summary)
        
        # Sort by last activity (most recent first)
        user_conversations.sort(key=lambda x: x['last_activity'], reverse=True)
        return user_conversations
    
    def clear_conversation(self, conversation_id: str):
        """
        Clear a specific conversation.
        
        Args:
            conversation_id: Conversation ID to clear
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
            # Remove from active conversations if it was active
            for user_id, active_id in list(self.active_conversations.items()):
                if active_id == conversation_id:
                    del self.active_conversations[user_id]
    
    def clear_all_conversations(self, user_id: str = "default"):
        """
        Clear all conversations for a user.
        
        Args:
            user_id: User identifier
        """
        # Find conversations to remove
        conversations_to_remove = [
            conv_id for conv_id, conv_data in self.conversations.items()
            if conv_data['user_id'] == user_id
        ]
        
        # Remove conversations
        for conv_id in conversations_to_remove:
            del self.conversations[conv_id]
        
        # Clear active conversation
        if user_id in self.active_conversations:
            del self.active_conversations[user_id]
