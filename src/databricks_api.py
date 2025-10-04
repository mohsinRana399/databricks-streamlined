"""
Databricks API integration for PDF upload and processing workflows.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.databricks_client import DatabricksClient
from utils.pdf_processor import PDFProcessor
from src.databricks_ai_engine import DatabricksAIEngine

logger = logging.getLogger(__name__)


class DatabricksAPIIntegration:
    """High-level API integration for PDF upload and processing workflows."""
    
    def __init__(self, host: str = None, token: str = None, max_file_size_mb: int = 50):
        """
        Initialize the API integration.

        Args:
            host: Databricks workspace URL
            token: Personal access token
            max_file_size_mb: Maximum file size for uploads
        """
        self.client = DatabricksClient(host, token)
        self.pdf_processor = PDFProcessor(max_file_size_mb)
        self.upload_base_path = os.getenv('DATABRICKS_UPLOAD_PATH', '/Workspace/Shared/pdf_uploads')
        self.notebook_base_path = os.getenv('DATABRICKS_NOTEBOOK_PATH', '/Workspace/Shared/pdf_processing')

        # Initialize Databricks AI Engine
        self.ai_engine = DatabricksAIEngine(
            databricks_client=self.client,
            model=os.getenv('DATABRICKS_AI_MODEL', 'databricks-gpt-oss-120b'),
            cluster_id=os.getenv('DATABRICKS_CLUSTER_ID')
        )
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Databricks workspace."""
        return self.client.test_connection()
    
    def upload_pdf_workflow(self, file_content: bytes, filename: str, 
                           create_processing_notebook: bool = True) -> Dict[str, Any]:
        """
        Complete workflow for uploading PDF and optionally creating processing notebook.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            create_processing_notebook: Whether to create a processing notebook
            
        Returns:
            Dict with complete workflow results
        """
        workflow_result = {
            'success': False,
            'steps': {},
            'pdf_path': None,
            'notebook_path': None,
            'metadata': None
        }
        
        try:
            # Step 1: Prepare PDF for upload
            logger.info(f"Preparing PDF {filename} for upload")
            preparation = self.pdf_processor.prepare_for_upload(
                file_content, filename, 
                workspace_path=f"{self.upload_base_path}/{filename}"
            )
            
            workflow_result['steps']['preparation'] = preparation
            
            if not preparation['ready_for_upload']:
                workflow_result['steps']['preparation']['status'] = 'failed'
                return workflow_result
            
            workflow_result['metadata'] = preparation['metadata']
            
            # Step 2: Upload PDF to workspace
            logger.info(f"Uploading PDF to {preparation['workspace_path']}")
            upload_result = self.client.upload_file_to_workspace(
                file_content=preparation['file_content'],
                workspace_path=preparation['workspace_path'],
                overwrite=True
            )
            
            workflow_result['steps']['upload'] = upload_result
            
            if not upload_result['success']:
                return workflow_result
            
            workflow_result['pdf_path'] = preparation['workspace_path']
            
            # Step 3: Create processing notebook (if requested)
            if create_processing_notebook:
                notebook_name = f"process_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                notebook_path = f"{self.notebook_base_path}/{notebook_name}"
                
                logger.info(f"Creating processing notebook at {notebook_path}")
                notebook_content = self._generate_processing_notebook(
                    pdf_path=preparation['workspace_path'],
                    metadata=preparation['metadata']
                )
                
                notebook_result = self.client.create_notebook_from_template(
                    notebook_path=notebook_path,
                    template_content=notebook_content,
                    overwrite=True
                )
                
                workflow_result['steps']['notebook_creation'] = notebook_result
                
                if notebook_result['success']:
                    workflow_result['notebook_path'] = notebook_path
            
            # Mark workflow as successful if all required steps completed
            workflow_result['success'] = (
                upload_result['success'] and 
                (not create_processing_notebook or workflow_result['steps'].get('notebook_creation', {}).get('success', False))
            )
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            workflow_result['steps']['error'] = str(e)
        
        return workflow_result
    
    def _generate_processing_notebook(self, pdf_path: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a Databricks notebook for processing the uploaded PDF.
        
        Args:
            pdf_path: Path to the uploaded PDF in workspace
            metadata: PDF metadata
            
        Returns:
            Notebook content as string
        """
        notebook_content = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # PDF Processing Notebook
# MAGIC 
# MAGIC **PDF File:** `{pdf_path}`
# MAGIC 
# MAGIC **Upload Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# MAGIC 
# MAGIC ## PDF Metadata
# MAGIC - **Pages:** {metadata.get('page_count', 'Unknown')}
# MAGIC - **File Size:** {metadata.get('file_size', 0) / 1024 / 1024:.2f} MB
# MAGIC - **Title:** {metadata.get('title', 'N/A')}
# MAGIC - **Author:** {metadata.get('author', 'N/A')}
# MAGIC - **Creation Date:** {metadata.get('creation_date', 'N/A')}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import os
import base64
from datetime import datetime

# PDF file path in workspace
pdf_path = "{pdf_path}"
print(f"Processing PDF: {{pdf_path}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read PDF File from Workspace

# COMMAND ----------

# Read the PDF file from workspace
try:
    # Note: In a real scenario, you might need to use dbutils.fs.cp to copy from workspace to DBFS
    # or use appropriate methods to read the file content
    print(f"PDF file is available at: {{pdf_path}}")
    print("File metadata:")
    print(f"- Pages: {metadata.get('page_count', 'Unknown')}")
    print(f"- Size: {metadata.get('file_size', 0) / 1024 / 1024:.2f} MB")
    print(f"- Encrypted: {metadata.get('is_encrypted', False)}")
except Exception as e:
    print(f"Error reading PDF: {{e}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Processing Functions
# MAGIC 
# MAGIC Add your custom PDF processing logic here.

# COMMAND ----------

def process_pdf_content():
    """
    Add your PDF processing logic here.
    This could include:
    - Text extraction
    - Data parsing
    - Analysis
    - Machine learning inference
    """
    print("Add your PDF processing logic here")
    
    # Example: You could integrate with your existing demo.py logic
    # for text chunking and OpenAI processing
    
    return "Processing completed"

# Execute processing
result = process_pdf_content()
print(f"Processing result: {{result}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and Output
# MAGIC 
# MAGIC Display or save your processing results here.

# COMMAND ----------

# Save results or display output
print("PDF processing workflow completed successfully!")
print(f"Processed at: {{datetime.now()}}")
'''
        
        return notebook_content
    
    def list_uploaded_pdfs(self) -> List[Dict[str, Any]]:
        """
        List all uploaded PDF files in the workspace.
        
        Returns:
            List of PDF file information
        """
        try:
            files = self.client.list_workspace_files(self.upload_base_path)
            pdf_files = [f for f in files if f['path'].lower().endswith('.pdf')]
            return pdf_files
        except Exception as e:
            logger.error(f"Failed to list uploaded PDFs: {str(e)}")
            return []
    
    def list_processing_notebooks(self) -> List[Dict[str, Any]]:
        """
        List all processing notebooks in the workspace.
        
        Returns:
            List of notebook information
        """
        try:
            files = self.client.list_workspace_files(self.notebook_base_path)
            notebooks = [f for f in files if f['object_type'] == 'NOTEBOOK']
            return notebooks
        except Exception as e:
            logger.error(f"Failed to list processing notebooks: {str(e)}")
            return []
    
    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get available cluster information."""
        return self.client.get_clusters()

    def query_pdf_with_ai(self, file_content: bytes, question: str,
                         conversation_id: str = None) -> Dict[str, Any]:
        """
        Query PDF content using Databricks AI functions.

        Args:
            file_content: PDF file content as bytes
            question: User question
            conversation_id: ID for conversation context

        Returns:
            Dict with AI query results
        """
        return self.ai_engine.query_pdf_with_databricks_ai(
            file_content=file_content,
            question=question,
            conversation_id=conversation_id
        )

    def get_ai_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get AI conversation history."""
        return self.ai_engine.get_conversation_history(conversation_id)

    def clear_ai_conversation(self, conversation_id: str):
        """Clear AI conversation context."""
        self.ai_engine.clear_conversation_context(conversation_id)
