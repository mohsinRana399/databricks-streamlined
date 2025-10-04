"""
PDF processing utilities for validation, metadata extraction, and text processing.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Utility class for processing PDF files."""
    
    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize PDF processor.
        
        Args:
            max_file_size_mb: Maximum allowed file size in MB
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate PDF file.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_size': len(file_content),
            'filename': filename
        }
        
        # Check file size
        if len(file_content) > self.max_file_size_bytes:
            validation_result['errors'].append(
                f"File size ({len(file_content) / 1024 / 1024:.2f} MB) exceeds maximum allowed size "
                f"({self.max_file_size_bytes / 1024 / 1024} MB)"
            )
        
        # Check file extension
        if not filename.lower().endswith('.pdf'):
            validation_result['errors'].append("File must have .pdf extension")
        
        # Check if file is empty
        if len(file_content) == 0:
            validation_result['errors'].append("File is empty")
            return validation_result
        
        # Try to read PDF content
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                validation_result['errors'].append("PDF has no pages")
            else:
                validation_result['page_count'] = len(pdf_reader.pages)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                validation_result['warnings'].append("PDF is encrypted/password protected")
            
        except Exception as e:
            validation_result['errors'].append(f"Invalid PDF format: {str(e)}")
        
        # Set validation status
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def extract_metadata(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            file_content: PDF file content as bytes
            
        Returns:
            Dict with extracted metadata
        """
        metadata = {
            'title': None,
            'author': None,
            'subject': None,
            'creator': None,
            'producer': None,
            'creation_date': None,
            'modification_date': None,
            'page_count': 0,
            'file_size': len(file_content),
            'is_encrypted': False
        }
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            # Basic info
            metadata['page_count'] = len(pdf_reader.pages)
            metadata['is_encrypted'] = pdf_reader.is_encrypted
            
            # Document metadata
            if pdf_reader.metadata:
                pdf_metadata = pdf_reader.metadata
                metadata['title'] = pdf_metadata.get('/Title')
                metadata['author'] = pdf_metadata.get('/Author')
                metadata['subject'] = pdf_metadata.get('/Subject')
                metadata['creator'] = pdf_metadata.get('/Creator')
                metadata['producer'] = pdf_metadata.get('/Producer')
                
                # Handle dates
                creation_date = pdf_metadata.get('/CreationDate')
                if creation_date:
                    try:
                        # PDF dates are in format D:YYYYMMDDHHmmSSOHH'mm'
                        if creation_date.startswith('D:'):
                            date_str = creation_date[2:16]  # Extract YYYYMMDDHHMMSS
                            metadata['creation_date'] = datetime.strptime(date_str, '%Y%m%d%H%M%S').isoformat()
                    except:
                        metadata['creation_date'] = str(creation_date)
                
                mod_date = pdf_metadata.get('/ModDate')
                if mod_date:
                    try:
                        if mod_date.startswith('D:'):
                            date_str = mod_date[2:16]
                            metadata['modification_date'] = datetime.strptime(date_str, '%Y%m%d%H%M%S').isoformat()
                    except:
                        metadata['modification_date'] = str(mod_date)
        
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
        
        return metadata
    
    def extract_text_preview(self, file_content: bytes, max_pages: int = 3) -> Dict[str, Any]:
        """
        Extract text preview from first few pages of PDF.
        
        Args:
            file_content: PDF file content as bytes
            max_pages: Maximum number of pages to extract text from
            
        Returns:
            Dict with extracted text and page information
        """
        result = {
            'text_preview': '',
            'pages_processed': 0,
            'total_pages': 0,
            'extraction_successful': False
        }
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            result['total_pages'] = len(pdf_reader.pages)
            
            text_parts = []
            pages_to_process = min(max_pages, len(pdf_reader.pages))
            
            for i in range(pages_to_process):
                try:
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"--- Page {i+1} ---\n{page_text}\n")
                        result['pages_processed'] += 1
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {str(e)}")
            
            result['text_preview'] = '\n'.join(text_parts)
            result['extraction_successful'] = len(text_parts) > 0
            
        except Exception as e:
            logger.error(f"Failed to extract text preview: {str(e)}")
        
        return result
    
    def prepare_for_upload(self, file_content: bytes, filename: str, 
                          workspace_path: str = None) -> Dict[str, Any]:
        """
        Prepare PDF file for upload to Databricks.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            workspace_path: Target workspace path
            
        Returns:
            Dict with preparation results and upload information
        """
        # Validate the PDF
        validation = self.validate_pdf(file_content, filename)
        if not validation['is_valid']:
            return {
                'ready_for_upload': False,
                'validation': validation
            }
        
        # Extract metadata
        metadata = self.extract_metadata(file_content)
        
        # Generate workspace path if not provided
        if not workspace_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_filename = os.path.splitext(filename)[0]
            workspace_path = f"/Workspace/Shared/pdf_uploads/{clean_filename}_{timestamp}.pdf"
        
        return {
            'ready_for_upload': True,
            'validation': validation,
            'metadata': metadata,
            'workspace_path': workspace_path,
            'file_content': file_content,
            'upload_timestamp': datetime.now().isoformat()
        }
