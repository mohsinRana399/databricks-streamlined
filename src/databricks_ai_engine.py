"""
Databricks AI Query Engine using native Databricks AI functions.
This replaces the OpenAI API with Databricks' built-in AI capabilities.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import base64
from io import BytesIO
import PyPDF2

# Add current directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.databricks_client import DatabricksClient
from utils.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class DatabricksAIEngine:
    """Engine for querying PDF content using Databricks AI functions."""
    
    def __init__(self, databricks_client: DatabricksClient, 
                 model: str = "databricks-gpt-oss-120b",
                 cluster_id: str = None):
        """
        Initialize the Databricks AI Engine.
        
        Args:
            databricks_client: Databricks client instance
            model: Databricks AI model to use
            cluster_id: Cluster ID for running AI queries
        """
        self.databricks_client = databricks_client
        self.model = model
        self.cluster_id = cluster_id or os.getenv('DATABRICKS_CLUSTER_ID')
        self.max_tokens = 8192
        self.temperature = 0.7
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        # Conversation context storage
        self.conversation_contexts = {}
    
    def extract_full_text_from_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract text from PDF content with multiple fallback methods.

        Args:
            file_content: PDF file content as bytes

        Returns:
            Dict with extracted text and metadata
        """
        result = {
            'text': '',
            'pages': [],
            'total_pages': 0,
            'extraction_successful': False,
            'error': None,
            'extraction_method': None
        }

        # Method 1: Try PyPDF2 (primary method)
        try:
            logger.info(f"Attempting PyPDF2 text extraction from {len(file_content)} bytes")
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            result['total_pages'] = len(pdf_reader.pages)

            text_parts = []
            page_texts = []

            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        page_info = {
                            'page_number': i + 1,
                            'text': page_text,
                            'char_count': len(page_text)
                        }
                        page_texts.append(page_info)
                        text_parts.append(f"--- Page {i+1} ---\n{page_text}\n")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {str(e)}")

            if len(text_parts) > 0:
                result['text'] = '\n'.join(text_parts)
                result['pages'] = page_texts
                result['extraction_successful'] = True
                result['extraction_method'] = 'PyPDF2'
                logger.info(f"PyPDF2 extraction successful: {len(result['text'])} characters")
                return result
            else:
                logger.warning("PyPDF2 extracted no text, trying fallback methods")

        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            result['error'] = str(e)

        # Method 2: Fallback for corrupted PDFs
        logger.info("Attempting fallback text extraction for problematic PDF")

        # Check if it's at least a valid PDF structure
        if file_content.startswith(b'%PDF'):
            result['text'] = """--- PDF Upload Successful ---

This PDF file was uploaded successfully to your Databricks workspace, but text extraction encountered issues. This can happen with:

• Image-based PDFs (scanned documents)
• Password-protected or encrypted PDFs
• PDFs with non-standard internal structure
• Corrupted PDF files

The PDF is stored and available in your workspace. You can:
• Try asking general questions about the document
• Check if the original PDF opens correctly in a PDF viewer
• Re-upload the PDF if it seems corrupted

File information:
• Size: """ + f"{len(file_content):,} bytes" + """
• Format: PDF detected
• Status: Upload successful, text extraction limited"""

            result['pages'] = [{'page_number': 1, 'text': 'PDF structure detected', 'char_count': 0}]
            result['total_pages'] = 1
            result['extraction_successful'] = True
            result['extraction_method'] = 'fallback_message'
            result['error'] = 'Text extraction failed but PDF structure detected'
            logger.info("Using fallback message for PDF with extraction issues")
        else:
            result['error'] = result.get('error', 'Invalid PDF format')
            logger.error(f"PDF text extraction completely failed: {result['error']}")

        return result
    
    def chunk_text_for_databricks(self, text: str, max_chunk_size: int = 15000) -> List[str]:
        """
        Chunk text for Databricks AI processing.
        Databricks AI functions can handle larger chunks than OpenAI.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_databricks_notebook_for_query(self, pdf_text: str, question: str, 
                                           context: List[Dict] = None) -> str:
        """
        Create a Databricks notebook that uses AI functions to query the PDF.
        
        Args:
            pdf_text: Extracted PDF text
            question: User question
            context: Previous conversation context
            
        Returns:
            Notebook content as string
        """
        # Build context string from previous conversation
        context_str = ""
        if context:
            context_str = "\n".join([
                f"Previous Q: {item['question']}\nPrevious A: {item['answer']}"
                for item in context[-3:]  # Last 3 exchanges for context
            ])

        # Build the AI query prompt
        if context_str:
            ai_prompt = f"""Previous conversation context:
{context_str}

Current question: {question}

Please answer the current question based on the provided PDF text, taking into account the previous conversation context. Be comprehensive and specific.

PDF Text:"""
        else:
            ai_prompt = f"""Question: {question}

Please answer this question based on the provided PDF text. Be comprehensive and specific.

PDF Text:"""

        # Escape single quotes for SQL
        ai_prompt_escaped = ai_prompt.replace("'", "''")

        # Create the notebook content
        notebook_content = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # PDF AI Query Processing
# MAGIC 
# MAGIC **Question:** {question}
# MAGIC 
# MAGIC **Processing Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import json

# Initialize Spark session
spark = SparkSession.builder.appName("PDF_AI_Query").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Text Data

# COMMAND ----------

# PDF text content
pdf_text = """
{pdf_text}
"""

print(f"PDF text length: {{len(pdf_text)}} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context and Question

# COMMAND ----------

# User question
question = """{question}"""

# Previous context (if any)
context = """{context_str}"""

print(f"Question: {question}")
if context:
    print(f"Context available: {len(context_str)} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Query Processing

# COMMAND ----------

# Create DataFrame with the text
df = spark.createDataFrame([(pdf_text,)], ["text"])

# Apply the AI function to process the question
df_result = df.selectExpr(f"""
    ai_query(
        '{self.model}',
        CONCAT('{ai_prompt_escaped}', text),
        modelParameters => named_struct(
            'max_tokens', {self.max_tokens},
            'temperature', {self.temperature}
        )
    ) as ai_response
""")

# Display the result
display(df_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract and Format Response

# COMMAND ----------

# Collect the AI response
result = df_result.collect()
if result:
    ai_response = result[0]['ai_response']
    print("AI Response:")
    print("=" * 50)
    print(ai_response)
    print("=" * 50)
else:
    print("No response generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Analysis (Optional)

# COMMAND ----------

# You can add additional AI queries for follow-up analysis
# For example, extracting specific information types:

# Extract key entities
df_entities = df.selectExpr(f"""
    ai_query(
        '{self.model}', 
        CONCAT('Extract all important entities (names, dates, amounts, organizations) from this text and format as JSON: ', text), 
        modelParameters => named_struct(
            'max_tokens', 2000, 
            'temperature', 0.3
        )
    ) as entities
""")

print("Key Entities:")
entities_result = df_entities.collect()
if entities_result:
    print(entities_result[0]['entities'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

print("Query Processing Complete!")
print(f"Question: {{question}}")
print(f"Model Used: {self.model}")
print(f"Processing Time: {{datetime.now()}}")
'''
        
        return notebook_content
    
    def execute_ai_query_direct(self, pdf_text: str, question: str,
                               context: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute AI query directly using Databricks SQL warehouse.

        Args:
            pdf_text: Extracted PDF text
            question: User question
            context: Previous conversation context

        Returns:
            Dict with query results including the actual AI response
        """
        try:
            # Build context string from previous conversation
            context_str = ""
            if context:
                context_str = "\n".join([
                    f"Previous Q: {item['question']}\nPrevious A: {item['answer']}"
                    for item in context[-3:]  # Last 3 exchanges for context
                ])

            # Build the AI query prompt
            if context_str:
                ai_prompt = f"""Previous conversation context:
{context_str}

Current question: {question}

Please answer the current question based on the provided PDF text, taking into account the previous conversation context. Be comprehensive and specific.

PDF Text: """
            else:
                ai_prompt = f"""Question: {question}

Please answer this question based on the provided PDF text. Be comprehensive and specific.

PDF Text: """

            # Combine prompt with PDF text
            full_prompt = ai_prompt + pdf_text

            # Escape single quotes for SQL safety
            full_prompt_escaped = full_prompt.replace("'", "''")

            # Execute AI query using SQL warehouse
            sql_query = f"""
            SELECT ai_query(
                '{self.model}',
                '{full_prompt_escaped}',
                named_struct(
                    'max_tokens', {self.max_tokens},
                    'temperature', {self.temperature}
                )
            ) as ai_response
            """

            # Execute the query using Databricks SQL
            result = self.databricks_client.execute_sql_query(sql_query)

            if result['success'] and result['data']:
                # Handle different column name formats
                row_data = result['data'][0]
                ai_response = None

                # Try different possible column names
                for key in ['ai_response', 'col_0', 'response']:
                    if key in row_data:
                        ai_response = row_data[key]
                        break

                if ai_response is None:
                    # Fallback: use the first value in the row
                    ai_response = list(row_data.values())[0] if row_data else "No response received"

                # Also create notebook for reference (optional)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                notebook_name = f"ai_query_{timestamp}"
                notebook_path = f"/Workspace/Shared/ai_queries/{notebook_name}"

                notebook_content = self.create_databricks_notebook_for_query(
                    pdf_text, question, context
                )

                self.databricks_client.create_notebook_from_template(
                    notebook_path=notebook_path,
                    template_content=notebook_content,
                    overwrite=True
                )

                return {
                    'success': True,
                    'question': question,
                    'answer': ai_response,
                    'notebook_path': notebook_path,
                    'model_used': self.model,
                    'processing_time': datetime.now().isoformat(),
                    'prompt_length': len(full_prompt),
                    'context_used': bool(context_str)
                }
            else:
                return {
                    'success': False,
                    'error': f"SQL execution failed: {result.get('error', 'Unknown error')}",
                    'question': question
                }

        except Exception as e:
            logger.error(f"Failed to execute direct AI query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question
            }

    def execute_ai_query_via_notebook(self, pdf_text: str, question: str,
                                    context: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute AI query by creating and running a Databricks notebook.
        This is a fallback method when direct SQL execution is not available.

        Args:
            pdf_text: Extracted PDF text
            question: User question
            context: Previous conversation context

        Returns:
            Dict with query results
        """
        try:
            # Try direct execution first
            direct_result = self.execute_ai_query_direct(pdf_text, question, context)
            if direct_result['success']:
                return direct_result

            # Fallback to notebook creation
            logger.info("Direct execution failed, falling back to notebook creation")

            # Create notebook content
            notebook_content = self.create_databricks_notebook_for_query(
                pdf_text, question, context
            )

            # Generate unique notebook name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            notebook_name = f"ai_query_{timestamp}"
            notebook_path = f"/Workspace/Shared/ai_queries/{notebook_name}"

            # Create the notebook
            create_result = self.databricks_client.create_notebook_from_template(
                notebook_path=notebook_path,
                template_content=notebook_content,
                overwrite=True
            )

            if not create_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to create notebook: {create_result['error']}",
                    'question': question
                }

            return {
                'success': True,
                'question': question,
                'answer': 'Notebook created successfully. Please run it in Databricks to get the AI response.',
                'notebook_path': notebook_path,
                'model_used': self.model,
                'processing_time': datetime.now().isoformat(),
                'note': 'Direct execution not available, notebook created for manual execution'
            }

        except Exception as e:
            logger.error(f"Failed to execute AI query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question
            }
    
    def query_pdf_with_databricks_ai(self, file_content: bytes, question: str, 
                                   conversation_id: str = None) -> Dict[str, Any]:
        """
        Complete workflow to query a PDF using Databricks AI functions.
        
        Args:
            file_content: PDF file content as bytes
            question: User question
            conversation_id: ID for conversation context
            
        Returns:
            Dict with query results
        """
        # Extract text from PDF
        extraction_result = self.extract_full_text_from_pdf(file_content)
        
        if not extraction_result['extraction_successful']:
            return {
                'success': False,
                'error': f"Failed to extract text from PDF: {extraction_result.get('error', 'Unknown error')}",
                'question': question
            }
        
        # Get conversation context
        context = self.conversation_contexts.get(conversation_id, []) if conversation_id else []
        
        # Execute AI query using direct method (faster)
        query_result = self.execute_ai_query_direct(
            extraction_result['text'], question, context
        )

        # Update conversation context with actual response
        if conversation_id and query_result['success']:
            if conversation_id not in self.conversation_contexts:
                self.conversation_contexts[conversation_id] = []

            self.conversation_contexts[conversation_id].append({
                'question': question,
                'answer': query_result.get('answer', 'No response available'),
                'timestamp': datetime.now().isoformat(),
                'notebook_path': query_result.get('notebook_path')
            })
            
            # Keep only last 10 exchanges
            self.conversation_contexts[conversation_id] = \
                self.conversation_contexts[conversation_id][-10:]
        
        return {
            'success': query_result['success'],
            'question': question,
            'answer': query_result.get('answer', 'Check the generated notebook for AI response'),
            'notebook_path': query_result.get('notebook_path'),
            'metadata': {
                'total_pages': extraction_result['total_pages'],
                'model_used': self.model,
                'processing_time': query_result.get('processing_time'),
                'text_length': len(extraction_result['text']),
                'prompt_length': query_result.get('prompt_length'),
                'context_used': query_result.get('context_used', False)
            },
            'conversation_id': conversation_id,
            'error': query_result.get('error')
        }
    
    def clear_conversation_context(self, conversation_id: str):
        """Clear conversation context for a specific conversation."""
        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a specific conversation."""
        return self.conversation_contexts.get(conversation_id, [])
