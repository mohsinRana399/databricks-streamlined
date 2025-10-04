"""
PDF Query Engine for interactive questioning and answering using OpenAI.
Handles text extraction, chunking, and context-aware responses.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import openai
import tiktoken
from io import BytesIO
import PyPDF2

# Add current directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.databricks_client import DatabricksClient
from utils.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class PDFQueryEngine:
    """Engine for querying PDF content using OpenAI with chunking and context management."""
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the PDF Query Engine.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use for queries
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_tokens_per_chunk = 3000
        self.max_response_tokens = 1500
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        # Conversation context storage
        self.conversation_contexts = {}
    
    def extract_full_text_from_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract full text from PDF content.
        
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
            'error': None
        }
        
        try:
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
            
            result['text'] = '\n'.join(text_parts)
            result['pages'] = page_texts
            result['extraction_successful'] = len(text_parts) > 0
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to extract text from PDF: {str(e)}")
        
        return result
    
    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Chunk text into smaller pieces for processing.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        max_tokens = max_tokens or self.max_tokens_per_chunk
        
        try:
            enc = tiktoken.encoding_for_model(self.model)
            tokens = enc.encode(text)
            
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i+max_tokens]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            # Fallback: simple character-based chunking
            chunk_size = max_tokens * 4  # Rough estimate: 4 chars per token
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def query_pdf_chunks(self, chunks: List[str], question: str, 
                        context: List[Dict] = None) -> Dict[str, Any]:
        """
        Query PDF chunks with a question and return consolidated response.
        
        Args:
            chunks: List of text chunks
            question: User question
            context: Previous conversation context
            
        Returns:
            Dict with response and metadata
        """
        chunk_responses = []
        total_tokens_used = 0
        
        # Build context string from previous conversation
        context_str = ""
        if context:
            context_str = "\n".join([
                f"Previous Q: {item['question']}\nPrevious A: {item['answer']}"
                for item in context[-3:]  # Last 3 exchanges for context
            ])
        
        for i, chunk in enumerate(chunks):
            try:
                # Create messages for this chunk
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful assistant that answers questions based on PDF content. "
                            "Analyze the provided text chunk and answer the user's question. "
                            "If the chunk doesn't contain relevant information, say so clearly. "
                            "Be concise but comprehensive in your response."
                        )
                    }
                ]
                
                # Add context if available
                if context_str:
                    messages.append({
                        "role": "system",
                        "content": f"Previous conversation context:\n{context_str}"
                    })
                
                # Add the main query
                messages.append({
                    "role": "user",
                    "content": f"Question: {question}\n\nText chunk {i+1}:\n{chunk}"
                })
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=0.3
                )
                
                chunk_response = {
                    'chunk_number': i + 1,
                    'response': response.choices[0].message.content,
                    'tokens_used': response.usage.total_tokens if response.usage else 0,
                    'has_relevant_info': 'relevant' in response.choices[0].message.content.lower() or 
                                       'information' in response.choices[0].message.content.lower()
                }
                
                chunk_responses.append(chunk_response)
                total_tokens_used += chunk_response['tokens_used']
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {str(e)}")
                chunk_responses.append({
                    'chunk_number': i + 1,
                    'response': f"Error processing chunk: {str(e)}",
                    'tokens_used': 0,
                    'has_relevant_info': False
                })
        
        # Consolidate responses
        consolidated_response = self._consolidate_chunk_responses(
            chunk_responses, question, context_str
        )
        
        return {
            'question': question,
            'consolidated_answer': consolidated_response['answer'],
            'chunk_responses': chunk_responses,
            'total_chunks_processed': len(chunks),
            'total_tokens_used': total_tokens_used + consolidated_response['tokens_used'],
            'processing_time': datetime.now().isoformat(),
            'model_used': self.model
        }
    
    def _consolidate_chunk_responses(self, chunk_responses: List[Dict], 
                                   question: str, context_str: str = "") -> Dict[str, Any]:
        """
        Consolidate multiple chunk responses into a single coherent answer.
        
        Args:
            chunk_responses: List of responses from individual chunks
            question: Original question
            context_str: Previous conversation context
            
        Returns:
            Dict with consolidated answer and metadata
        """
        # Filter responses that have relevant information
        relevant_responses = [
            resp for resp in chunk_responses 
            if resp['has_relevant_info'] and 'error' not in resp['response'].lower()
        ]
        
        if not relevant_responses:
            return {
                'answer': "I couldn't find relevant information to answer your question in the provided PDF content.",
                'tokens_used': 0
            }
        
        # If only one relevant response, return it directly
        if len(relevant_responses) == 1:
            return {
                'answer': relevant_responses[0]['response'],
                'tokens_used': 0
            }
        
        # Consolidate multiple responses
        try:
            responses_text = "\n\n".join([
                f"Response from chunk {resp['chunk_number']}: {resp['response']}"
                for resp in relevant_responses
            ])
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that consolidates multiple responses into a single, "
                        "coherent answer. Remove redundancy, synthesize information, and provide a "
                        "comprehensive response to the user's question."
                    )
                }
            ]
            
            if context_str:
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation context:\n{context_str}"
                })
            
            messages.append({
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Multiple responses to consolidate:\n{responses_text}\n\n"
                    f"Please provide a single, comprehensive answer that synthesizes the above responses."
                )
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_response_tokens,
                temperature=0.3
            )
            
            return {
                'answer': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to consolidate responses: {str(e)}")
            # Fallback: return the first relevant response
            return {
                'answer': relevant_responses[0]['response'],
                'tokens_used': 0
            }
    
    def query_pdf(self, file_content: bytes, question: str, 
                  conversation_id: str = None) -> Dict[str, Any]:
        """
        Complete workflow to query a PDF with a question.
        
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
        
        # Chunk the text
        chunks = self.chunk_text(extraction_result['text'])
        
        # Get conversation context
        context = self.conversation_contexts.get(conversation_id, []) if conversation_id else []
        
        # Query the chunks
        query_result = self.query_pdf_chunks(chunks, question, context)
        
        # Update conversation context
        if conversation_id:
            if conversation_id not in self.conversation_contexts:
                self.conversation_contexts[conversation_id] = []
            
            self.conversation_contexts[conversation_id].append({
                'question': question,
                'answer': query_result['consolidated_answer'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges
            self.conversation_contexts[conversation_id] = \
                self.conversation_contexts[conversation_id][-10:]
        
        return {
            'success': True,
            'question': question,
            'answer': query_result['consolidated_answer'],
            'metadata': {
                'total_pages': extraction_result['total_pages'],
                'total_chunks': query_result['total_chunks_processed'],
                'total_tokens_used': query_result['total_tokens_used'],
                'model_used': query_result['model_used'],
                'processing_time': query_result['processing_time']
            },
            'conversation_id': conversation_id
        }
    
    def clear_conversation_context(self, conversation_id: str):
        """Clear conversation context for a specific conversation."""
        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a specific conversation."""
        return self.conversation_contexts.get(conversation_id, [])
