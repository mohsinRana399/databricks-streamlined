# Databricks notebook source
# MAGIC %md
# MAGIC # PDF Processing Template
# MAGIC 
# MAGIC This notebook demonstrates how to process PDF files uploaded to Databricks workspace.
# MAGIC It integrates with the existing text chunking and OpenAI processing workflow.
# MAGIC 
# MAGIC **Features:**
# MAGIC - Read PDF files from workspace
# MAGIC - Extract and chunk text content
# MAGIC - Process with OpenAI API
# MAGIC - Extract structured data (insurance quotes)
# MAGIC - Save results to Delta tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# Install required packages if not available
%pip install PyPDF2 openai tiktoken python-dotenv

# COMMAND ----------

import os
import base64
import pandas as pd
from datetime import datetime
import PyPDF2
from io import BytesIO

# Import existing modules (assuming they're available in the workspace)
import openai
import tiktoken

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - Update these values
PDF_WORKSPACE_PATH = "/Workspace/Shared/pdf_uploads/your_file.pdf"  # Update with actual path
OPENAI_API_KEY = "your-openai-api-key"  # Set your OpenAI API key
OUTPUT_TABLE_NAME = "pdf_processing_results"  # Delta table for results

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Processing Functions

# COMMAND ----------

def read_pdf_from_workspace(workspace_path):
    """
    Read PDF content from Databricks workspace.
    
    Args:
        workspace_path: Path to PDF in workspace
        
    Returns:
        PDF content as bytes
    """
    try:
        # Read file content from workspace
        # Note: This is a simplified example. In practice, you might need to:
        # 1. Copy from workspace to DBFS using dbutils.fs.cp
        # 2. Or use workspace API to read the file
        
        # For demonstration, assuming file is accessible
        with open(workspace_path.replace('/Workspace', '/dbfs/mnt/workspace'), 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_text_from_pdf(pdf_content):
    """
    Extract text from PDF content.
    
    Args:
        pdf_content: PDF content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def chunk_text(text, model="gpt-4o", max_tokens=12000):
    """
    Chunk text into smaller pieces for processing.
    (Reusing the function from demo.py)
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def process_chunks_with_openai(chunks, model="gpt-3.5-turbo"):
    """
    Process text chunks with OpenAI API.
    (Enhanced version of the demo.py logic)
    """
    results = []
    
    for i, chunk in enumerate(chunks, 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts structured data."},
                    {"role": "user", "content": (
                        "Extract all insurance quote details from the following text "
                        "and present them in a markdown table format.\n\n"
                        f"Chunk {i}:\n{chunk}"
                    )}
                ],
                max_tokens=1500
            )
            
            result = {
                'chunk_number': i,
                'content': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'processing_time': datetime.now()
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            results.append({
                'chunk_number': i,
                'content': f"Error: {str(e)}",
                'tokens_used': 0,
                'processing_time': datetime.now()
            })
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Processing Workflow

# COMMAND ----------

def process_pdf_workflow(pdf_path, output_table=None):
    """
    Complete PDF processing workflow.
    
    Args:
        pdf_path: Path to PDF in workspace
        output_table: Optional table name to save results
        
    Returns:
        Processing results
    """
    print(f"Starting PDF processing workflow for: {pdf_path}")
    print(f"Timestamp: {datetime.now()}")
    
    # Step 1: Read PDF
    print("\n1. Reading PDF from workspace...")
    pdf_content = read_pdf_from_workspace(pdf_path)
    if not pdf_content:
        return {"error": "Failed to read PDF"}
    
    # Step 2: Extract text
    print("2. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_content)
    if not text:
        return {"error": "Failed to extract text"}
    
    print(f"   Extracted {len(text)} characters")
    
    # Step 3: Chunk text
    print("3. Chunking text for processing...")
    chunks = chunk_text(text, model="gpt-3.5-turbo", max_tokens=3000)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 4: Process with OpenAI
    print("4. Processing chunks with OpenAI...")
    results = process_chunks_with_openai(chunks)
    
    # Step 5: Compile results
    print("5. Compiling results...")
    workflow_result = {
        'pdf_path': pdf_path,
        'processing_timestamp': datetime.now(),
        'total_chunks': len(chunks),
        'total_characters': len(text),
        'results': results,
        'success': True
    }
    
    # Step 6: Save to table (optional)
    if output_table:
        print(f"6. Saving results to table: {output_table}")
        save_results_to_table(workflow_result, output_table)
    
    return workflow_result

def save_results_to_table(workflow_result, table_name):
    """
    Save processing results to a Delta table.
    
    Args:
        workflow_result: Results from PDF processing
        table_name: Name of the Delta table
    """
    try:
        # Prepare data for DataFrame
        rows = []
        for result in workflow_result['results']:
            rows.append({
                'pdf_path': workflow_result['pdf_path'],
                'processing_timestamp': workflow_result['processing_timestamp'],
                'chunk_number': result['chunk_number'],
                'extracted_content': result['content'],
                'tokens_used': result['tokens_used'],
                'chunk_processing_time': result['processing_time']
            })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        spark_df = spark.createDataFrame(df)
        
        # Write to Delta table
        spark_df.write.format("delta").mode("append").saveAsTable(table_name)
        print(f"Results saved to table: {table_name}")
        
    except Exception as e:
        print(f"Error saving to table: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Processing

# COMMAND ----------

# Execute the workflow
try:
    results = process_pdf_workflow(
        pdf_path=PDF_WORKSPACE_PATH,
        output_table=OUTPUT_TABLE_NAME
    )
    
    if results.get('success'):
        print("\n" + "="*50)
        print("PDF PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Display summary
        print(f"PDF Path: {results['pdf_path']}")
        print(f"Processing Time: {results['processing_timestamp']}")
        print(f"Total Chunks: {results['total_chunks']}")
        print(f"Total Characters: {results['total_characters']}")
        
        # Display extracted content
        print("\n" + "="*30)
        print("EXTRACTED CONTENT:")
        print("="*30)
        
        for i, result in enumerate(results['results'], 1):
            print(f"\n=== Table from Chunk {i} ===")
            print(result['content'])
            print(f"Tokens used: {result['tokens_used']}")
    else:
        print(f"Processing failed: {results.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"Workflow execution failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Analysis and Visualization

# COMMAND ----------

# Query results from Delta table (if saved)
try:
    df = spark.sql(f"SELECT * FROM {OUTPUT_TABLE_NAME} ORDER BY processing_timestamp DESC, chunk_number")
    display(df)
except:
    print("No results table found or error querying results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup and Next Steps
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Customize the text extraction logic for your specific PDF format
# MAGIC 2. Modify the OpenAI prompts for your use case
# MAGIC 3. Add data validation and quality checks
# MAGIC 4. Implement error handling and retry logic
# MAGIC 5. Set up automated scheduling if needed
# MAGIC 
# MAGIC **Notes:**
# MAGIC - Update the `PDF_WORKSPACE_PATH` variable with your actual PDF path
# MAGIC - Set your OpenAI API key in the configuration section
# MAGIC - Modify the processing logic as needed for your specific requirements
