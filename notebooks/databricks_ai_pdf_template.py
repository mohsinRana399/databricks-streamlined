# Databricks notebook source
# MAGIC %md
# MAGIC # PDF Processing with Databricks AI Functions
# MAGIC 
# MAGIC This notebook demonstrates how to process PDF files using Databricks native AI functions.
# MAGIC It replaces external OpenAI API calls with efficient Databricks AI capabilities.
# MAGIC 
# MAGIC **Features:**
# MAGIC - Native Databricks AI functions (ai_query)
# MAGIC - No external API keys required
# MAGIC - Cost-effective processing
# MAGIC - High-performance text analysis
# MAGIC - Integrated with Databricks ecosystem

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder.appName("PDF_AI_Processing").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - Update these values
PDF_WORKSPACE_PATH = "/Workspace/Shared/pdf_uploads/your_file.pdf"  # Update with actual path
AI_MODEL = "databricks-gpt-oss-120b"  # Databricks AI model
OUTPUT_TABLE_NAME = "pdf_ai_processing_results"  # Delta table for results

# AI model parameters
MAX_TOKENS = 8192
TEMPERATURE = 0.7

print(f"Using AI Model: {AI_MODEL}")
print(f"Max Tokens: {MAX_TOKENS}")
print(f"Temperature: {TEMPERATURE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Text Processing Functions

# COMMAND ----------

def process_pdf_text_with_ai(text_content, question, model=AI_MODEL):
    """
    Process PDF text using Databricks AI functions.
    
    Args:
        text_content: Extracted PDF text
        question: Question to ask about the PDF
        model: AI model to use
        
    Returns:
        AI response
    """
    # Create DataFrame with the text
    df = spark.createDataFrame([(text_content,)], ["text"])
    
    # Apply AI function
    df_result = df.selectExpr(f"""
        ai_query(
            '{model}', 
            CONCAT('{question}: ', text), 
            modelParameters => named_struct(
                'max_tokens', {MAX_TOKENS}, 
                'temperature', {TEMPERATURE}
            )
        ) as ai_response
    """)
    
    # Get the result
    result = df_result.collect()
    if result:
        return result[0]['ai_response']
    else:
        return "No response generated"

def extract_structured_data_with_ai(text_content, data_type="insurance quotes"):
    """
    Extract structured data using Databricks AI.
    
    Args:
        text_content: PDF text content
        data_type: Type of data to extract
        
    Returns:
        Structured data in YAML format
    """
    prompt = f"Please extract all {data_type} from this document and return as YAML format"
    return process_pdf_text_with_ai(text_content, prompt)

def summarize_document_with_ai(text_content):
    """
    Summarize document using Databricks AI.
    
    Args:
        text_content: PDF text content
        
    Returns:
        Document summary
    """
    prompt = "Please provide a comprehensive summary of this document including key points, main topics, and important details"
    return process_pdf_text_with_ai(text_content, prompt)

def answer_question_with_ai(text_content, question):
    """
    Answer specific questions about the document.
    
    Args:
        text_content: PDF text content
        question: User question
        
    Returns:
        AI answer
    """
    return process_pdf_text_with_ai(text_content, question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample PDF Text (Replace with actual extracted text)

# COMMAND ----------

# Sample PDF text - replace this with actual extracted text from your PDF
sample_pdf_text = """
This is a sample insurance quote document.

Policy Details:
- Policy Number: INS-2024-001
- Policyholder: John Smith
- Coverage Amount: $500,000
- Premium: $2,400 annually
- Deductible: $1,000

Coverage Types:
- Auto Insurance: $300,000 liability
- Home Insurance: $200,000 property coverage
- Life Insurance: $500,000 term life

Contact Information:
- Agent: Sarah Johnson
- Phone: (555) 123-4567
- Email: sarah.johnson@insurance.com
- Office: 123 Main St, Anytown, ST 12345

Terms and Conditions:
- Policy effective from January 1, 2024
- Annual renewal required
- Claims must be reported within 30 days
- Coverage subject to policy terms and conditions
"""

print(f"Sample text length: {len(sample_pdf_text)} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Processing Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Document Summary

# COMMAND ----------

print("=== DOCUMENT SUMMARY ===")
summary = summarize_document_with_ai(sample_pdf_text)
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Extract Insurance Quote Details

# COMMAND ----------

print("=== INSURANCE QUOTE EXTRACTION ===")
insurance_data = extract_structured_data_with_ai(sample_pdf_text, "insurance quotes")
print(insurance_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Answer Specific Questions

# COMMAND ----------

# Example questions
questions = [
    "What is the policy number?",
    "Who is the policyholder?",
    "What is the total coverage amount?",
    "What is the annual premium?",
    "Who is the insurance agent?",
    "What are the contact details?",
    "When does the policy become effective?",
    "What is the deductible amount?"
]

print("=== QUESTION & ANSWER SESSION ===")
for i, question in enumerate(questions, 1):
    print(f"\nQ{i}: {question}")
    answer = answer_question_with_ai(sample_pdf_text, question)
    print(f"A{i}: {answer}")
    print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced AI Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract All Entities

# COMMAND ----------

print("=== ENTITY EXTRACTION ===")
entities_prompt = "Extract all important entities from this document including names, dates, amounts, phone numbers, addresses, and policy numbers. Format as JSON"
entities = process_pdf_text_with_ai(sample_pdf_text, entities_prompt)
print(entities)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Risk Assessment

# COMMAND ----------

print("=== RISK ASSESSMENT ===")
risk_prompt = "Analyze this insurance document and provide a risk assessment including coverage gaps, recommendations, and potential issues"
risk_analysis = process_pdf_text_with_ai(sample_pdf_text, risk_prompt)
print(risk_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Delta Table

# COMMAND ----------

# Prepare results for storage
results_data = [
    ("summary", summary),
    ("insurance_data", insurance_data),
    ("entities", entities),
    ("risk_analysis", risk_analysis)
]

# Create DataFrame
results_df = spark.createDataFrame(results_data, ["analysis_type", "result"])

# Add metadata
results_df = results_df.withColumn("pdf_path", lit(PDF_WORKSPACE_PATH)) \
                      .withColumn("processing_timestamp", lit(datetime.now())) \
                      .withColumn("ai_model", lit(AI_MODEL)) \
                      .withColumn("max_tokens", lit(MAX_TOKENS)) \
                      .withColumn("temperature", lit(TEMPERATURE))

# Save to Delta table
results_df.write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE_NAME)

print(f"Results saved to table: {OUTPUT_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Query Section

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom Question Processing
# MAGIC 
# MAGIC Use this section to ask custom questions about your PDF

# COMMAND ----------

# Custom question - modify this as needed
custom_question = "What are the key terms and conditions mentioned in this document?"

print(f"Custom Question: {custom_question}")
print("=" * 60)

custom_answer = answer_question_with_ai(sample_pdf_text, custom_question)
print(f"AI Answer: {custom_answer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary and Export

# COMMAND ----------

# Display final results summary
print("=== PROCESSING SUMMARY ===")
print(f"PDF Path: {PDF_WORKSPACE_PATH}")
print(f"AI Model: {AI_MODEL}")
print(f"Processing Time: {datetime.now()}")
print(f"Text Length: {len(sample_pdf_text)} characters")
print(f"Results Table: {OUTPUT_TABLE_NAME}")

# Query the results table
print("\n=== SAVED RESULTS ===")
saved_results = spark.sql(f"SELECT * FROM {OUTPUT_TABLE_NAME} ORDER BY processing_timestamp DESC LIMIT 10")
display(saved_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps and Customization
# MAGIC 
# MAGIC **To customize this notebook:**
# MAGIC 1. Replace `sample_pdf_text` with actual extracted PDF text
# MAGIC 2. Modify questions in the Q&A section
# MAGIC 3. Adjust AI model parameters (temperature, max_tokens)
# MAGIC 4. Add custom analysis functions
# MAGIC 5. Integrate with your specific business logic
# MAGIC 
# MAGIC **Benefits of Databricks AI Functions:**
# MAGIC - No external API keys required
# MAGIC - Cost-effective compared to external APIs
# MAGIC - High performance and scalability
# MAGIC - Integrated with Databricks security
# MAGIC - Native Delta Lake integration
# MAGIC - Built-in monitoring and logging
