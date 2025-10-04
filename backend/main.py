"""
FastAPI Backend Server for Databricks PDF Processing
Replaces Streamlit with REST API endpoints
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.databricks_api import DatabricksAPIIntegration
from src.pdf_manager import PDFManager
from src.databricks_ai_engine import DatabricksAIEngine
from config import Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Databricks PDF Processing API",
    description="REST API for PDF upload, processing, and AI-powered querying using Databricks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001","http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for connections
databricks_api: Optional[DatabricksAPIIntegration] = None
pdf_manager: Optional[PDFManager] = None
ai_engine: Optional[DatabricksAIEngine] = None

def clean_result_for_json(obj):
    """
    Recursively clean a result object to remove bytes and other non-JSON serializable objects.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if key == 'file_content' and isinstance(value, bytes):
                # Skip file_content bytes to avoid JSON serialization error
                cleaned[key] = f"<bytes object: {len(value)} bytes>"
            else:
                cleaned[key] = clean_result_for_json(value)
        return cleaned
    elif isinstance(obj, list):
        return [clean_result_for_json(item) for item in obj]
    elif isinstance(obj, bytes):
        return f"<bytes object: {len(obj)} bytes>"
    else:
        return obj

# Pydantic models for request/response
class ConnectionConfig(BaseModel):
    host: str
    token: str

class AIConfig(BaseModel):
    provider: str  # "databricks" or "openai"
    model: str
    cluster_id: Optional[str] = None
    openai_api_key: Optional[str] = None

class ChatMessage(BaseModel):
    question: str
    pdf_path: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Dependency to get databricks connection
async def get_databricks_connection():
    global databricks_api
    if not databricks_api:
        raise HTTPException(status_code=400, detail="Databricks connection not established")
    return databricks_api

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Databricks PDF Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "databricks_connected": databricks_api is not None
    }
@app.get("/api/databricks/setup")
async def setup_system():
    """Initialize Databricks connection and configure AI in one step."""
    global databricks_api, pdf_manager, ai_engine

    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    provider = os.getenv("AI_PROVIDER", "databricks")
    model = os.getenv("AI_MODEL", "databricks-gpt-oss-120b")

    try:
        # Step 1: Connect to Databricks
        databricks_api = DatabricksAPIIntegration(DATABRICKS_HOST, DATABRICKS_TOKEN)
        connection_result = databricks_api.test_connection()

        if not connection_result["success"]:
            return {
                "success": False,
                "stage": "databricks_connect",
                "error": connection_result.get("error"),
            }

        # Step 2: Initialize PDF Manager
        pdf_manager = PDFManager(databricks_api.client)

        # Step 3: Get workspace info (clusters etc.)
        try:
            clusters = databricks_api.get_cluster_info()
        except Exception as e:
            clusters = f"Failed to fetch clusters: {str(e)}"

        # Step 4: Configure AI
        ai_config_result = {}
        if provider == "databricks":
            ai_engine = DatabricksAIEngine(databricks_client=databricks_api.client, model=model)
            ai_config_result = {
                "success": True,
                "provider": "databricks",
                "model": model,
                "message": "Databricks AI configured successfully",
            }

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                ai_config_result = {"success": False, "error": "OPENAI_API_KEY missing in env"}
            else:
                os.environ["OPENAI_API_KEY"] = api_key
                # (Optional: instantiate OpenAI client here)
                ai_config_result = {
                    "success": True,
                    "provider": "openai",
                    "model": model,
                    "message": "OpenAI configured successfully",
                }

        else:
            ai_config_result = {"success": False, "error": f"Unsupported AI provider: {provider}"}

        # Step 5: Return combined response
        return {
            "success": True,
            "databricks": {
                "connected": True,
                "user": connection_result.get("user"),
                "workspace_url": connection_result.get("workspace_url"),
                "clusters": clusters,
            },
            "ai": ai_config_result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/api/pdf/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    create_notebook: bool = Form(False),
    db: DatabricksAPIIntegration = Depends(get_databricks_connection)
):
    """Upload PDF to Databricks workspace using proven method."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Read file content
        file_content = await file.read()

        # Use the proven upload method from single-page-app
        success = await upload_pdf_direct_method(
            file_content=file_content,
            filename=file.filename,
            db=db
        )

        if success:
            return {
                'success': True,
                'pdf_path': f"/Workspace/Shared/pdf_uploads/{file.filename}",
                'filename': file.filename,
                'size': len(file_content),
                'upload_method': 'direct_workspace_api',
                'message': f'PDF uploaded successfully to /Workspace/Shared/pdf_uploads/{file.filename}'
            }
        else:
            raise HTTPException(status_code=500, detail="Upload failed")

    except Exception as e:
        logger.error(f"Failed to upload PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def upload_pdf_direct_method(file_content: bytes, filename: str, db: DatabricksAPIIntegration) -> bool:
    """
    Upload PDF using the proven method from single-page-app (demo-try2.py)
    """
    try:
        import base64
        import requests

        # Get Databricks credentials
        host = db.client.host.rstrip('/')
        token = db.client.token

        # Prepare the API call (same as demo-try2.py)
        url = f"{host}/api/2.0/workspace/import"
        headers = {"Authorization": f"Bearer {token}"}

        # Base64 encode the file content
        b64_data = base64.b64encode(file_content).decode("utf-8")

        data = {
            "path": f"/Workspace/Shared/pdf_uploads/{filename}",
            "overwrite": True,
            "format": "AUTO",      # AUTO = auto-detect notebook type
            "language": "PYTHON",  # Needed for notebooks, ignored for binary files
            "content": b64_data
        }

        # Make the API call
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        logger.info(f"Direct upload successful for {filename} ✅")
        return True

    except Exception as e:
        logger.error(f"Direct upload failed for {filename}: {str(e)}")
        return False


    """Analyze PDF using the same method as single-page-app."""
    try:
        # Import the single-page-app logic
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'single-page-app'))

        from databricks_ai import DatabricksAI

        # Get Databricks credentials
        host = db.client.host.rstrip('/')
        token = db.client.token

        # Create AI client (same as single-page-app)
        ai_client = DatabricksAI(host, token)

        # Analyze PDF (same method as single-page-app)
        result = ai_client.analyze_pdf(message.pdf_path, message.question)

        return {
            "success": result.get('success', False),
            "answer": result.get('answer', ''),
            "conversation_id": message.conversation_id,
            "error": result.get('error'),
            "metadata": {
                "pdf_path": result.get('pdf_path'),
                "pages_analyzed": result.get('pages_analyzed'),
                "text_length": result.get('text_length')
            }
        }

    except Exception as e:
        logger.error(f"Direct PDF analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "conversation_id": message.conversation_id
        }
@app.post("/api/analyze/pdf")
async def analyze_pdf_direct(
    db: DatabricksAPIIntegration = Depends(get_databricks_connection)
):
    """Analyze PDF with hardcoded file path and multiple prompts."""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'single-page-app'))
        from databricks_ai import DatabricksAI

      
        pdf_path = "/Workspace/Shared/pdf_uploads/Principal-Sample-Life-Insurance-Policy.pdf"  
        prompts = [
            "Summarize the key points of this document.",
            "Extract all policy numbers and their coverage details.",
            "List all important dates (issue date, renewal date, expiry date)."
        ]

        # Create AI client
        host = db.client.host.rstrip('/')
        token = db.client.token
        ai_client = DatabricksAI(host, token)

        responses = []
        merged_text = ""

        # Run all prompts
        for prompt in prompts:
            result = ai_client.analyze_pdf(pdf_path, prompt)
            responses.append({
                "prompt": prompt,
                "answer": result.get("answer", ""),
                "success": result.get("success", False),
                "error": result.get("error")
            })
            merged_text += f"\nPrompt: {prompt}\nAnswer: {result.get('answer', '')}\n"

        # Create a merged summary (AI-assisted or just concatenation)
        # You could call AI again with merged_text for a polished summary
        merged_summary = f"Combined Analysis:\n{merged_text}"

        return {
            "success": True,
            "pdf_path": pdf_path,
            "responses": responses,
            "merged_summary": merged_summary
        }

    except Exception as e:
        logger.error(f"Direct PDF analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
@app.post("/api/pdf/upload-and-analyze")
async def upload_and_analyze_pdf(
    file: UploadFile = File(...),
    db: DatabricksAPIIntegration = Depends(get_databricks_connection)
):
    """Upload a PDF and immediately analyze it with hardcoded prompts."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Read file content
        file_content = await file.read()

        # Upload to Databricks Workspace
        success = await upload_pdf_direct_method(
            file_content=file_content,
            filename=file.filename,
            db=db
        )

        if not success:
            raise HTTPException(status_code=500, detail="PDF upload failed")

        # Construct path used for analysis
        pdf_path = f"/Workspace/Shared/pdf_uploads/{file.filename}"

        # Hardcoded prompts
        prompts = [
            "Summarize the document",
            "Extract key financial terms",
            "Identify risks mentioned"
        ]

        # Import AI client
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'single-page-app'))
        from databricks_ai import DatabricksAI

        host = db.client.host.rstrip('/')
        token = db.client.token
        ai_client = DatabricksAI(host, token)

        responses = []
        for prompt in prompts:
            try:
                result = ai_client.analyze_pdf(pdf_path, prompt)
                responses.append({
                    "prompt": prompt,
                    "answer": result.get("answer", ""),
                    "success": result.get("success", False),
                    "error": result.get("error")
                })
            except Exception as e:
                responses.append({
                    "prompt": prompt,
                    "answer": "",
                    "success": False,
                    "error": str(e)
                })

        # Merge results (simple join)
        merged_summary = " ".join([r["answer"] for r in responses if r["success"]])

        return {
            "success": True,
            "upload": {
                "pdf_path": pdf_path,
                "filename": file.filename,
                "size": len(file_content),
                "upload_method": "direct_workspace_api",
                "message": f"PDF uploaded successfully to {pdf_path}"
            },
            "analysis": {
                "responses": responses,
                "merged_summary": merged_summary
            }
        }

    except Exception as e:
        logger.error(f"Upload + Analyze failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
