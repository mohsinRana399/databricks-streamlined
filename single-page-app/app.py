"""
Single Page PDF Chat Application with Databricks AI
Uses the upload method from demo-try2.py
"""
import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv
import base64
import json
from typing import List, Dict, Any
import logging
from databricks_ai import DatabricksAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chat with Databricks AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'databricks_host' not in st.session_state:
    st.session_state.databricks_host = ""
if 'databricks_token' not in st.session_state:
    st.session_state.databricks_token = ""
if 'pdf_list' not in st.session_state:
    st.session_state.pdf_list = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def upload_to_workspace(uploaded_file, workspace_path: str, host: str, token: str) -> bool:
    """
    Uploads a file to the Databricks Workspace folder using the Workspace Import API.
    This is the same method from demo-try2.py
    """
    try:
        url = f"{host}/api/2.0/workspace/import"
        headers = {"Authorization": f"Bearer {token}"}

        # Read bytes from UploadedFile (Streamlit)
        file_data = uploaded_file.read()
        b64_data = base64.b64encode(file_data).decode("utf-8")

        data = {
            "path": workspace_path,
            "overwrite": True,
            "format": "AUTO",      # AUTO = auto-detect notebook type
            "language": "PYTHON",  # Needed for notebooks, ignored for binary files
            "content": b64_data
        }

        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        logger.info("Upload to Workspace successful ✅")
        return True
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        st.error(f"Upload failed: {str(e)}")
        return False

def test_databricks_connection(host: str, token: str) -> bool:
    """Test connection to Databricks"""
    try:
        url = f"{host}/api/2.0/clusters/list"
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

def list_workspace_pdfs(host: str, token: str, folder_path: str = "/Workspace/Shared/pdf_uploads") -> List[Dict]:
    """List PDF files in the workspace folder"""
    try:
        url = f"{host}/api/2.0/workspace/list"
        headers = {"Authorization": f"Bearer {token}"}
        
        data = {"path": folder_path}
        response = requests.get(url, headers=headers, params=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        pdf_files = []
        
        if 'objects' in result:
            for obj in result['objects']:
                if obj.get('path', '').lower().endswith('.pdf'):
                    pdf_files.append({
                        'name': os.path.basename(obj['path']),
                        'path': obj['path'],
                        'object_type': obj.get('object_type', 'FILE')
                    })
        
        return pdf_files
    except Exception as e:
        logger.error(f"Failed to list PDFs: {str(e)}")
        return []

def query_pdf_with_ai(host: str, token: str, pdf_path: str, question: str) -> Dict[str, Any]:
    """
    Query PDF using Databricks AI
    """
    try:
        ai_client = DatabricksAI(host, token)
        result = ai_client.analyze_pdf(pdf_path, question)
        return result
    except Exception as e:
        logger.error(f"AI query failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'question': question,
            'pdf_path': pdf_path
        }

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📄 PDF Chat with Databricks AI</h1>', unsafe_allow_html=True)
    
    # Section 1: Databricks Connection
    st.markdown('<h2 class="section-header">🔗 Databricks Connection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input(
            "Databricks Host", 
            value=os.getenv('DATABRICKS_HOST', ''),
            placeholder="https://your-workspace.cloud.databricks.com",
            help="Your Databricks workspace URL"
        )
    
    with col2:
        token = st.text_input(
            "Databricks Token", 
            value=os.getenv('DATABRICKS_TOKEN', ''),
            type="password",
            placeholder="dapi...",
            help="Your Databricks personal access token"
        )
    
    col3, col4, col5 = st.columns([1, 1, 2])
    
    with col3:
        if st.button("🔌 Connect", type="primary"):
            if host and token:
                with st.spinner("Testing connection..."):
                    if test_databricks_connection(host, token):
                        st.session_state.connected = True
                        st.session_state.databricks_host = host
                        st.session_state.databricks_token = token
                        st.success("✅ Connected successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Connection failed. Please check your credentials.")
            else:
                st.error("Please enter both host and token.")
    
    with col4:
        if st.button("🔄 Refresh PDFs"):
            if st.session_state.connected:
                with st.spinner("Loading PDFs..."):
                    st.session_state.pdf_list = list_workspace_pdfs(
                        st.session_state.databricks_host, 
                        st.session_state.databricks_token
                    )
                st.rerun()
    
    with col5:
        if st.session_state.connected:
            st.markdown('<div class="status-box success-box">🟢 Connected to Databricks</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box error-box">🔴 Not connected</div>', unsafe_allow_html=True)
    
    # Only show the rest if connected
    if not st.session_state.connected:
        st.info("👆 Please connect to Databricks first to continue.")
        return
    
    # Section 2: File Upload
    st.markdown('<h2 class="section-header">📤 Upload PDF</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=["pdf"],
            help="Upload a PDF file to analyze with AI"
        )
    
    with col2:
        if uploaded_file:
            st.info(f"📄 **{uploaded_file.name}**\n\nSize: {uploaded_file.size:,} bytes")
    
    if uploaded_file:
        if st.button("📤 Upload to Databricks", type="primary"):
            workspace_path = f"/Workspace/Shared/pdf_uploads/{uploaded_file.name}"
            
            with st.spinner("📤 Uploading file to Databricks Workspace..."):
                success = upload_to_workspace(
                    uploaded_file, 
                    workspace_path,
                    st.session_state.databricks_host,
                    st.session_state.databricks_token
                )
            
            if success:
                st.success("✅ Upload successful!")
                # Refresh PDF list
                st.session_state.pdf_list = list_workspace_pdfs(
                    st.session_state.databricks_host, 
                    st.session_state.databricks_token
                )
                st.rerun()
    
    # Section 3: PDF List and Chat
    st.markdown('<h2 class="section-header">💬 Chat with PDFs</h2>', unsafe_allow_html=True)
    
    # Load PDF list if not already loaded
    if not st.session_state.pdf_list:
        with st.spinner("Loading PDFs..."):
            st.session_state.pdf_list = list_workspace_pdfs(
                st.session_state.databricks_host, 
                st.session_state.databricks_token
            )
    
    if st.session_state.pdf_list:
        st.info(f"📊 Found {len(st.session_state.pdf_list)} PDF files in workspace")
        
        # PDF selection and chat
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Select PDF")
            
            pdf_options = [f"{pdf['name']}" for pdf in st.session_state.pdf_list]
            selected_pdf_name = st.selectbox(
                "Choose a PDF to chat with:",
                pdf_options,
                help="Select a PDF file to ask questions about"
            )
            
            if selected_pdf_name:
                selected_pdf = next(pdf for pdf in st.session_state.pdf_list if pdf['name'] == selected_pdf_name)
                st.success(f"📄 Selected: **{selected_pdf['name']}**")
        
        with col2:
            st.subheader("💭 Ask Questions")
            
            # if selected_pdf_name:
            #     question = st.text_area(
            #         "Your question:",
            #         placeholder="What is this document about? Who is the policyholder? What are the key benefits?",
            #         height=100
            #     )
                
            #     if st.button("🤖 Ask AI", type="primary"):
            #         if question.strip():
            #             selected_pdf = next(pdf for pdf in st.session_state.pdf_list if pdf['name'] == selected_pdf_name)
                        
            #             with st.spinner("🤖 AI is analyzing the PDF..."):
            #                 result = query_pdf_with_ai(
            #                     st.session_state.databricks_host,
            #                     st.session_state.databricks_token,
            #                     selected_pdf['path'],
            #                     question
            #                 )
                        
            #             # Add to chat history
            #             st.session_state.chat_history.append({
            #                 'pdf': selected_pdf_name,
            #                 'question': question,
            #                 'result': result,
            #                 'timestamp': time.time()
            #             })
                        
            #             st.rerun()
            #         else:
            #             st.error("Please enter a question.")
            if selected_pdf_name:
                question = "What is the policy number and who is the policyholder?"
                selected_pdf = next(pdf for pdf in st.session_state.pdf_list if pdf['name'] == selected_pdf_name)

                with st.spinner("🤖 AI is analyzing the PDF..."):
                    result = query_pdf_with_ai(
                        st.session_state.databricks_host,
                        st.session_state.databricks_token,
                        selected_pdf['path'],
                        question
                    )
                
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'pdf': selected_pdf_name,
                        'question': question,
                        'result': result,
                        'timestamp': time.time()
                    })
    else:
        st.warning("📭 No PDF files found. Upload a PDF first!")
    
    # Section 4: Chat History
    if st.session_state.chat_history:
        st.markdown('<h2 class="section-header">💬 Chat History</h2>', unsafe_allow_html=True)
        
        # Show recent chats (last 5)
        recent_chats = st.session_state.chat_history[-5:]
        
        for i, chat in enumerate(reversed(recent_chats)):
            with st.expander(f"💭 {chat['question'][:50]}... (PDF: {chat['pdf']})"):
                st.markdown(f"**📄 PDF:** {chat['pdf']}")
                st.markdown(f"**❓ Question:** {chat['question']}")
                
                if chat['result']['success']:
                    st.markdown(f"**🤖 Answer:**")
                    st.markdown(chat['result']['answer'])
                else:
                    st.error(f"❌ Error: {chat['result'].get('error', 'Unknown error')}")
                
                st.caption(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chat['timestamp']))}")
        
        if len(st.session_state.chat_history) > 5:
            st.info(f"Showing 5 most recent chats. Total: {len(st.session_state.chat_history)}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
