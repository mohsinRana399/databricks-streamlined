"""
Streamlit application for uploading PDFs to Databricks.
"""
import streamlit as st
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.databricks_api import DatabricksAPIIntegration
from utils.pdf_processor import PDFProcessor
from src.pdf_query_engine import PDFQueryEngine
from src.pdf_manager import PDFManager, ConversationManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF Upload & Query with Databricks",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'databricks_client' not in st.session_state:
        st.session_state.databricks_client = None
    if 'databricks_api' not in st.session_state:
        st.session_state.databricks_api = None
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = None
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []
    if 'pdf_query_engine' not in st.session_state:
        st.session_state.pdf_query_engine = None
    if 'pdf_manager' not in st.session_state:
        st.session_state.pdf_manager = None
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'openai_configured' not in st.session_state:
        st.session_state.openai_configured = False

def create_databricks_connection():
    """Create and test Databricks connection."""
    with st.sidebar:
        st.header("🔗 Databricks Configuration")
        
        # Connection inputs
        host = st.text_input(
            "Databricks Host URL",
            value=os.getenv('DATABRICKS_HOST', ''),
            placeholder="https://your-workspace.cloud.databricks.com",
            help="Your Databricks workspace URL"
        )
        
        token = st.text_input(
            "Personal Access Token",
            value=os.getenv('DATABRICKS_TOKEN', ''),
            type="password",
            help="Your Databricks personal access token"
        )
        
        max_file_size = st.slider(
            "Max File Size (MB)",
            min_value=1,
            max_value=100,
            value=int(os.getenv('MAX_FILE_SIZE_MB', 50)),
            help="Maximum allowed file size for uploads"
        )
        
        # Test connection button
        if st.button("🔍 Test Connection", type="primary"):
            if host and token:
                try:
                    with st.spinner("Testing connection..."):
                        client = DatabricksAPIIntegration(host, token, max_file_size)
                        connection_result = client.test_connection()
                        
                        if connection_result['success']:
                            st.session_state.databricks_client = client.client  # Store the raw client
                            st.session_state.databricks_api = client  # Store the API integration
                            st.session_state.connection_status = connection_result
                            st.session_state.pdf_manager = PDFManager(client.client)
                            st.success(f"✅ Connected as {connection_result['user']}")
                        else:
                            st.error(f"❌ Connection failed: {connection_result['error']}")
                            st.session_state.connection_status = connection_result
                            
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
            else:
                st.warning("Please provide both host URL and access token")
        
        # Display connection status
        if st.session_state.connection_status:
            if st.session_state.connection_status['success']:
                st.success("🟢 Connected to Databricks")
                st.info(f"User: {st.session_state.connection_status['user']}")
            else:
                st.error("🔴 Not connected")

        # AI Configuration
        st.header("🤖 AI Configuration")

        # AI Provider Selection
        ai_provider = st.radio(
            "Choose AI Provider",
            options=["Databricks AI", "OpenAI API"],
            index=0,
            help="Select which AI provider to use for PDF querying"
        )

        if ai_provider == "OpenAI API":
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv('OPENAI_API_KEY', ''),
                type="password",
                help="Your OpenAI API key for PDF querying"
            )

            openai_model = st.selectbox(
                "OpenAI Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                index=0,
                help="Choose the OpenAI model for processing"
            )
        else:
            # Databricks AI Configuration
            databricks_ai_model = st.selectbox(
                "Databricks AI Model",
                options=["databricks-gpt-oss-120b", "databricks-llama-2-70b-chat", "databricks-mpt-30b-instruct"],
                index=0,
                help="Choose the Databricks AI model for processing"
            )

            cluster_id = st.text_input(
                "Cluster ID (Optional)",
                value=os.getenv('DATABRICKS_CLUSTER_ID', ''),
                help="Databricks cluster ID for running AI queries"
            )

        if st.button("🔧 Configure AI", type="secondary"):
            if ai_provider == "OpenAI API":
                if openai_api_key:
                    try:
                        st.session_state.pdf_query_engine = PDFQueryEngine(
                            openai_api_key=openai_api_key,
                            model=openai_model
                        )
                        st.session_state.ai_provider = "openai"
                        st.session_state.openai_configured = True
                        st.success("✅ OpenAI configured successfully!")
                    except Exception as e:
                        st.error(f"❌ OpenAI configuration failed: {str(e)}")
                        st.session_state.openai_configured = False
                else:
                    st.warning("Please provide OpenAI API key")
            else:
                # Configure Databricks AI
                if st.session_state.databricks_api:
                    try:
                        # Databricks AI is ready if Databricks connection is established
                        st.session_state.ai_provider = "databricks"
                        st.session_state.databricks_ai_model = databricks_ai_model
                        st.session_state.databricks_cluster_id = cluster_id
                        st.session_state.openai_configured = True  # Reusing this flag for AI readiness
                        st.success("✅ Databricks AI configured successfully!")
                    except Exception as e:
                        st.error(f"❌ Databricks AI configuration failed: {str(e)}")
                        st.session_state.openai_configured = False
                else:
                    st.warning("Please connect to Databricks first")

        # Display OpenAI status
        if st.session_state.openai_configured:
            st.success("🟢 OpenAI Ready")
            # st.info(f"Model: {openai_model}")
        else:
            st.error("🔴 OpenAI Not Configured")

def main_interface():
    """Main application interface with tabs."""
    st.title("📄 PDF Upload & Query with Databricks")
    st.markdown("Upload PDF files to Databricks and ask questions using OpenAI.")

    # Check if connected
    if not st.session_state.databricks_api:
        st.warning("⚠️ Please configure and test your Databricks connection in the sidebar first.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload PDFs", "💬 Chat with PDFs"])

    with tab1:
        upload_interface()

    with tab2:
        chat_interface()

def upload_interface():
    """PDF upload interface."""
    st.header("📤 Upload PDF Files")
    st.markdown("Upload PDF files to your Databricks workspace.")
    
    # Upload section
    st.header("📤 Upload PDF File")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Select a PDF file to upload to Databricks"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Upload options
        st.subheader("Upload Options")
        create_notebook = st.checkbox(
            "Create processing notebook",
            value=True,
            help="Create a Databricks notebook for processing this PDF"
        )
        
        # Upload button
        if st.button("🚀 Upload to Databricks", type="primary"):
            upload_pdf_file(uploaded_file, create_notebook)

def upload_pdf_file(uploaded_file, create_notebook: bool):
    """Handle PDF file upload."""
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Validation
        status_text.text("Validating PDF file...")
        progress_bar.progress(20)
        
        pdf_processor = PDFProcessor()
        validation = pdf_processor.validate_pdf(file_content, uploaded_file.name)
        
        if not validation['is_valid']:
            st.error("❌ PDF validation failed:")
            for error in validation['errors']:
                st.error(f"• {error}")
            return
        
        # Step 2: Upload workflow
        status_text.text("Uploading to Databricks...")
        progress_bar.progress(50)

        if not st.session_state.databricks_api:
            st.error("❌ Databricks API not initialized. Please reconnect to Databricks.")
            return

        workflow_result = st.session_state.databricks_api.upload_pdf_workflow(
            file_content=file_content,
            filename=uploaded_file.name,
            create_processing_notebook=create_notebook
        )
        
        progress_bar.progress(100)
        status_text.text("Upload completed!")
        
        # Display results
        if workflow_result['success']:
            st.success("✅ Upload completed successfully!")
            
            # Show upload details
            st.subheader("📋 Upload Details")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**PDF Path:** `{workflow_result['pdf_path']}`")
                if workflow_result['notebook_path']:
                    st.info(f"**Notebook Path:** `{workflow_result['notebook_path']}`")
            
            with col2:
                if workflow_result['metadata']:
                    metadata = workflow_result['metadata']
                    st.info(f"**Pages:** {metadata.get('page_count', 'Unknown')}")
                    st.info(f"**Author:** {metadata.get('author', 'N/A')}")
                    st.info(f"**Title:** {metadata.get('title', 'N/A')}")
            
            # Add to upload history
            st.session_state.upload_history.append({
                'filename': uploaded_file.name,
                'upload_time': datetime.now(),
                'pdf_path': workflow_result['pdf_path'],
                'notebook_path': workflow_result.get('notebook_path'),
                'success': True
            })

            # Cache the PDF content for querying
            if st.session_state.pdf_manager:
                # Use the workspace path for caching to match retrieval
                workspace_path = workflow_result['pdf_path']
                st.session_state.pdf_manager.cache_pdf_content(
                    workspace_path,
                    file_content
                )
                logger.info(f"Cached PDF content for path: {workspace_path}")
            
        else:
            st.error("❌ Upload failed!")
            for step, result in workflow_result['steps'].items():
                if isinstance(result, dict) and not result.get('success', True):
                    st.error(f"**{step.title()}:** {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        logger.error(f"Upload failed: {str(e)}")

def chat_interface():
    """Chat interface for querying PDFs."""
    st.header("💬 Chat with Your PDFs")

    # Check if AI is configured
    if not st.session_state.openai_configured:
        st.warning("⚠️ Please configure AI provider in the sidebar first.")
        return

    # Display current AI provider
    ai_provider = st.session_state.get('ai_provider', 'openai')
    if ai_provider == 'databricks':
        st.info(f"🤖 Using Databricks AI: {st.session_state.get('databricks_ai_model', 'databricks-gpt-oss-120b')}")
    else:
        st.info("🤖 Using OpenAI API")

    # PDF Selection
    st.subheader("📋 Select PDF to Query")

    if st.session_state.pdf_manager:
        available_pdfs = st.session_state.pdf_manager.list_available_pdfs()

        if not available_pdfs:
            st.info("No PDFs available. Please upload a PDF first.")
            return

        # PDF selection dropdown
        pdf_options = {pdf['display_name']: pdf for pdf in available_pdfs}
        selected_pdf_name = st.selectbox(
            "Choose a PDF to chat with:",
            options=list(pdf_options.keys()),
            help="Select a PDF file to ask questions about"
        )

        if selected_pdf_name:
            selected_pdf = pdf_options[selected_pdf_name]
            st.session_state.selected_pdf = selected_pdf

            # Display PDF info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PDF Name", selected_pdf['filename'])
            with col2:
                st.metric("Status", "Cached" if selected_pdf['cached'] else "Not Cached")
            with col3:
                if selected_pdf['cached']:
                    pdf_info = st.session_state.pdf_manager.get_pdf_info(selected_pdf['workspace_path'])
                    st.metric("Size", f"{pdf_info.get('file_size_mb', 0):.1f} MB")

    # Chat Interface
    st.subheader("💭 Ask Questions")

    # Display chat history
    if st.session_state.chat_messages:
        for message in st.session_state.chat_messages:
            with st.chat_message("user"):
                st.write(message['question'])
            with st.chat_message("assistant"):
                st.write(message['answer'])
                if 'metadata' in message:
                    with st.expander("📊 Response Details"):
                        metadata = message['metadata']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tokens Used", metadata.get('total_tokens_used', 0))
                            st.metric("Chunks Processed", metadata.get('total_chunks', 0))
                        with col2:
                            st.metric("Model", metadata.get('model_used', 'N/A'))
                            st.metric("Pages", metadata.get('total_pages', 0))

    # Chat input
    if st.session_state.selected_pdf:
        question = st.chat_input("Ask a question about the PDF...")

        if question:
            # Display user question
            with st.chat_message("user"):
                st.write(question)

            # Get PDF content
            selected_path = st.session_state.selected_pdf['workspace_path']
            logger.info(f"Trying to get cached content for path: {selected_path}")

            # First try to get cached content
            pdf_content = st.session_state.pdf_manager.get_cached_pdf_content(selected_path)

            # If not cached, try to download from workspace
            if not pdf_content:
                logger.info(f"Content not cached, trying to download from workspace: {selected_path}")
                pdf_content = st.session_state.pdf_manager.get_pdf_content(selected_path, use_cache=False)
                if pdf_content:
                    # Cache it for future use
                    st.session_state.pdf_manager.cache_pdf_content(selected_path, pdf_content)
                    logger.info(f"Downloaded and cached content for: {selected_path}")

            if pdf_content:
                # Process the question
                with st.chat_message("assistant"):
                    with st.spinner("Processing your question..."):
                        try:
                            # Create conversation ID
                            conversation_id = st.session_state.conversation_manager.get_active_conversation()
                            if not conversation_id:
                                conversation_id = st.session_state.conversation_manager.create_conversation(
                                    st.session_state.selected_pdf['workspace_path']
                                )

                            # Query the PDF based on AI provider
                            ai_provider = st.session_state.get('ai_provider', 'openai')

                            if ai_provider == 'databricks':
                                # Use Databricks AI
                                if not st.session_state.databricks_api:
                                    st.error("❌ Databricks API not initialized. Please reconnect to Databricks.")
                                    return

                                result = st.session_state.databricks_api.query_pdf_with_ai(
                                    file_content=pdf_content,
                                    question=question,
                                    conversation_id=conversation_id
                                )
                            else:
                                # Use OpenAI API
                                result = st.session_state.pdf_query_engine.query_pdf(
                                    file_content=pdf_content,
                                    question=question,
                                    conversation_id=conversation_id
                                )

                            if result['success']:
                                st.write(result['answer'])

                                # Add to chat history
                                chat_message = {
                                    'question': question,
                                    'answer': result['answer'],
                                    'metadata': result['metadata'],
                                    'timestamp': datetime.now().isoformat()
                                }
                                st.session_state.chat_messages.append(chat_message)

                                # Update conversation manager
                                st.session_state.conversation_manager.add_message(
                                    conversation_id, question, result['answer']
                                )

                                # Show metadata
                                with st.expander("📊 Response Details"):
                                    metadata = result['metadata']
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if ai_provider == 'databricks':
                                            st.metric("Text Length", metadata.get('text_length', 0))
                                            st.metric("Pages", metadata.get('total_pages', 0))
                                            if 'notebook_path' in result:
                                                st.text_input("Notebook Path", result['notebook_path'], disabled=True)
                                        else:
                                            st.metric("Tokens Used", metadata.get('total_tokens_used', 0))
                                            st.metric("Chunks Processed", metadata.get('total_chunks', 0))
                                    with col2:
                                        st.metric("Model", metadata.get('model_used', 'N/A'))
                                        if ai_provider == 'openai':
                                            st.metric("Pages", metadata.get('total_pages', 0))
                                        else:
                                            st.metric("Processing Time", metadata.get('processing_time', 'N/A'))
                            else:
                                st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"❌ Error processing question: {str(e)}")
                            logger.error(f"Chat query failed: {str(e)}")
            else:
                st.error("❌ PDF content not available. Please re-upload the PDF.")

    # Chat controls
    st.subheader("🔧 Chat Controls")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            if st.session_state.selected_pdf:
                conversation_id = st.session_state.conversation_manager.get_active_conversation()
                if conversation_id:
                    st.session_state.conversation_manager.clear_conversation(conversation_id)
            st.rerun()

    with col2:
        if st.button("🔄 Refresh PDF List"):
            if st.session_state.pdf_manager:
                st.session_state.pdf_manager.clear_cache()
            st.rerun()

def display_upload_history():
    """Display upload history."""
    if st.session_state.upload_history:
        st.header("📚 Upload History")
        
        for i, upload in enumerate(reversed(st.session_state.upload_history)):
            with st.expander(f"📄 {upload['filename']} - {upload['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"PDF Path: {upload['pdf_path']}")
                    if upload['notebook_path']:
                        st.text(f"Notebook: {upload['notebook_path']}")
                with col2:
                    status = "✅ Success" if upload['success'] else "❌ Failed"
                    st.text(f"Status: {status}")

def display_workspace_info():
    """Display workspace information."""
    if st.session_state.databricks_api:
        st.header("🏢 Workspace Information")
        
        # Get cluster info
        try:
            clusters = st.session_state.databricks_api.get_cluster_info()
            if clusters:
                st.subheader("Available Clusters")
                for cluster in clusters[:5]:  # Show first 5 clusters
                    st.text(f"• {cluster['cluster_name']} ({cluster['state']})")
        except Exception as e:
            st.warning(f"Could not fetch cluster information: {str(e)}")

def main():
    """Main application function."""
    initialize_session_state()
    
    # Sidebar for configuration
    create_databricks_connection()
    
    # Main interface
    main_interface()
    
    # Additional sections
    col1, col2 = st.columns(2)
    with col1:
        display_upload_history()
    with col2:
        display_workspace_info()

if __name__ == "__main__":
    main()
