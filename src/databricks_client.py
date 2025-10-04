"""
Databricks client module for handling connections and file operations.
"""
import os
import base64
import logging
from typing import Optional, Dict, Any, List
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service import workspace
import requests

logger = logging.getLogger(__name__)


class DatabricksClient:
    """Client for interacting with Databricks workspace and APIs."""
    
    def __init__(self, host: str = None, token: str = None):
        """
        Initialize Databricks client.
        
        Args:
            host: Databricks workspace URL
            token: Personal access token
        """
        self.host = host or os.getenv('DATABRICKS_HOST')
        self.token = token or os.getenv('DATABRICKS_TOKEN')
        
        if not self.host or not self.token:
            raise ValueError("Databricks host and token must be provided")
        
        # Initialize the workspace client
        self.config = Config(host=self.host, token=self.token)
        self.workspace_client = WorkspaceClient(config=self.config)
        
        # Set up headers for direct API calls
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Databricks workspace.
        
        Returns:
            Dict with connection status and user info
        """
        try:
            current_user = self.workspace_client.current_user.me()
            return {
                'success': True,
                'user': current_user.user_name,
                'workspace_url': self.host
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def upload_file_to_workspace(self, file_content: bytes, workspace_path: str,
                                overwrite: bool = True) -> Dict[str, Any]:
        """
        Upload a file to Databricks workspace or DBFS.

        Args:
            file_content: File content as bytes
            workspace_path: Target path in workspace
            overwrite: Whether to overwrite existing files

        Returns:
            Dict with upload status and details
        """
        try:
            logger.info(f"Uploading file to {workspace_path}, size: {len(file_content)} bytes")

            # For PDF files, we need to handle them as binary files
            # The workspace API expects base64 encoded content, but it will base64 encode it again
            # So we need to upload the raw binary content and let the API handle the encoding

            if workspace_path.lower().endswith('.pdf'):
                logger.info(f"Uploading PDF as binary file to workspace: {workspace_path}")

                # For PDF files, use proper base64 encoding to avoid corruption
                # The latin1 method was causing UTF-8 re-encoding corruption

                try:
                    # Method 1: Use base64 encoding (proper approach for binary files)
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    logger.info(f"Encoded PDF content: {len(file_content)} bytes -> {len(encoded_content)} base64 chars")

                    self.workspace_client.workspace.upload(
                        path=workspace_path,
                        content=encoded_content,
                        format=workspace.ImportFormat.AUTO,
                        overwrite=overwrite
                    )
                    logger.info(f"PDF uploaded successfully using base64 encoding")

                    return {
                        'success': True,
                        'path': workspace_path,
                        'message': f'PDF uploaded successfully to {workspace_path}',
                        'upload_method': 'workspace_base64'
                    }
                except Exception as base64_error:
                    logger.error(f"Base64 upload failed: {base64_error}")

                    # Method 2: Fallback - try with DBC format (for very problematic cases)
                    try:
                        encoded_content = base64.b64encode(file_content).decode('utf-8')

                        self.workspace_client.workspace.upload(
                            path=workspace_path,
                            content=encoded_content,
                            format=workspace.ImportFormat.DBC,
                            overwrite=overwrite
                        )
                        logger.info(f"PDF uploaded successfully using DBC format fallback")

                        return {
                            'success': True,
                            'path': workspace_path,
                            'message': f'PDF uploaded successfully to {workspace_path} (DBC fallback)',
                            'upload_method': 'workspace_dbc_fallback'
                        }
                    except Exception as dbc_error:
                        logger.error(f"DBC fallback upload failed: {dbc_error}")
                        raise Exception(f"All PDF upload methods failed. Base64: {base64_error}, DBC: {dbc_error}")

                    return {
                        'success': True,
                        'path': workspace_path,
                        'message': f'PDF uploaded successfully to {workspace_path} (base64)',
                        'upload_method': 'workspace_base64'
                    }
            else:
                # For other files, use standard base64 encoding
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                self.workspace_client.workspace.upload(
                    path=workspace_path,
                    content=encoded_content,
                    format=workspace.ImportFormat.AUTO,
                    overwrite=overwrite
                )
                logger.info(f"File uploaded successfully using workspace AUTO format")

                return {
                    'success': True,
                    'path': workspace_path,
                    'message': f'File uploaded successfully to {workspace_path}',
                    'upload_method': 'workspace'
                }
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_notebook_from_template(self, notebook_path: str, template_content: str,
                                    overwrite: bool = True) -> Dict[str, Any]:
        """
        Create a notebook in Databricks workspace.
        
        Args:
            notebook_path: Path for the new notebook
            template_content: Notebook content
            overwrite: Whether to overwrite existing notebook
            
        Returns:
            Dict with creation status
        """
        try:
            self.workspace_client.workspace.upload(
                path=notebook_path,
                content=base64.b64encode(template_content.encode()).decode(),
                format=workspace.ImportFormat.AUTO,
                overwrite=overwrite
            )
            
            return {
                'success': True,
                'path': notebook_path,
                'message': f'Notebook created successfully at {notebook_path}'
            }
            
        except Exception as e:
            logger.error(f"Notebook creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_workspace_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files in a workspace directory and DBFS.

        Args:
            path: Workspace path to list

        Returns:
            List of file information dictionaries
        """
        files = []

        try:
            # List workspace files
            objects = self.workspace_client.workspace.list(path)

            for obj in objects:
                files.append({
                    'path': obj.path,
                    'object_type': obj.object_type.value if obj.object_type else 'unknown',
                    'language': obj.language.value if obj.language else None,
                    'source': 'workspace'
                })

        except Exception as e:
            logger.warning(f"Failed to list workspace files: {str(e)}")

        # Note: DBFS listing removed since DBFS is disabled in this workspace

        return files
    
    def export_workspace_file(self, workspace_path: str) -> Optional[bytes]:
        """
        Export/download a file from Databricks workspace or DBFS.

        Args:
            workspace_path: Path to file in workspace

        Returns:
            File content as bytes or None if failed
        """
        try:
            # Use workspace export for all files (DBFS is disabled)
            logger.info(f"Trying workspace export for: {workspace_path}")

            # Use the workspace export API to get file content
            # Try different export formats for PDF files
            exported_content = None

            # For PDF files, try SOURCE format first (raw binary)
            if workspace_path.lower().endswith('.pdf'):
                try:
                    exported_content = self.workspace_client.workspace.export(
                        path=workspace_path,
                        format=workspace.ExportFormat.SOURCE
                    )
                except Exception as e:
                    logger.warning(f"SOURCE format failed for {workspace_path}: {e}")

            # Fallback to other formats if SOURCE fails
            if not exported_content:
                for format_type in [workspace.ExportFormat.SOURCE, workspace.ExportFormat.HTML, workspace.ExportFormat.JUPYTER]:
                    try:
                        exported_content = self.workspace_client.workspace.export(
                            path=workspace_path,
                            format=format_type
                        )
                        if exported_content and exported_content.content:
                            logger.info(f"Successfully exported {workspace_path} using {format_type}")
                            break
                    except Exception as e:
                        logger.debug(f"Export format {format_type} failed for {workspace_path}: {e}")
                        continue

            if exported_content and exported_content.content:
                # The content might be base64 encoded string or already bytes
                try:
                    # Check content type and handle base64 decoding
                    logger.info(f"Content type: {type(exported_content.content)}")
                    logger.info(f"Content length: {len(exported_content.content) if exported_content.content else 0}")

                    if exported_content.content:
                        # Show first 50 characters/bytes for debugging
                        if isinstance(exported_content.content, bytes):
                            logger.info(f"Content (bytes) first 50: {exported_content.content[:50]}")
                        else:
                            logger.info(f"Content (string) first 50: {exported_content.content[:50]}")

                    # Handle both bytes and string content
                    content_to_decode = exported_content.content

                    # If it's bytes, convert to string for base64 decoding
                    if isinstance(content_to_decode, bytes):
                        content_to_decode = content_to_decode.decode('utf-8')

                    # Check if it looks like base64 encoded PDF
                    if content_to_decode.startswith('JVBERi'):  # '%PDF' in base64
                        logger.info(f"Detected base64 encoded PDF, decoding...")
                        try:
                            file_content = base64.b64decode(content_to_decode)
                            logger.info(f"Successfully decoded base64 content from {workspace_path} ({len(file_content)} bytes)")
                            logger.info(f"Decoded content starts with: {file_content[:20]}")
                            return file_content
                        except Exception as decode_error:
                            logger.error(f"Base64 decode failed: {decode_error}")
                            return None
                    elif content_to_decode.startswith('SlZCRVJp'):  # Double base64 encoded PDF
                        logger.info(f"Detected DOUBLE base64 encoded PDF, decoding twice...")
                        try:
                            # First decode
                            first_decode = base64.b64decode(content_to_decode).decode('utf-8')
                            logger.info(f"First decode result starts with: {first_decode[:20]}")

                            # Second decode
                            file_content = base64.b64decode(first_decode)
                            logger.info(f"Successfully double-decoded content from {workspace_path} ({len(file_content)} bytes)")
                            logger.info(f"Final content starts with: {file_content[:20]}")
                            return file_content
                        except Exception as decode_error:
                            logger.error(f"Double base64 decode failed: {decode_error}")
                            return None
                    else:
                        # Not base64 encoded, return as is
                        if isinstance(exported_content.content, bytes):
                            logger.info(f"Content is already binary, returning as-is")
                            return exported_content.content
                        else:
                            logger.info(f"Content is text, encoding as UTF-8")
                            return exported_content.content.encode('utf-8')

                except Exception as decode_error:
                    # If base64 decode fails, the content might be plain text
                    logger.warning(f"Base64 decode failed, trying direct content: {decode_error}")
                    if isinstance(exported_content.content, str):
                        # For text files, encode as UTF-8
                        return exported_content.content.encode('utf-8')
                    else:
                        # Last resort - return as is
                        return exported_content.content
            else:
                logger.warning(f"No content returned from workspace export: {workspace_path}")
                # Try alternative download method for PDFs
                return self._download_file_direct(workspace_path)

        except Exception as e:
            logger.error(f"Failed to export file from {workspace_path}: {str(e)}")
            # Try alternative download method as fallback
            return self._download_file_direct(workspace_path)

    def _download_file_direct(self, workspace_path: str) -> Optional[bytes]:
        """
        Alternative method to download files using direct API calls.

        Args:
            workspace_path: Path to file in workspace

        Returns:
            File content as bytes or None if failed
        """
        try:
            logger.info(f"Attempting direct download of {workspace_path}")

            # Try using the workspace client's download method if available
            try:
                response = self.workspace_client.workspace.download(workspace_path)
                if response:
                    logger.info(f"Successfully downloaded file from {workspace_path} ({len(response)} bytes)")
                    return response
            except AttributeError:
                logger.debug("Download method not available, trying REST API")
            except Exception as e:
                logger.debug(f"Download method failed: {e}")

            # Fallback to REST API call
            return self._download_via_rest_api(workspace_path)

        except Exception as e:
            logger.error(f"Direct download failed for {workspace_path}: {str(e)}")
            return None

    def _download_via_rest_api(self, workspace_path: str) -> Optional[bytes]:
        """
        Download file using direct REST API calls.

        Args:
            workspace_path: Path to file in workspace

        Returns:
            File content as bytes or None if failed
        """
        try:
            logger.info(f"Attempting REST API download of {workspace_path}")

            # Get the workspace URL and token from the client config
            host = self.workspace_client.config.host
            token = self.workspace_client.config.token

            # Construct the export API URL
            url = f"{host}/api/2.0/workspace/export"

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            # Try different formats for PDF files
            for format_type in ['SOURCE', 'AUTO']:
                try:
                    payload = {
                        'path': workspace_path,
                        'format': format_type
                    }

                    response = requests.get(url, headers=headers, params=payload)

                    if response.status_code == 200:
                        result = response.json()
                        if 'content' in result:
                            # Decode base64 content
                            file_content = base64.b64decode(result['content'])
                            logger.info(f"Successfully downloaded via REST API: {workspace_path} ({len(file_content)} bytes)")
                            return file_content
                    else:
                        logger.debug(f"REST API format {format_type} failed with status {response.status_code}")

                except Exception as e:
                    logger.debug(f"REST API format {format_type} failed: {e}")
                    continue

            logger.warning(f"All REST API download methods failed for {workspace_path}")
            return None

        except Exception as e:
            logger.error(f"REST API download failed for {workspace_path}: {str(e)}")
            return None

    def execute_sql_query(self, sql_query: str, warehouse_id: str = None) -> Dict[str, Any]:
        """
        Execute a SQL query using Databricks SQL warehouse.

        Args:
            sql_query: SQL query to execute
            warehouse_id: Optional warehouse ID (uses default if not provided)

        Returns:
            Dict with execution results
        """
        try:
            # Import SQL execution client
            from databricks.sdk.service import sql

            # Get available warehouses if no warehouse_id provided
            if not warehouse_id:
                warehouses = list(self.workspace_client.warehouses.list())
                if not warehouses:
                    return {
                        'success': False,
                        'error': 'No SQL warehouses available'
                    }
                warehouse_id = warehouses[0].id
                logger.info(f"Using warehouse: {warehouse_id}")

            # Check warehouse status and start if needed
            try:
                warehouse_info = self.workspace_client.warehouses.get(warehouse_id)
                logger.info(f"Warehouse state: {warehouse_info.state}")

                if warehouse_info.state == sql.State.STOPPED:
                    logger.info("Warehouse is stopped, starting it...")
                    self.workspace_client.warehouses.start(warehouse_id)
                    logger.info("Warehouse start command sent. It may take 1-2 minutes to start.")
                elif warehouse_info.state == sql.State.STARTING:
                    logger.info("Warehouse is already starting up...")
                elif warehouse_info.state == sql.State.RUNNING:
                    logger.info("Warehouse is running and ready")

            except Exception as warehouse_error:
                logger.warning(f"Could not check/start warehouse: {warehouse_error}")
                # Continue anyway - the query execution will handle warehouse startup

            # Execute the query
            logger.info(f"Executing SQL query on warehouse {warehouse_id}")

            # Create a statement execution with maximum allowed timeout
            statement = self.workspace_client.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=sql_query,
                wait_timeout="50s"  # Maximum allowed timeout
            )

            # Check statement status and handle different states
            logger.info(f"Statement status: {statement.status.state}")

            if statement.status.state == sql.StatementState.SUCCEEDED:
                # Extract results
                result_data = []
                if statement.result and statement.result.data_array:
                    # Debug: Log result structure
                    logger.info(f"Result type: {type(statement.result)}")
                    logger.info(f"Result attributes: {dir(statement.result)}")

                    # Get column names - handle different result formats
                    columns = []
                    try:
                        if hasattr(statement.result, 'schema') and statement.result.schema:
                            columns = [col.name for col in statement.result.schema.columns]
                            logger.info(f"Found schema with {len(columns)} columns")
                        elif hasattr(statement.result, 'manifest') and statement.result.manifest:
                            # Alternative schema location
                            if hasattr(statement.result.manifest, 'schema'):
                                columns = [col.name for col in statement.result.manifest.schema.columns]
                                logger.info(f"Found manifest schema with {len(columns)} columns")
                    except Exception as schema_error:
                        logger.warning(f"Schema extraction failed: {schema_error}")
                        # Fallback: use generic column names
                        if statement.result.data_array:
                            first_row = statement.result.data_array[0]
                            columns = [f"col_{i}" for i in range(len(first_row))]
                            logger.info(f"Using generic column names: {columns}")

                    # Process rows
                    for row in statement.result.data_array:
                        row_dict = {}
                        for i, value in enumerate(row):
                            column_name = columns[i] if i < len(columns) else f"col_{i}"
                            row_dict[column_name] = value
                        result_data.append(row_dict)

                logger.info(f"SQL query executed successfully, {len(result_data)} rows returned")
                return {
                    'success': True,
                    'data': result_data,
                    'statement_id': statement.statement_id,
                    'warehouse_id': warehouse_id
                }
            elif statement.status.state == sql.StatementState.PENDING:
                # Handle pending state - warehouse might be starting up
                logger.warning(f"Query is still pending after timeout. This usually means the warehouse is starting up.")
                logger.info(f"Statement ID: {statement.statement_id}")

                # Try to wait a bit more for warehouse startup
                import time
                max_additional_wait = 60  # Additional 60 seconds
                wait_interval = 5  # Check every 5 seconds

                for i in range(0, max_additional_wait, wait_interval):
                    logger.info(f"Waiting for warehouse startup... ({i+wait_interval}s)")
                    time.sleep(wait_interval)

                    # Check status again
                    try:
                        updated_statement = self.workspace_client.statement_execution.get_statement(statement.statement_id)
                        if updated_statement.status.state == sql.StatementState.SUCCEEDED:
                            logger.info("Query completed after additional wait!")
                            statement = updated_statement
                            break
                        elif updated_statement.status.state in [sql.StatementState.FAILED, sql.StatementState.CANCELED]:
                            logger.error(f"Query failed during additional wait: {updated_statement.status.state}")
                            statement = updated_statement
                            break
                    except Exception as e:
                        logger.warning(f"Failed to check statement status: {e}")
                        continue

                # If still pending after additional wait, return error
                if statement.status.state == sql.StatementState.PENDING:
                    error_msg = "Query timed out - warehouse may be starting up. Please try again in a few minutes."
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'statement_id': statement.statement_id,
                        'suggestion': 'The SQL warehouse might be starting up. Please wait a few minutes and try again.'
                    }

                # If query succeeded after additional wait, process results
                if statement.status.state == sql.StatementState.SUCCEEDED:
                    # Process results (same logic as above)
                    result_data = []
                    if statement.result and statement.result.data_array:
                        # Get column names
                        columns = []
                        try:
                            if hasattr(statement.result, 'schema') and statement.result.schema:
                                columns = [col.name for col in statement.result.schema.columns]
                            elif hasattr(statement.result, 'manifest') and statement.result.manifest:
                                if hasattr(statement.result.manifest, 'schema'):
                                    columns = [col.name for col in statement.result.manifest.schema.columns]
                        except Exception:
                            if statement.result.data_array:
                                first_row = statement.result.data_array[0]
                                columns = [f"col_{i}" for i in range(len(first_row))]

                        # Process rows
                        for row in statement.result.data_array:
                            row_dict = {}
                            for i, value in enumerate(row):
                                column_name = columns[i] if i < len(columns) else f"col_{i}"
                                row_dict[column_name] = value
                            result_data.append(row_dict)

                    logger.info(f"SQL query completed after additional wait, {len(result_data)} rows returned")
                    return {
                        'success': True,
                        'data': result_data,
                        'statement_id': statement.statement_id,
                        'warehouse_id': warehouse_id
                    }

            # Handle other failure states
            error_msg = f"Query failed with state: {statement.status.state}"
            if statement.status.error:
                error_msg += f", Error: {statement.status.error.message}"

            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'statement_id': statement.statement_id
            }

        except Exception as e:
            logger.error(f"Failed to execute SQL query: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get list of available clusters.

        Returns:
            List of cluster information
        """
        try:
            clusters = self.workspace_client.clusters.list()
            cluster_list = []

            for cluster in clusters:
                cluster_list.append({
                    'cluster_id': cluster.cluster_id,
                    'cluster_name': cluster.cluster_name,
                    'state': cluster.state.value if cluster.state else 'unknown',
                    'node_type_id': cluster.node_type_id
                })

            return cluster_list

        except Exception as e:
            logger.error(f"Failed to get clusters: {str(e)}")
            return []
