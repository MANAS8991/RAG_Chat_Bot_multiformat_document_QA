# rag_chatbot/agents/ingestion_agent.py

import os
from typing import Dict, Any

# Import BaseAgent and MCP components
from agents.base_agent import BaseAgent
from utils.mcp import MCPMessage, message_bus

# Import document parsing utility
from utils.document_parser import parse_document

class IngestionAgent(BaseAgent):
    """
    The IngestionAgent is responsible for parsing uploaded documents
    and preparing their text content for further processing (chunking and embedding).
    """
    def __init__(self):
        # Initialize the base agent with its specific name
        super().__init__("IngestionAgent")
        print("IngestionAgent: Initialized.")

    def handle_message(self, message: MCPMessage):
        """
        Handles incoming MCP messages relevant to document ingestion.

        Supported message types:
        - "UPLOAD_DOCUMENT": Triggered when a user uploads a new document.
                             Payload should contain 'file_path' and 'file_name'.
        """
        print(f"IngestionAgent: Received message of type '{message.type}' from {message.sender} (Trace ID: {message.trace_id})")

        if message.type == "UPLOAD_DOCUMENT":
            self._process_document_upload(message)
        else:
            print(f"IngestionAgent: Unrecognized message type: {message.type}")

    def _process_document_upload(self, message: MCPMessage):
        """
        Processes an UPLOAD_DOCUMENT message by parsing the document
        and sending the raw text to the RetrievalAgent.

        Args:
            message (MCPMessage): The UPLOAD_DOCUMENT message.
        """
        file_path = message.payload.get("file_path")
        file_name = message.payload.get("file_name")
        file_type = message.payload.get("file_type") # e.g., '.pdf', '.docx'

        if not file_path:
            print("IngestionAgent Error: 'file_path' missing in UPLOAD_DOCUMENT payload.")
            return

        print(f"IngestionAgent: Attempting to parse document: {file_name} ({file_path})")
        try:
            # Use the utility function to parse the document
            raw_text = parse_document(file_path)
            print(f"IngestionAgent: Successfully parsed {file_name}. Text length: {len(raw_text)} characters.")

            # Prepare metadata for the document chunks
            source_metadata = {
                "file_name": file_name,
                "file_type": file_type,
                "original_path": file_path # Keep original path for debugging/reference
            }

            # Send a message to the RetrievalAgent with the raw text and metadata.
            # The RetrievalAgent will then handle chunking and embedding.
            response_message = MCPMessage(
                sender=self.name,
                receiver="RetrievalAgent", # Target the RetrievalAgent
                type="INGESTION_COMPLETE",
                trace_id=message.trace_id, # Maintain the same trace ID
                payload={
                    "raw_text": raw_text,
                    "source_metadata": source_metadata
                }
            )
            message_bus.send_message(response_message)

        except ValueError as ve:
            print(f"IngestionAgent Error: Failed to parse document {file_name}: {ve}")
            # Optionally, send an error message back to the Coordinator or UI
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent", # Send error back to coordinator
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": str(ve),
                    "context": f"Failed to parse document: {file_name}"
                }
            )
            message_bus.send_message(error_message)
        except Exception as e:
            print(f"IngestionAgent Unexpected Error: {e}")
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": f"An unexpected error occurred during ingestion: {e}",
                    "context": f"Document: {file_name}"
                }
            )
            message_bus.send_message(error_message)

