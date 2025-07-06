# rag_chatbot/agents/coordinator_agent.py

from typing import Dict, Any, Callable, List

# Import BaseAgent and MCP components
from agents.base_agent import BaseAgent
from utils.mcp import MCPMessage, message_bus
import uuid # For generating trace_ids

class CoordinatorAgent(BaseAgent):
    """
    The CoordinatorAgent acts as the central orchestrator of the RAG system.
    It receives requests from the UI, routes them to the appropriate agents,
    and collects final responses to send back to the UI.
    """
    def __init__(self, ui_callback: Callable[[Dict[str, Any]], None]):
        # Initialize the base agent with its specific name
        super().__init__("CoordinatorAgent")
        # Store a callback function to send responses back to the Streamlit UI.
        self.ui_callback = ui_callback
        # Dictionary to store ongoing conversation states, keyed by trace_id.
        self.conversation_states: Dict[str, Dict[str, Any]] = {}
        print("CoordinatorAgent: Initialized with UI callback.")

    def handle_message(self, message: MCPMessage):
        """
        Handles incoming MCP messages from other agents or the UI.

        Supported message types:
        - "UI_UPLOAD_REQUEST": From UI, contains file details for ingestion.
        - "UI_QUERY_REQUEST": From UI, contains user's question.
        - "INGESTION_COMPLETE_CONFIRMATION": (Optional) From IngestionAgent after successful parse.
        - "RETRIEVAL_RESULT": From RetrievalAgent, contains retrieved chunks.
        - "FINAL_RESPONSE": From LLMResponseAgent, contains the generated answer.
        - "ERROR_MESSAGE": From any agent, indicating an error.
        """
        print(f"CoordinatorAgent: Received message of type '{message.type}' from {message.sender} (Trace ID: {message.trace_id})")

        if message.type == "UI_UPLOAD_REQUEST":
            self._handle_ui_upload_request(message)
        elif message.type == "UI_QUERY_REQUEST":
            self._handle_ui_query_request(message)
        elif message.type == "INGESTION_COMPLETE":
            # This message is typically handled by RetrievalAgent, but Coordinator can log/monitor
            print(f"CoordinatorAgent: Ingestion for {message.payload.get('source_metadata', {}).get('file_name')} complete.")
            # No direct action needed here, as RetrievalAgent will pick it up.
        elif message.type == "RETRIEVAL_RESULT":
            # This message is typically handled by LLMResponseAgent, but Coordinator can log/monitor
            print(f"CoordinatorAgent: Retrieval complete for query '{message.payload.get('query')}'.")
            # No direct action needed here, as LLMResponseAgent will pick it up.
        elif message.type == "FINAL_RESPONSE":
            self._handle_final_response(message)
        elif message.type == "ERROR_MESSAGE":
            self._handle_error_message(message)
        else:
            print(f"CoordinatorAgent: Unrecognized message type: {message.type}")

    def _handle_ui_upload_request(self, message: MCPMessage):
        """
        Handles a document upload request from the UI.
        Initiates the ingestion process by sending a message to the IngestionAgent.
        """
        file_path = message.payload.get("file_path")
        file_name = message.payload.get("file_name")
        file_type = message.payload.get("file_type")

        if not file_path or not file_name or not file_type:
            print("CoordinatorAgent Error: Missing file details in UI_UPLOAD_REQUEST payload.")
            self._send_ui_error("Missing file details for upload.", message.trace_id)
            return

        print(f"CoordinatorAgent: Routing upload request for '{file_name}' to IngestionAgent.")
        # Create a new trace_id for this specific upload operation
        upload_trace_id = str(uuid.uuid4())
        
        # Store initial state for this trace
        self.conversation_states[upload_trace_id] = {
            "status": "uploading",
            "file_name": file_name,
            "original_query": None # Not a query, but a file upload
        }

        # Send message to IngestionAgent
        ingestion_message = MCPMessage(
            sender=self.name,
            receiver="IngestionAgent",
            type="UPLOAD_DOCUMENT",
            trace_id=upload_trace_id, # Use the new trace ID
            payload={
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_type
            }
        )
        message_bus.send_message(ingestion_message)
        
        # Send an immediate UI update to show processing status
        self.ui_callback({
            "type": "STATUS_UPDATE",
            "status": "Processing document...",
            "file_name": file_name,
            "trace_id": upload_trace_id
        })


    def _handle_ui_query_request(self, message: MCPMessage):
        """
        Handles a user query request from the UI.
        Initiates the retrieval process by sending a message to the RetrievalAgent.
        """
        query = message.payload.get("query")
        if not query:
            print("CoordinatorAgent Error: Missing 'query' in UI_QUERY_REQUEST payload.")
            self._send_ui_error("Your query is empty. Please ask a question.", message.trace_id)
            return

        print(f"CoordinatorAgent: Routing query '{query}' to RetrievalAgent.")
        # Create a new trace_id for this query operation
        query_trace_id = str(uuid.uuid4())

        # Store initial state for this trace
        self.conversation_states[query_trace_id] = {
            "status": "querying",
            "original_query": query,
            "answer": None,
            "source_chunks": [],
            "source_metadata": []
        }
        
        # Send message to RetrievalAgent
        retrieval_message = MCPMessage(
            sender=self.name,
            receiver="RetrievalAgent",
            type="QUERY_REQUEST",
            trace_id=query_trace_id, # Use the new trace ID
            payload={
                "query": query
            }
        )
        message_bus.send_message(retrieval_message)

        # Send an immediate UI update to show processing status
        self.ui_callback({
            "type": "STATUS_UPDATE",
            "status": "Searching for answers...",
            "original_query": query,
            "trace_id": query_trace_id
        })

    def _handle_final_response(self, message: MCPMessage):
        """
        Handles the final response from the LLMResponseAgent.
        Sends the answer and source context back to the UI.
        """
        trace_id = message.trace_id
        answer = message.payload.get("answer")
        source_chunks = message.payload.get("source_chunks", [])
        source_metadata = message.payload.get("source_metadata", [])
        original_query = message.payload.get("original_query")

        if trace_id in self.conversation_states:
            self.conversation_states[trace_id].update({
                "status": "complete",
                "answer": answer,
                "source_chunks": source_chunks,
                "source_metadata": source_metadata
            })
            print(f"CoordinatorAgent: Final response received for Trace ID: {trace_id}. Sending to UI.")
            # Send the complete response back to the UI via the callback
            self.ui_callback({
                "type": "FINAL_RESPONSE",
                "trace_id": trace_id,
                "query": original_query,
                "answer": answer,
                "source_chunks": source_chunks,
                "source_metadata": source_metadata
            })
            # Clean up the conversation state after completion
            del self.conversation_states[trace_id]
        else:
            print(f"CoordinatorAgent Warning: Received FINAL_RESPONSE for unknown trace ID: {trace_id}")
            # Fallback to send directly if trace_id not found (e.g., for direct testing)
            self.ui_callback({
                "type": "FINAL_RESPONSE",
                "trace_id": trace_id,
                "query": original_query,
                "answer": answer,
                "source_chunks": source_chunks,
                "source_metadata": source_metadata
            })


    def _handle_error_message(self, message: MCPMessage):
        """
        Handles error messages from any agent and sends them to the UI.
        """
        trace_id = message.trace_id
        error_details = message.payload.get("error", "An unknown error occurred.")
        context = message.payload.get("context", "")

        print(f"CoordinatorAgent Error: Received error from {message.sender} (Trace ID: {trace_id}): {error_details} - Context: {context}")

        if trace_id in self.conversation_states:
            self.conversation_states[trace_id].update({
                "status": "error",
                "error": error_details,
                "context": context
            })
            # Send the error back to the UI
            self.ui_callback({
                "type": "ERROR_MESSAGE",
                "trace_id": trace_id,
                "error": error_details,
                "context": context,
                "sender": message.sender
            })
            # Optionally, clean up the state or keep it for debugging
            # del self.conversation_states[trace_id]
        else:
            print(f"CoordinatorAgent Warning: Received ERROR_MESSAGE for unknown trace ID: {trace_id}. Sending directly to UI.")
            self.ui_callback({
                "type": "ERROR_MESSAGE",
                "trace_id": trace_id,
                "error": error_details,
                "context": context,
                "sender": message.sender
            })

    def _send_ui_error(self, error_message: str, trace_id: str = None):
        """Helper to send a generic error message to the UI."""
        self.ui_callback({
            "type": "ERROR_MESSAGE",
            "trace_id": trace_id or str(uuid.uuid4()),
            "error": error_message,
            "context": "CoordinatorAgent initiated error."
        })
