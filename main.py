# rag_chatbot/main.py

import streamlit as st
import os
import uuid # For generating unique IDs for uploaded files
from typing import Dict, Any # For type hinting

# Import agents and message bus
from agents.coordinator_agent import CoordinatorAgent
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.mcp import message_bus, MCPMessage

# --- Global Agent Initialization ---
# Initialize agents only once when the Streamlit app starts.
# Use Streamlit's session state to store agent instances.
# This prevents re-initialization on every rerun.

if 'coordinator_agent' not in st.session_state:
    # Initialize the UI message queue
    st.session_state.ui_message_queue = []

    # Callback function for the CoordinatorAgent to send messages back to the UI
    def ui_message_callback(message: Dict[str, Any]):
        # This function is called by the CoordinatorAgent to update the UI
        # We append to a message queue. Streamlit's rerun will then process it.
        st.session_state.ui_message_queue.append(message)
        print(f"UI Callback: Added message to queue. Type: {message.get('type')}, Trace ID: {message.get('trace_id')}, Queue size: {len(st.session_state.ui_message_queue)}")

    st.session_state.ingestion_agent = IngestionAgent()
    st.session_state.retrieval_agent = RetrievalAgent()
    st.session_state.llm_response_agent = LLMResponseAgent()
    st.session_state.coordinator_agent = CoordinatorAgent(ui_message_callback)

    # Ensure all agents are registered with the message bus.
    message_bus.process_queued_messages() # Process any messages queued during agent initialization

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

st.title("📚 Agentic RAG Chatbot for Multi-Format Document QA")
st.markdown("""
Welcome! Upload your PDF, PPTX, CSV, DOCX, TXT, or Markdown files and ask questions about their content.
This chatbot uses an agentic architecture with Model Context Protocol (MCP) for internal communication.
""")

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files...",
        type=["pdf", "pptx", "csv", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Uploaded Files:")
        for uploaded_file in uploaded_files:
            # Create a unique temporary path for each uploaded file
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            temp_file_path = os.path.join("./temp_uploaded_files", unique_filename)

            # Ensure the temporary directory exists
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write(f"- {uploaded_file.name}")

            # Send a UI_UPLOAD_REQUEST message to the CoordinatorAgent
            upload_message = MCPMessage(
                sender="UI",
                receiver="CoordinatorAgent",
                type="UI_UPLOAD_REQUEST",
                payload={
                    "file_path": temp_file_path,
                    "file_name": uploaded_file.name,
                    "file_type": os.path.splitext(uploaded_file.name)[1].lower()
                }
            )
            message_bus.send_message(upload_message)
            st.toast(f"Processing {uploaded_file.name}...", icon="⏳")

# --- Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source_info" in message and message["source_info"]:
            with st.expander("Source Context"):
                for idx, source in enumerate(message["source_info"]):
                    st.markdown(f"**Chunk {idx + 1} from {source.get('file_name', 'Unknown File')}:**")
                    st.code(source.get('content', 'No content available'))
                    st.markdown("---")

# React to user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user message in chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send a UI_QUERY_REQUEST message to the CoordinatorAgent
    query_message = MCPMessage(
        sender="UI",
        receiver="CoordinatorAgent",
        type="UI_QUERY_REQUEST",
        payload={"query": prompt}
    )
    message_bus.send_message(query_message)
    print(f"UI: Sent UI_QUERY_REQUEST for query '{prompt}' with Trace ID: {query_message.trace_id}")

    # Add a placeholder for the bot's response while processing
    with st.chat_message("assistant"):
        # Use st.empty() to create a placeholder that can be updated
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

        response_received = False
        bot_response_content = "Still thinking..."
        source_context_info = []
        current_query_trace_id = query_message.trace_id

        # Loop to process messages from the queue until a final response for this trace_id is found
        while not response_received:
            # Create a new list for messages that are not processed in this iteration
            messages_to_keep = []
            
            # Process messages from the queue
            for msg in st.session_state.ui_message_queue:
                if msg.get("trace_id") == current_query_trace_id:
                    if msg["type"] == "FINAL_RESPONSE":
                        bot_response_content = msg["answer"]
                        source_context_info = []
                        for i, chunk_content in enumerate(msg["source_chunks"]):
                            metadata = msg["source_metadata"][i] if i < len(msg["source_metadata"]) else {}
                            source_context_info.append({
                                "file_name": metadata.get("file_name", "N/A"),
                                "content": chunk_content
                            })
                        response_received = True
                        print(f"UI: FINAL_RESPONSE received for Trace ID: {current_query_trace_id}")
                        break # Exit inner loop, found response
                    elif msg["type"] == "ERROR_MESSAGE":
                        bot_response_content = f"Error: {msg['error']}\nContext: {msg['context']}"
                        response_received = True
                        print(f"UI: ERROR_MESSAGE received for Trace ID: {current_query_trace_id}")
                        break # Exit inner loop, found error
                    elif msg["type"] == "STATUS_UPDATE":
                        # Update placeholder with status, but don't stop waiting
                        response_placeholder.markdown(f"*{msg['status']}*")
                        print(f"UI: STATUS_UPDATE received for Trace ID: {current_query_trace_id}: {msg['status']}")
                        messages_to_keep.append(msg) # Keep status updates until final response
                else:
                    messages_to_keep.append(msg) # Keep messages not for this trace

            # Update the queue, removing processed messages for the current trace
            st.session_state.ui_message_queue = messages_to_keep
            
            if not response_received:
                # If no response yet, force a rerun to check the queue again
                # This is crucial for Streamlit to re-execute and pick up new messages
                print(f"UI: No final response for Trace ID {current_query_trace_id}. Rerunning...")
                import time
                time.sleep(0.1) # Small delay to prevent excessive reruns
                st.rerun()

        # Display bot response after it's received and loop has broken
        response_placeholder.markdown(bot_response_content) # Update the placeholder with the final answer
        if source_context_info:
            with st.expander("Source Context"):
                for idx, source in enumerate(source_context_info):
                    st.markdown(f"**Chunk {idx + 1} from {source.get('file_name', 'Unknown File')}:**")
                    st.code(source.get('content', 'No content available'))
                    st.markdown("---")

    # Add bot response to chat history (this happens on the next rerun after the loop breaks)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response_content,
        "source_info": source_context_info
    })

# --- Cleanup temporary files on exit (optional, for persistent storage) ---
temp_dir = "./temp_uploaded_files"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

