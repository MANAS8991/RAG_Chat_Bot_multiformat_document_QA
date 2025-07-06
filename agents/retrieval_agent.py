# rag_chatbot/agents/retrieval_agent.py

from typing import Dict, Any, List

# Import BaseAgent and MCP components
from agents.base_agent import BaseAgent
from utils.mcp import MCPMessage, message_bus

# Import the VectorStoreManager
from utils.vector_store_manager import VectorStoreManager
from config import TOP_K_RETRIEVED_CHUNKS # Import the configuration for top_k

class RetrievalAgent(BaseAgent):
    """
    The RetrievalAgent is responsible for managing the vector store (FAISS).
    It handles:
    1. Receiving processed text from IngestionAgent and adding it to the vector store.
    2. Receiving user queries from CoordinatorAgent and retrieving relevant chunks.
    """
    def __init__(self):
        # Initialize the base agent with its specific name
        super().__init__("RetrievalAgent")
        # Initialize the VectorStoreManager, which handles embeddings and FAISS operations.
        self.vector_store_manager = VectorStoreManager()
        print("RetrievalAgent: Initialized with VectorStoreManager.")

    def handle_message(self, message: MCPMessage):
        """
        Handles incoming MCP messages relevant to document retrieval and indexing.

        Supported message types:
        - "INGESTION_COMPLETE": Triggered by IngestionAgent after parsing a document.
                                Payload contains 'raw_text' and 'source_metadata'.
        - "QUERY_REQUEST": Triggered by CoordinatorAgent when a user asks a question.
                           Payload contains 'query'.
        """
        print(f"RetrievalAgent: Received message of type '{message.type}' from {message.sender} (Trace ID: {message.trace_id})")

        if message.type == "INGESTION_COMPLETE":
            self._add_documents_to_store(message)
        elif message.type == "QUERY_REQUEST":
            self._retrieve_chunks_for_query(message)
        else:
            print(f"RetrievalAgent: Unrecognized message type: {message.type}")

    def _add_documents_to_store(self, message: MCPMessage):
        """
        Adds parsed document text (chunks) to the vector store.

        Args:
            message (MCPMessage): The INGESTION_COMPLETE message.
        """
        raw_text = message.payload.get("raw_text")
        source_metadata = message.payload.get("source_metadata", {})

        if not raw_text:
            print("RetrievalAgent Error: 'raw_text' missing in INGESTION_COMPLETE payload.")
            return

        print(f"RetrievalAgent: Adding document '{source_metadata.get('file_name', 'unknown')}' to vector store.")
        try:
            # Use the VectorStoreManager to split text, generate embeddings, and add to FAISS.
            added_documents = self.vector_store_manager.add_documents_to_index(raw_text, source_metadata)
            
            if added_documents:
                print(f"RetrievalAgent: Successfully added {len(added_documents)} chunks to vector store.")
                # Optionally, send a confirmation message back to Coordinator or log.
            else:
                print("RetrievalAgent: No documents were added to the vector store.")

        except Exception as e:
            print(f"RetrievalAgent Error: Failed to add documents to vector store: {e}")
            # Send an error message back to the CoordinatorAgent
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": str(e),
                    "context": f"Failed to add document '{source_metadata.get('file_name', 'unknown')}' to vector store."
                }
            )
            message_bus.send_message(error_message)

    def _retrieve_chunks_for_query(self, message: MCPMessage):
        """
        Retrieves relevant text chunks from the vector store based on a user query.

        Args:
            message (MCPMessage): The QUERY_REQUEST message.
        """
        query = message.payload.get("query")

        if not query:
            print("RetrievalAgent Error: 'query' missing in QUERY_REQUEST payload.")
            return

        print(f"RetrievalAgent: Retrieving chunks for query: '{query}'")
        try:
            # Use the VectorStoreManager to perform the similarity search.
            # TOP_K_RETRIEVED_CHUNKS is defined in config.py
            retrieved_chunks = self.vector_store_manager.retrieve_relevant_chunks(query, k=TOP_K_RETRIEVED_CHUNKS)
            
            # Extract the text content and metadata from the retrieved LangChain Document objects
            context_texts = [doc.page_content for doc in retrieved_chunks]
            source_info = [doc.metadata for doc in retrieved_chunks] # Keep metadata for source context

            print(f"RetrievalAgent: Retrieved {len(context_texts)} chunks for query.")

            # Send the retrieved context and the original query to the LLMResponseAgent.
            response_message = MCPMessage(
                sender=self.name,
                receiver="LLMResponseAgent", # Target the LLMResponseAgent
                type="RETRIEVAL_RESULT",
                trace_id=message.trace_id, # Maintain the same trace ID
                payload={
                    "query": query,
                    "retrieved_context": context_texts, # List of strings
                    "source_metadata": source_info # List of dicts
                }
            )
            message_bus.send_message(response_message)

        except Exception as e:
            print(f"RetrievalAgent Error: Failed to retrieve chunks for query '{query}': {e}")
            # Send an error message back to the CoordinatorAgent
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": str(e),
                    "context": f"Failed to retrieve information for query: '{query}'"
                }
            )
            message_bus.send_message(error_message)

