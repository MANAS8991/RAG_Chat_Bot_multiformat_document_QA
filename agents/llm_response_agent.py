# rag_chatbot/agents/llm_response_agent.py

import requests
import json
from typing import Dict, Any, List

# Import BaseAgent and MCP components
from agents.base_agent import BaseAgent
from utils.mcp import MCPMessage, message_bus

# Import configuration settings
from config import LLM_MODEL_NAME, GEMINI_API_KEY, GEMINI_API_URL, TOP_K_RETRIEVED_CHUNKS

class LLMResponseAgent(BaseAgent):
    """
    The LLMResponseAgent is responsible for:
    1. Receiving retrieved context and the user query from the RetrievalAgent.
    2. Constructing a prompt for the Large Language Model (LLM).
    3. Calling the LLM (Gemini API) to generate an answer.
    4. Sending the final answer and source context back to the CoordinatorAgent.
    """
    def __init__(self):
        # Initialize the base agent with its specific name
        super().__init__("LLMResponseAgent")
        # No direct LLM object initialization here, as we'll make direct API calls
        print("LLMResponseAgent: Initialized. Ready to call Gemini API.")

    def handle_message(self, message: MCPMessage):
        """
        Handles incoming MCP messages relevant to LLM response generation.

        Supported message types:
        - "RETRIEVAL_RESULT": Triggered by RetrievalAgent after finding relevant chunks.
                              Payload contains 'query', 'retrieved_context', and 'source_metadata'.
        """
        print(f"LLMResponseAgent: Received message of type '{message.type}' from {message.sender} (Trace ID: {message.trace_id})")

        if message.type == "RETRIEVAL_RESULT":
            self._generate_response(message)
        else:
            print(f"LLMResponseAgent: Unrecognized message type: {message.type}")


    def _generate_response(self, message: MCPMessage):
        """
        Generates a response using the LLM (Gemini API) based on the query and retrieved context.

        Args:
            message (MCPMessage): The RETRIEVAL_RESULT message.
        """
        query = message.payload.get("query")
        retrieved_context = message.payload.get("retrieved_context", [])
        source_metadata = message.payload.get("source_metadata", [])

        if not query:
            print("LLMResponseAgent Error: 'query' missing in RETRIEVAL_RESULT payload.")
            return

        # Construct the prompt for the LLM
        context_str = "\n".join(retrieved_context)
        
        # Define the system instruction for the LLM
        system_instruction = """You are an AI assistant designed to answer questions based on the provided context.
        If the answer is not found in the context, state that you don't have enough information.
        Do not make up answers."""

        # Prepare the chat history for the Gemini API call
        # The prompt should combine system instruction, context, and query
        full_prompt = f"{system_instruction}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": full_prompt}]})

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 50,
                "maxOutputTokens": 500,
            }
        }

        # Construct the API URL with the API key (which will be provided by Canvas if empty)
        api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

        print(f"LLMResponseAgent: Calling Gemini API at {api_url_with_key} for query: '{query}'")
        try:
            # Make the POST request to the Gemini API with a timeout
            response = requests.post(
                api_url_with_key,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30 # Add a 30-second timeout for the API call
            )
            
            print(f"LLMResponseAgent: Received response status code: {response.status_code}")
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            print(f"LLMResponseAgent: Received Gemini API response: {json.dumps(result, indent=2)}") # Log full response

            generated_answer = "No answer generated."

            # Parse the Gemini API response
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                generated_answer = result["candidates"][0]["content"]["parts"][0].get("text", "No text part found in response.")
            elif result.get("error"): # Check for explicit error object in response
                error_message = result["error"].get("message", "Unknown error from Gemini API.")
                generated_answer = f"Error from LLM API: {error_message}"
                print(f"LLMResponseAgent Error: Gemini API returned an error: {error_message}")
            else:
                print(f"LLMResponseAgent Warning: Unexpected Gemini API response structure: {result}")
                generated_answer = f"Error: Unexpected response from LLM. Details: {json.dumps(result)}"

            print(f"LLMResponseAgent: Generated answer (first 100 chars): {generated_answer[:100]}...")

            # Send the final answer and source context back to the CoordinatorAgent
            final_response_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent", # Target the CoordinatorAgent
                type="FINAL_RESPONSE",
                trace_id=message.trace_id, # Maintain the same trace ID
                payload={
                    "answer": generated_answer.strip(), # Clean up whitespace
                    "source_chunks": retrieved_context, # The actual text chunks used
                    "source_metadata": source_metadata, # Metadata of the chunks
                    "original_query": query
                }
            )
            message_bus.send_message(final_response_message)

        except requests.exceptions.Timeout:
            print(f"LLMResponseAgent Error: Gemini API request timed out after 30 seconds for query: '{query}'")
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": "LLM API request timed out. Please try again.",
                    "context": f"Query: '{query}'"
                }
            )
            message_bus.send_message(error_message)
        except requests.exceptions.RequestException as e:
            print(f"LLMResponseAgent Error: Network or API request failed: {e}")
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": f"LLM API request failed: {e}",
                    "context": f"Query: '{query}'"
                }
            )
            message_bus.send_message(error_message)
        except json.JSONDecodeError as e:
            print(f"LLMResponseAgent Error: Failed to parse JSON response from LLM: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}")
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": f"LLM response parsing failed: {e}",
                    "context": f"Query: '{query}'"
                }
            )
            message_bus.send_message(error_message)
        except Exception as e:
            print(f"LLMResponseAgent Unexpected Error: {e}")
            error_message = MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                type="ERROR_MESSAGE",
                trace_id=message.trace_id,
                payload={
                    "error": f"An unexpected error occurred during LLM response generation: {e}",
                    "context": f"Query: '{query}'"
                }
            )
            message_bus.send_message(error_message)

    def _initialize_llm(self):
        print("LLMResponseAgent: _initialize_llm called, but direct API calls are used for Gemini.")
        return None
