# rag_chatbot/utils/mcp.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
import uuid

# Define the Model Context Protocol (MCP) message structure using a dataclass.
# This ensures consistency and type safety for messages exchanged between agents.
@dataclass
class MCPMessage:
    """
    Represents a message conforming to the Model Context Protocol (MCP).

    Attributes:
        sender (str): The name of the agent sending the message.
        receiver (str): The name of the agent intended to receive the message.
        type (str): The type of message (e.g., "UPLOAD_DOCUMENT", "QUERY_REQUEST",
                    "INGESTION_COMPLETE", "RETRIEVAL_RESULT", "FINAL_RESPONSE").
        trace_id (str): A unique identifier to trace a conversation or task flow.
                        Generated automatically if not provided.
        payload (Dict[str, Any]): The actual data being transmitted.
                                  Contents vary based on message type.
    """
    sender: str
    receiver: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# A simple in-memory message bus to simulate communication between agents.
# In a more complex system, this could be a real message queue (e.g., RabbitMQ, Kafka)
# or a REST API endpoint. For this project, in-memory is sufficient and simpler.
class MessageBus:
    """
    A simple in-memory message bus for inter-agent communication.
    Agents can register handlers for specific message types and send messages.
    """
    def __init__(self):
        # Dictionary to store registered handlers.
        # Key: receiver agent name (str), Value: list of handler functions (Callable)
        self._handlers: Dict[str, List[Callable[[MCPMessage], None]]] = {}
        # List to store messages temporarily if no handler is immediately available
        self._message_queue: List[MCPMessage] = []

    def register_handler(self, receiver_name: str, handler_func: Callable[[MCPMessage], None]):
        """
        Registers a handler function for a specific receiver agent.
        When a message is sent to 'receiver_name', the 'handler_func' will be called.

        Args:
            receiver_name (str): The name of the agent that will receive messages.
            handler_func (Callable): The function to call when a message for
                                     this receiver is received. It must accept
                                     one argument: an MCPMessage object.
        """
        if receiver_name not in self._handlers:
            self._handlers[receiver_name] = []
        self._handlers[receiver_name].append(handler_func)
        print(f"MessageBus: Handler registered for {receiver_name}")

    def send_message(self, message: MCPMessage):
        """
        Sends an MCPMessage to its intended receiver.
        If a handler is registered for the receiver, the message is processed immediately.
        Otherwise, it's added to a queue (though for this synchronous model,
        handlers should typically be registered before messages are sent).

        Args:
            message (MCPMessage): The message to send.
        """
        print(f"MessageBus: Sending message from {message.sender} to {message.receiver} "
              f"of type '{message.type}' (Trace ID: {message.trace_id})")
        
        # Find handlers for the intended receiver
        handlers = self._handlers.get(message.receiver)

        if handlers:
            for handler in handlers:
                try:
                    # Call the handler function with the message
                    handler(message)
                except Exception as e:
                    print(f"MessageBus Error: Handler for {message.receiver} failed "
                          f"to process message type '{message.type}': {e}")
        else:
            # If no handler is registered, queue the message.
            # In this project's synchronous flow, this might indicate an issue
            # or a message intended for a future state.
            self._message_queue.append(message)
            print(f"MessageBus Warning: No handler registered for {message.receiver}. "
                  f"Message queued. Current queue size: {len(self._message_queue)}")

    def process_queued_messages(self):
        """
        Attempts to process any messages that were queued because no handler was available.
        This can be called after all agents have registered their handlers.
        """
        if not self._message_queue:
            return

        print(f"MessageBus: Attempting to process {len(self._message_queue)} queued messages.")
        new_queue = []
        for message in self._message_queue:
            handlers = self._handlers.get(message.receiver)
            if handlers:
                for handler in handlers:
                    try:
                        handler(message)
                    except Exception as e:
                        print(f"MessageBus Error: Queued handler for {message.receiver} failed: {e}")
            else:
                new_queue.append(message) # Still no handler, keep in queue

        self._message_queue = new_queue
        if self._message_queue:
            print(f"MessageBus: {len(self._message_queue)} messages remain in queue after processing.")
        else:
            print("MessageBus: All queued messages processed.")

# Global instance of the MessageBus for easy access across agents.
# In a production system, this might be passed as a dependency.
message_bus = MessageBus()
