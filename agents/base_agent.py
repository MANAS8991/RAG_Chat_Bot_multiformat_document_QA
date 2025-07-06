# rag_chatbot/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any

# Import the MCPMessage and message_bus from utils
from utils.mcp import MCPMessage, message_bus

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the RAG chatbot system.
    Defines the common interface for agents to handle messages.
    """
    def __init__(self, name: str):
        """
        Initializes the BaseAgent with a unique name.

        Args:
            name (str): The name of the agent. This name is used as the 'receiver'
                        in MCP messages.
        """
        self.name = name
        # Register this agent's message handling method with the global message bus.
        # This allows other agents to send messages to this agent by its name.
        message_bus.register_handler(self.name, self.handle_message)
        print(f"BaseAgent: {self.name} initialized and registered with MessageBus.")

    @abstractmethod
    def handle_message(self, message: MCPMessage):
        """
        Abstract method to be implemented by concrete agent classes.
        This method defines how an agent processes an incoming MCPMessage.

        Args:
            message (MCPMessage): The incoming message to be processed.
        """
        pass # Concrete implementations will provide the logic here

