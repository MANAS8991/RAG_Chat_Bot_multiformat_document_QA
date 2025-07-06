# Agentic RAG Chatbot for Multi-Format Document QA
This project implements an agent-based Retrieval-Augmented Generation (RAG) chatbot capable of answering user questions using uploaded documents of various formats. It features an agentic architecture and utilizes the Model Context Protocol (MCP) for seamless communication between agents and the LLM.

# 🚀 Core Features
Diverse Document Format Support:

✅ PDF (.pdf)

✅ PPTX (.pptx)

✅ CSV (.csv)

✅ DOCX (.docx)

✅ TXT / Markdown (.txt, .md)

Agentic Architecture:

IngestionAgent: Parses and preprocesses documents.

RetrievalAgent: Handles embedding generation, vector storage, and semantic retrieval.

LLMResponseAgent: Forms the final LLM query with retrieved context and generates answers.

CoordinatorAgent: Orchestrates the workflow and manages inter-agent communication.

Model Context Protocol (MCP): Structured in-memory messaging for agent-to-agent and agent-to-LLM communication, ensuring clear context passing.

Vector Store + Embeddings:

Embeddings: Uses Sentence Transformers (all-MiniLM-L6-v2) for efficient text embedding.

Vector DB: Leverages FAISS for fast similarity search and retrieval.

Chatbot Interface (UI): Built with Streamlit, allowing users to:

Upload multiple documents.

Ask multi-turn questions.

View responses with clear source context (retrieved chunks).

🧠 Agent-based Architecture with MCP Integration
The system is designed with a modular, agent-based approach, where each agent specializes in a specific task. The Model Context Protocol (MCP) serves as the standardized communication layer, ensuring structured and traceable message passing.

Agents Overview:
CoordinatorAgent: The central orchestrator. It receives requests from the Streamlit UI (document uploads, user queries) and routes them to the appropriate specialized agents. It also collects final responses and errors to send back to the UI.

IngestionAgent: Responsible for handling raw document files. It parses various formats (PDF, DOCX, PPTX, CSV, TXT/MD) into plain text and prepares them for the next stage.

RetrievalAgent: Manages the vector database. It takes processed text, chunks it, generates embeddings, and stores them in FAISS. When a query comes in, it performs a semantic search to retrieve the most relevant text chunks.

LLMResponseAgent: Interacts with the Large Language Model. It receives the user's query and the retrieved context, constructs a well-formed prompt, calls the LLM, and generates the final human-readable answer.

System Flow Diagram (with Message Passing)
User Interaction (UI)

Document Upload:

User uploads sales_review.pdf, metrics.csv.

➡️ UI sends UI_UPLOAD_REQUEST (MCP) to CoordinatorAgent.

User Query:

User asks: "What KPIs were tracked in Q1?"

➡️ UI sends UI_QUERY_REQUEST (MCP) to CoordinatorAgent.

CoordinatorAgent Orchestration

Receives UI_UPLOAD_REQUEST.

➡️ Sends UPLOAD_DOCUMENT (MCP) to IngestionAgent (payload: file_path, file_name, file_type).

Receives UI_QUERY_REQUEST.

➡️ Sends QUERY_REQUEST (MCP) to RetrievalAgent (payload: query).

IngestionAgent Processing

Receives UPLOAD_DOCUMENT.

Parses the document (e.g., sales_review.pdf) into raw text.

➡️ Sends INGESTION_COMPLETE (MCP) to RetrievalAgent (payload: raw_text, source_metadata).

RetrievalAgent Operations

Receives INGESTION_COMPLETE.

Chunks the raw_text.

Generates embeddings for chunks.

Adds embeddings and chunks to the FAISS vector store.

Receives QUERY_REQUEST.

Generates embedding for the query.

Performs similarity search in FAISS.

Retrieves top_chunks (e.g., "slide 3: revenue up", "doc: Q1 summary...").

➡️ Sends RETRIEVAL_RESULT (MCP) to LLMResponseAgent (payload: query, retrieved_context, source_metadata).

LLMResponseAgent Generation

Receives RETRIEVAL_RESULT.

Constructs a prompt using query and retrieved_context.

Calls the underlying LLM (Google Gemini 2.0 Flash via API).

Generates the answer.

➡️ Sends FINAL_RESPONSE (MCP) to CoordinatorAgent (payload: answer, source_chunks, source_metadata, original_query).

CoordinatorAgent Finalization

Receives FINAL_RESPONSE.

Updates the conversation state.

➡️ Sends FINAL_RESPONSE (UI Callback) to the Streamlit UI (payload: answer, source_chunks, source_metadata, original_query).

User Interface (UI)

Displays the answer to the user.

Presents the source_chunks in an expandable section for transparency.

## 🛠️ Tech Stack Used
Python: The core programming language.

Streamlit: For building the interactive web user interface.

LangChain: A framework for developing applications powered by language models. Used for:

RecursiveCharacterTextSplitter: For document chunking.

SentenceTransformerEmbeddings: For generating text embeddings.

FAISS: As the vector store for efficient similarity search.

PromptTemplate: For interacting with LLMs.

requests: For making HTTP API calls to the Google Gemini API.

sentence-transformers: Provides pre-trained models for creating embeddings.

transformers: Required for SentenceTransformerEmbeddings and other potential future integrations.

## Document Parsers:

pypdf: For PDF document parsing.

python-docx: For DOCX document parsing.

python-pptx: For PPTX document parsing.

pandas: For CSV document parsing.

tiktoken: Used by LangChain for token counting.

## ⚙️ Setup and Installation
Clone the Repository:

git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot

Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Configure API Key (Google Gemini):
To use the Google Gemini API, you need to provide your API key.

Go to Google AI Studio to generate your API key.

Directly in config.py: Open rag_chatbot/config.py and replace "YOUR_ACTUAL_GEMINI_API_KEY_HERE" with your copied Gemini API key.

# rag_chatbot/config.py
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"

Note: If running in a Google Canvas/Immersive Document environment, the GEMINI_API_KEY can be left as an empty string ("") as the key is automatically provided at runtime. For local execution, it must be set.

## ▶️ How to Run
Ensure your virtual environment is activated.

Navigate to the rag_chatbot directory.

Run the Streamlit application:

streamlit run main.py

Your web browser will automatically open the chatbot interface (usually at http://localhost:8501).

## 📝 Usage Instructions
Upload Documents:

On the left sidebar, click "Browse files" under "Upload Documents".

Select one or more PDF, PPTX, CSV, DOCX, TXT, or Markdown files from your computer.

The app will automatically start processing the uploaded documents. You'll see "Processing [filename]..." toasts.

Ask Questions:

Once documents are processed (this might take a moment depending on file size and number), type your question in the chat input box at the bottom of the screen.

Press Enter or click the send button.

View Responses:

The chatbot will display its answer.

An expandable "Source Context" section will appear below the answer, showing the exact text chunks from your uploaded documents that were used to formulate the response.



## 🚧 Challenges Faced
Synchronous Streamlit with Asynchronous Agent Flow: Streamlit's execution model is largely synchronous. Simulating an asynchronous agent communication flow (where agents send messages and the UI waits for a response) required careful management of Streamlit's st.session_state and a polling mechanism for the ui_message_queue to ensure the UI updates correctly without infinite loops.

Temporary File Management: Uploaded files in Streamlit are BytesIO objects. Agents require file paths to read. This necessitated saving uploaded files to a temporary directory (./temp_uploaded_files) for agents to access, and considering how to manage/clean these files (currently, they persist until manual deletion or app restart).

LLM Integration and API Key Management: Transitioning from local/OpenAI models to the Google Gemini API required adapting the LLM interaction logic and ensuring proper API key configuration for both local and potential Canvas environments.

Error Handling and User Feedback: Implementing robust error handling across agents and ensuring these errors are clearly communicated back to the user via the Streamlit UI was crucial for a good user experience.

## 🔮 Future Scope / Improvements
Persistent Storage for Vector Store: Currently, the FAISS index is saved locally (faiss_index.bin). For a more robust application, integrate with cloud-based vector databases (e.g., Pinecone, Weaviate, ChromaDB) or a more sophisticated local persistence mechanism.

Asynchronous Message Bus: Replace the simple in-memory MessageBus with a true asynchronous message queue (e.g., Celery with Redis/RabbitMQ, or a pub/sub service) to enable non-blocking agent execution and improve scalability.

Advanced Document Preprocessing: Implement more sophisticated text cleaning, OCR for image-based PDFs, and handling of tables/figures.

Multi-Modal RAG: Extend the system to handle image or audio inputs and retrieve information from multi-modal documents.

User Authentication: For multi-user scenarios, implement user authentication to manage personal document collections.

Monitoring and Logging: Add more detailed logging and monitoring capabilities to track agent performance and message flows.

Generative UI: Explore using LLMs to dynamically generate UI elements or refine the user's query based on context.

Conversation History Persistence: Save chat history to a database so it persists across sessions.


## Architecture

rag_chatbot/
├── agents/                 # Specialized AI agents
│   ├── base_agent.py
│   ├── coordinator_agent.py
│   ├── ingestion_agent.py
│   ├── llm_response_agent.py
│   └── retrieval_agent.py
├── utils/                  # Helper functions & MCP
│   ├── document_parser.py
│   ├── mcp.py
│   └── vector_store_manager.py
├── config.py               # Configuration settings (API keys, model names)
├── main.py                 # Streamlit UI & application entry point
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies