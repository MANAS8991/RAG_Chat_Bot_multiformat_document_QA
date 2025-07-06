# rag_chatbot/utils/vector_store_manager.py

import os
from typing import List, Dict, Any, Optional

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# Import configuration settings
from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH

class VectorStoreManager:
    """
    Manages the FAISS vector store, including embedding generation,
    document chunking, adding documents, and retrieval.
    """
    def __init__(self):
        # Initialize the embedding model.
        # SentenceTransformerEmbeddings uses the 'sentence-transformers' library.
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Initialize the text splitter for breaking down large documents.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, # Adds metadata about the chunk's original position
        )
        
        # Initialize FAISS vector store as None; it will be loaded or created later.
        self.vector_store: Optional[FAISS] = None
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """
        Loads an existing FAISS index from disk if it exists,
        otherwise initializes an empty one.
        """
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"VectorStoreManager: Loading FAISS index from {FAISS_INDEX_PATH}")
            try:
                # Load the FAISS index with the initialized embeddings
                self.vector_store = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("VectorStoreManager: FAISS index loaded successfully.")
            except Exception as e:
                print(f"VectorStoreManager Error: Could not load FAISS index: {e}. Creating a new one.")
                # If loading fails, create a new empty one.
                self.vector_store = FAISS.from_texts([""], self.embeddings) # Initialize with a dummy text
                self._save_vector_store() # Save the newly created empty store
        else:
            print("VectorStoreManager: No existing FAISS index found. Creating a new one.")
            # Initialize an empty FAISS store.
            # FAISS.from_texts requires at least one text to initialize.
            # We'll add a dummy text and then delete it if necessary, or just overwrite.
            self.vector_store = FAISS.from_texts([""], self.embeddings)
            self._save_vector_store() # Save the newly created empty store

    def _save_vector_store(self):
        """
        Saves the current state of the FAISS index to disk.
        """
        if self.vector_store:
            try:
                self.vector_store.save_local(FAISS_INDEX_PATH)
                print(f"VectorStoreManager: FAISS index saved to {FAISS_INDEX_PATH}")
            except Exception as e:
                print(f"VectorStoreManager Error: Could not save FAISS index: {e}")
        else:
            print("VectorStoreManager: No vector store to save.")

    def add_documents_to_index(self, raw_text: str, source_metadata: Dict[str, Any]) -> List[LangchainDocument]:
        """
        Splits raw text into chunks, generates embeddings, and adds them to the FAISS index.

        Args:
            raw_text (str): The full text content of a document.
            source_metadata (Dict[str, Any]): Metadata associated with the document
                                               (e.g., file_name, file_type).

        Returns:
            List[LangchainDocument]: A list of LangChain Document objects that were added.
        """
        print(f"VectorStoreManager: Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        # Create LangChain Document objects from the raw text
        # The metadata will be attached to each chunk
        documents = self.text_splitter.create_documents(
            texts=[raw_text],
            metadatas=[source_metadata]
        )
        print(f"VectorStoreManager: Created {len(documents)} chunks.")

        if not documents:
            print("VectorStoreManager: No documents to add after splitting.")
            return []

        # Add the documents (chunks with embeddings) to the vector store
        try:
            # If the vector store was initialized with a dummy text, we need to handle it.
            # A simpler way is to just add the new documents. FAISS.add_documents
            # will append to the existing index.
            self.vector_store.add_documents(documents)
            self._save_vector_store() # Save after adding documents
            print(f"VectorStoreManager: Added {len(documents)} chunks to FAISS index.")
            return documents
        except Exception as e:
            print(f"VectorStoreManager Error: Failed to add documents to FAISS index: {e}")
            return []

    def retrieve_relevant_chunks(self, query_text: str, k: int) -> List[LangchainDocument]:
        """
        Performs a similarity search in the FAISS index to find the most relevant chunks.

        Args:
            query_text (str): The user's query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[LangchainDocument]: A list of LangChain Document objects (chunks)
                                     most relevant to the query.
        """
        if not self.vector_store:
            print("VectorStoreManager: Vector store not initialized. Cannot retrieve chunks.")
            return []

        print(f"VectorStoreManager: Retrieving top {k} relevant chunks for query: '{query_text}'")
        try:
            # Perform similarity search
            # The 'search_type="similarity"' is default, but explicitly stated for clarity.
            # 'k' specifies the number of results to return.
            retrieved_docs = self.vector_store.similarity_search(query_text, k=k)
            print(f"VectorStoreManager: Retrieved {len(retrieved_docs)} chunks.")
            return retrieved_docs
        except Exception as e:
            print(f"VectorStoreManager Error: Failed to retrieve chunks: {e}")
            return []

