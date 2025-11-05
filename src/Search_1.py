# Import necessary libraries and modules
import os  # For environment variables and file operations
from logger import logging  # For logging and debugging
from dotenv import load_dotenv  # For loading environment variables from .env file
from src.VectorStore_1 import FaissVectorStore  # Our custom vector store
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM
from langchain_core.prompts import ChatPromptTemplate  # For creating structured prompts
from typing import List,Dict,Any

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

class RAGSearch:
    def __init__(
        self, 
        persist_dir: str = "faiss_store",  # Directory for vector store persistence
        embedding_model: str = "all-MiniLM-L6-v2",  # Embedding model name
        llm_model: str = "gemini-2.5-flash",  # LLM model name
        chunk_size: int = 1200,  # Document chunk size
        chunk_overlap: int = 250  # Document chunk overlap
    ):
        """
        Initialize the RAG search system with vector store and LLM
        """
        # Store the persistence directory path
        self.persist_dir = persist_dir
        
        # Initialize the vector store with configuration
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize or load the vector store - critical step!
        if not self._initialize_vectorstore():
            logging.error("Failed to initialize vector store")
            # Raise exception to prevent using broken RAG system
            raise RuntimeError("Vector store initialization failed")
        
        # Initialize the Large Language Model (Google Gemini)
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model,
                temperature=0.5,  # Controls randomness: 0 = deterministic, 1 = creative
                max_output_tokens=2000  # Maximum length of generated response
            )
            logging.info(f"Initialized Gemini model: {llm_model}")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {str(e)}")
            raise  # Stop initialization if LLM fails
    
    def _initialize_vectorstore(self) -> bool:
        """
        Private method to initialize vector store
        - Loads existing store if available
        - Builds new store from documents if not exists
        Returns: True if successful, False if failed
        """
        # Check if vector store files already exist
        if self.vectorstore.exists():
            logging.info("Loading existing vector store")
            # Try to load the existing vector store
            return self.vectorstore.load()
        else:
            logging.info("Building new vector store from documents")
            # Import here to avoid circular imports
            from src.Data_loader_1 import load_all_documents
            
            # Load all documents from the data directory
            docs = load_all_documents("data")
            if not docs:
                logging.error("No documents found in data directory")
                return False
            
            # Build vector store from loaded documents
            return self.vectorstore.build_from_documents(docs)
    
    def search_and_summarize(
        self, 
        query: str, 
        top_k: int = 5,  # Number of similar documents to retrieve
        include_sources: bool = True  # Whether to include source citations
    ) -> str:
        """
        Main RAG method: Search for relevant documents and generate a summary
        
        Args:
            query: User's search question
            top_k: Number of top similar documents to retrieve
            include_sources: Whether to include source information in response
        
        Returns:
            Generated summary with optional sources
        """
        logging.info(f"Processing query: '{query}'")
        
        # Step 1: Search for relevant documents in vector store
        results = self.vectorstore.query(query, top_k=top_k)
        
        # Check if we found any relevant documents
        if not results:
            return "No relevant documents found for your query."
        
        # Step 2: Extract context and sources from search results
        texts = []   # Will store the actual text content
        sources = [] # Will store source information for attribution
        
        # Process each search result
        for i, result in enumerate(results):
            # Get the text content from metadata
            text = result["metadata"].get("text", "")
            # Get the source file information
            source = result["metadata"].get("source", "Unknown")
            # Get the similarity score (0-1, higher is better)
            similarity = result["similarity"]
            
            # Format the text with numbering and similarity score
            texts.append(f"Document {i+1} (Similarity: {similarity:.3f}):\n{text}")
            # Format source information for citation
            sources.append(f"- {source} (Similarity: {similarity:.3f})")
        
        # Combine all text chunks into a single context string
        context = "\n\n".join(texts)
        
        # Step 3: Create an enhanced prompt for the LLM
        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Based on the following context documents, provide a comprehensive and accurate summary that directly addresses the user's query.

User Query: {query}

Context Documents:
{context}

Please provide a well-structured summary in markdown format that:
1. Directly answers the query
2. Cites relevant information from the context
3. Is organized and easy to read
4. Acknowledges limitations if information is incomplete

Summary:
""")
        
        try:
            # Format the prompt with actual query and context
            prompt = prompt_template.format(query=query, context=context)
            
            # Step 4: Send prompt to LLM and get response
            response = self.llm.invoke(prompt)
            
            # Step 5: Add source citations if requested
            if include_sources and sources:
                source_info = "\n\n## Sources\n" + "\n".join(sources)
                return response.content + source_info
            
            # Return just the LLM response if sources not requested
            return response.content
            
        except Exception as e:
            # Handle any errors during LLM processing
            logging.error(f"LLM invocation failed: {str(e)}")
            return "I encountered an error while generating the response. Please try again."
    
    def simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple search returning raw results without LLM processing
        Useful for debugging or when you want raw search results
        """
        return self.vectorstore.query(query, top_k=top_k)
    
    def get_vectorstore_stats(self) -> Dict:
        """
        Get statistics about the vector store
        Useful for monitoring and debugging
        """
        return self.vectorstore.get_stats()

# Example usage and testing
if __name__ == "__main__":
    # Initialize the complete RAG system
    rag = RAGSearch()
    
    # Example query - the system will:
    # 1. Search vector store for similar documents
    # 2. Combine them into context
    # 3. Generate a summary using Gemini
    result = rag.search_and_summarize("What is machine learning?")
    print(result)
    
    # Get statistics about the vector store
    stats = rag.get_vectorstore_stats()
    print(f"Vector store stats: {stats}")