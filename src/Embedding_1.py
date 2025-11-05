# Import necessary libraries and modules
from typing import List, Any  # For type hints
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from sentence_transformers import SentenceTransformer  # For generating embeddings
import numpy as np  # For numerical operations and array handling
from logger import logging  # For logging and debugging



class EmbeddingPipeline:
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",  # Default embedding model
        chunk_size: int = 1200,  # Maximum characters per chunk
        chunk_overlap: int = 250  # Characters overlapping between chunks
    ):
        """
        Initialize the embedding pipeline with model and chunking parameters
        """
        # Store chunking parameters as instance variables
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            # Load the pre-trained sentence transformer model
            self.model = SentenceTransformer(model_name)
            # Log successful model loading
            logging.info(f"Loaded embedding model: {model_name}")
            # Log the model's maximum sequence length (important for chunk sizing)
            logging.info(f"Model max sequence length: {self.model.max_seq_length}")
        except Exception as e:
            # Log any errors during model loading and re-raise the exception
            logging.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise  # This stops the pipeline if model can't be loaded
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for better embedding and retrieval
        """
        # Check if documents list is empty
        if not documents:
            logging.warning("No documents to chunk")
            return []  # Return empty list if no documents provided
        
        # Create a text splitter instance with configured parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,        # Maximum size of each chunk
            chunk_overlap=self.chunk_overlap,  # Overlap between consecutive chunks
            separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchy of separators to split on
            length_function=len  # Function to calculate text length (using Python's len)
        )
        
        try:
            # Split the documents into chunks using the text splitter
            chunks = splitter.split_documents(documents)
            # Log the chunking results
            logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            # Log any errors during chunking
            logging.error(f"Document chunking failed: {str(e)}")
            return []  # Return empty list on failure
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate vector embeddings for text chunks
        """
        # Check if chunks list is empty
        if not chunks:
            logging.warning("No chunks to embed")
            return np.array([])  # Return empty numpy array
        
        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            # Check if chunk has 'page_content' attribute (LangChain Document)
            if hasattr(chunk, 'page_content'):
                texts.append(chunk.page_content)
            else:
                # If not a standard document, convert to string
                texts.append(str(chunk))
        
        # Check if any text content was extracted
        if not texts:
            logging.warning("No text content found in chunks")
            return np.array([])
        
        # Log the embedding generation process
        logging.info(f"Generating embeddings for {len(texts)} chunks")
        
        try:
            # Generate embeddings using the sentence transformer model
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,  # Show progress bar for large batches
                normalize_embeddings=True,  # Normalize to unit vectors for cosine similarity
                batch_size=32  # Process 32 texts at a time for efficiency
            )
            # Log successful embedding generation
            logging.info(f"Embeddings generated: {embeddings.shape}")
            # Convert to float32 for FAISS compatibility and return
            return embeddings.astype('float32')
        except Exception as e:
            # Log any errors during embedding generation
            logging.error(f"Embedding generation failed: {str(e)}")
            return np.array([])
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings (utility method)
        """
        # Check if texts list is empty
        if not texts:
            return np.array([])
        
        try:
            # Generate embeddings with normalization
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            # Convert to float32 and return
            return embeddings.astype('float32')
        except Exception as e:
            # Log errors and return empty array
            logging.error(f"Text embedding failed: {str(e)}")
            return np.array([])