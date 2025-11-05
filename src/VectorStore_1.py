# Import necessary libraries and modules
import os  # For file and directory operations
import faiss  # Facebook AI Similarity Search library for efficient vector search
import numpy as np  # For numerical operations
import pickle  # For serializing metadata to disk
from typing import List, Any, Dict  # For type hints
from sentence_transformers import SentenceTransformer  # For generating query embeddings
from src.Embedding_1 import EmbeddingPipeline  # Our custom embedding pipeline
from logger import logging  # For logging and debugging



class FaissVectorStore:
    def __init__(
        self, 
        persist_dir: str = "faiss_store",  # Directory to store the vector index
        embedding_model: str = "all-MiniLM-L6-v2",  # Model for generating embeddings
        chunk_size: int = 1200,  # Document chunk size
        chunk_overlap: int = 250  # Document chunk overlap
    ):
        """
        Initialize the FAISS vector store with configuration parameters
        """
        # Store the directory where index will be saved/loaded
        self.persist_dir = persist_dir
        # Create the directory if it doesn't exist (exist_ok=True prevents errors if it exists)
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize instance variables
        self.index = None  # FAISS index object (will be created later)
        self.metadata = []  # List to store metadata for each vector
        self.embedding_model = embedding_model  # Store model name for reference
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            # Load the embedding model for generating query embeddings
            self.model = SentenceTransformer(embedding_model)
            logging.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            # If model loading fails, log error and stop initialization
            logging.error(f"Failed to load embedding model: {str(e)}")
            raise  # Re-raise exception to prevent using broken vector store
    
    def build_from_documents(self, documents: List[Any]) -> bool:
        """
        Build the vector store from a list of documents
        Returns: True if successful, False if failed
        """
        # Validate input documents
        if not documents:
            logging.error("No documents provided for building vector store")
            return False
        
        logging.info(f"Building vector store from {len(documents)} documents")
        
        try:
            # Create embedding pipeline for chunking and embedding
            emb_pipe = EmbeddingPipeline(
                model_name=self.embedding_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Step 1: Split documents into chunks
            chunks = emb_pipe.chunk_documents(documents)
            if not chunks:
                logging.error("No chunks generated from documents")
                return False
            
            # Step 2: Generate embeddings for chunks
            embeddings = emb_pipe.embed_chunks(chunks)
            if embeddings.size == 0:
                logging.error("No embeddings generated")
                return False
            
            # Step 3: Create metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                # Extract source from chunk metadata properly
                source = "unknown"
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    source = chunk.metadata.get('source', 'unknown')
                
                meta = {
                    "text": chunk.page_content,
                    "source": source,
                    "chunk_id": len(metadata),
                    "chunk_index": i
                }
                metadata.append(meta)
            
            # Step 4: Add embeddings and metadata to the index
            self.add_embeddings(embeddings, metadata)
            
            # Step 5: Save to disk for future use
            self.save()
            
            logging.info(f"Vector store built successfully with {len(metadata)} chunks")
            return True
            
        except Exception as e:
            logging.error(f"Failed to build vector store: {str(e)}")
            return False
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict] = None):
        """
        Add embeddings and associated metadata to the FAISS index
        """
        # Check if we have embeddings to add
        if embeddings.size == 0:
            logging.warning("No embeddings to add")
            return
        
        # Get the dimensionality of embeddings (e.g., 384 for all-MiniLM-L6-v2)
        dim = embeddings.shape[1]
        
        # Create a new index if one doesn't exist
        if self.index is None:
            # Use Inner Product (dot product) for cosine similarity
            # Since we normalize embeddings, dot product = cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            logging.info(f"Created new FAISS index with dimension {dim}")
        
        # Normalize embeddings for cosine similarity
        # This converts vectors to unit length so that dot product = cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to the FAISS index
        self.index.add(embeddings)
        
        # Add associated metadata if provided
        if metadata:
            self.metadata.extend(metadata)
        
        # Log the operation
        logging.info(f"Added {embeddings.shape[0]} vectors to index. Total: {self.index.ntotal}")
    
    def save(self) -> bool:
        """
        Save the FAISS index and metadata to disk
        Returns: True if successful, False if failed
        """
        try:
            # Define file paths
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            
            # Save FAISS index if it exists
            if self.index is not None:
                faiss.write_index(self.index, faiss_path)
            
            # Save metadata using pickle serialization
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            
            logging.info(f"Saved vector store to {self.persist_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save vector store: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk
        Returns: True if successful, False if failed
        """
        try:
            # Define file paths
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            
            # Check if both files exist
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                logging.warning("Vector store files not found")
                return False
            
            # Load FAISS index from file
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata from pickle file
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            
            logging.info(f"Loaded vector store with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def exists(self) -> bool:
        """
        Check if vector store files exist on disk
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search the index with a query embedding
        Returns: List of search results with metadata and similarity scores
        """
        # Check if index is ready for searching
        if self.index is None or self.index.ntotal == 0:
            logging.warning("Index is empty or not initialized")
            return []
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        try:
            # Perform the search
            # D = distances (similarity scores), I = indices of results
            D, I = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Process and format results
            results = []
            # I[0]: Indices of top_k most similar vectors
            # D[0]: Similarity scores for those vectors 
            for idx, distance in zip(I[0], D[0]):
                # Check if index is within metadata bounds
                if idx < len(self.metadata):
                    # Convert distance to similarity score
                    # For IndexFlatIP, distance is actually cosine similarity (0-1)
                    similarity = float(distance)
                    results.append({
                        "index": idx,  # Position in the index
                        "similarity": similarity,  # Similarity score (0-1, higher is better)
                        "metadata": self.metadata[idx]  # Associated metadata
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return []
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query the vector store with text (converts to embedding automatically)
        """
        logging.info(f"Querying: '{query_text}'")
        
        try:
            # Convert query text to embedding
            query_emb = self.model.encode([query_text]).astype('float32')
            # Search using the embedding
            return self.search(query_emb, top_k=top_k)
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_metadata": len(self.metadata),
            "embedding_dim": self.index.d if self.index else 0,
            "persist_dir": self.persist_dir
        }
    
# For larger datasets, you can use:
# - faiss.IndexIVFFlat: Inverted file index for faster search
# - faiss.IndexHNSW: Graph-based index for very large datasets
# - faiss.IndexPQ: Product quantization for memory efficiency