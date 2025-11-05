# Import necessary libraries and modules
from pathlib import Path  # For handling file paths in a cross-platform way
from typing import List, Any  # For type hints
from langchain_core.documents import Document  # LangChain's document structure
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, CSVLoader, 
    Docx2txtLoader, UnstructuredExcelLoader
)  # Import loaders for different file formats
import json  # For handling JSON files
from logger import logging # For logging and debugging


class DocumentLoader:
    def __init__(self):
        """
        Initialize the DocumentLoader with a mapping of file extensions to loader methods.
        This makes it easy to add support for new file types.
        """
        # Dictionary mapping file extensions to their respective loader methods
        self.loader_map = {
            '.pdf': self._load_pdf,      # PDF files
            '.txt': self._load_text,     # Text files
            '.csv': self._load_csv,      # CSV files
            '.docx': self._load_docx,    # Word documents
            '.xlsx': self._load_excel,   # Excel files (new format)
            '.xls': self._load_excel,    # Excel files (old format)
            '.json': self._load_json     # JSON files
        }
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF files using PyMuPDFLoader
        """
        # Create a PDF loader instance with markdown table extraction
        loader = PyMuPDFLoader(file_path=file_path,mode='single', extract_tables="markdown")
        # Load and return the documents
        return loader.load()
    
    def _load_text(self, file_path: str) -> List[Document]:
        """
        Load text files with UTF-8 encoding
        """
        # Create a text loader with UTF-8 encoding to handle special characters
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """
        Load CSV files
        """
        # Create a CSV loader (automatically handles headers and rows)
        loader = CSVLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """
        Load Word documents
        """
        # Create a DOCX loader for Microsoft Word files
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def _load_excel(self, file_path: str) -> List[Document]:
        """
        Load Excel files (both .xlsx and .xls formats)
        """
        # Create an Excel loader for spreadsheet files
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()
    
    def _load_json(self, file_path: str) -> List[Document]:
        """
        Load JSON files and convert them to document format
        This is a custom implementation since LangChain doesn't have a built-in JSON loader
        """
        # Open and read the JSON file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Parse JSON content into Python objects
        
        # Handle different JSON structures:
        if isinstance(data, list):
            # If JSON is a list, convert each item to string
            texts = [str(item) for item in data]
        elif isinstance(data, dict):
            # If JSON is a dictionary, convert key-value pairs to strings
            texts = [f"{key}: {value}" for key, value in data.items()]
        else:
            # For other types (string, number, etc.), convert to string directly
            texts = [str(data)]
        
        # Create LangChain Document objects from the extracted texts
        return [Document(page_content=text) for text in texts]
    
    def load_single_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on its file extension
        """
        # Get the file extension in lowercase (e.g., '.pdf', '.txt')
        suffix = file_path.suffix.lower()
        
        # Check if the file type is supported
        if suffix not in self.loader_map:
            # Log a warning for unsupported file types
            logging.warning(f"Unsupported file type: {suffix} for {file_path}")
            return []  # Return empty list for unsupported files
        
        try:
            # Call the appropriate loader method based on file extension
            documents = self.loader_map[suffix](str(file_path))
            # Log successful loading
            logging.info(f"Successfully loaded {len(documents)} chunks from {file_path}")
            return documents
        except Exception as e:
            # Log any errors that occur during loading
            logging.error(f"Failed to load {file_path}: {str(e)}")
            return []  # Return empty list on failure

def load_all_documents(data_dir: str) -> List[Document]:
            """
            Main function: Load all documents from a directory and its subdirectories
            """
            # Convert the data directory path to a Path object and resolve it to absolute path
            data_path = Path(data_dir).resolve()
            
            # Check if the directory exists
            if not data_path.exists():
                logging.error(f"Data directory does not exist: {data_path}")
                return []  # Return empty list if directory doesn't exist
            
            # Log the directory being scanned
            logging.info(f"Loading documents from: {data_path}")
            
            # Create a DocumentLoader instance
            loader = DocumentLoader()
            # Initialize an empty list to store all loaded documents
            all_documents = []
            # Get all supported file extensions from the loader map
            supported_extensions = tuple(loader.loader_map.keys())
            
            # Find all files in the directory and its subdirectories
            files = list(data_path.glob("**/*"))  # '**/*' means all files in all subdirectories
            # Filter only supported files that are actual files (not directories)
            supported_files = [f for f in files if f.suffix.lower() in supported_extensions and f.is_file()]
            
            # Log how many supported files were found
            logging.info(f"Found {len(supported_files)} supported files")
            
            # Load each supported file
            for file_path in supported_files:
                # Load documents from the current file
                documents = loader.load_single_document(file_path)
                # Add the loaded documents to our main list
                all_documents.extend(documents)
            
            # Log the total number of documents loaded
            logging.info(f"Total documents loaded: {len(all_documents)}")
            # Return all loaded documents
            return all_documents