import streamlit as st
import os
import time
import json
from datetime import datetime
from pathlib import Path
from src.Search_1 import RAGSearch

# Configure page
st.set_page_config(
    page_title="🤖 Doc Search AI",
    page_icon="🔍",
    layout="wide"
)

class SimpleRAGApp:
    def __init__(self):
        self.rag_system = None
        self.data_dir = "data"
        self.metadata_file = "doc_metadata.json"
        self.setup_directories()
        self.load_metadata()
        self.init_rag()
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            self.save_metadata({})
    
    def load_metadata(self):
        """Load document metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                st.session_state.doc_meta = json.load(f)
        except:
            st.session_state.doc_meta = {}
    
    def save_metadata(self, meta=None):
        """Save document metadata"""
        if meta is None:
            meta = st.session_state.doc_meta
        with open(self.metadata_file, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def init_rag(self):
        """Initialize RAG system"""
        try:
            with st.spinner("🚀 Starting AI System..."):
                self.rag_system = RAGSearch()
            # Don't show success message every time
            if 'init_shown' not in st.session_state:
                st.success("✅ Ready to search!")
                st.session_state.init_shown = True
        except Exception as e:
            st.error(f"❌ Startup failed: {e}")
    
    def get_file_count(self):
        """Get number of uploaded files"""
        return len(st.session_state.doc_meta)
    
    def display_header(self):
        """Display main header"""
        st.title("🤖 Doc Search AI")
        file_count = self.get_file_count()
        
        st.info(f"""
        **How to use:**
        1. 📁 Upload documents (PDF, TXT, DOCX, CSV, Excel, JSON)
        2. 🔄 Click 'Update Search Index' 
        3. 💬 Ask questions about your documents
        4. 🔍 Get AI answers with sources
        
        **Currently have:** {file_count} documents
        """)
    
    def display_sidebar(self):
        """Display sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # File upload - FIX: Only process when new files are selected
            st.subheader("📤 Upload Files")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'docx', 'csv', 'xlsx', 'json'],
                accept_multiple_files=True,
                key="file_uploader"
            )
            
            # FIX: Only process if we have new files and button is clicked
            if uploaded_files and len(uploaded_files) > 0:
                if st.button("📥 Process Uploaded Files", key="process_upload_btn"):
                    self.process_uploads(uploaded_files)
            
            # Management buttons
            st.subheader("🛠️ Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Update Index", use_container_width=True, key="update_btn"):
                    self.rebuild_index()
            
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True, type="secondary", key="clear_btn"):
                    self.clear_all()
            
            # Stats
            st.subheader("📊 Stats")
            if self.rag_system:
                try:
                    stats = self.rag_system.get_vectorstore_stats()
                    st.metric("Indexed Chunks", stats["total_vectors"])
                    st.metric("Stored Files", self.get_file_count())
                except:
                    st.write("No stats available")
    
    def process_uploads(self, uploaded_files):
        """Process uploaded files - FIXED INFINITE LOOP"""
        if not uploaded_files:
            return
        
        # FIX: Check if we already processed these files
        current_files = set()
        for file in uploaded_files:
            file_id = f"{file.name}_{file.size}"
            current_files.add(file_id)
        
        # Check if these are the same files we already processed
        if 'last_processed_files' in st.session_state:
            if st.session_state.last_processed_files == current_files:
                st.info("📁 These files were already processed.")
                return
        
        # Store current files to avoid reprocessing
        st.session_state.last_processed_files = current_files
        
        success = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, file in enumerate(uploaded_files):
                if file is None or file.size == 0:
                    continue
                    
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"📥 Saving {file.name}...")
                
                try:
                    # Ensure data directory exists
                    os.makedirs(self.data_dir, exist_ok=True)
                    
                    # Create safe filename
                    safe_filename = "".join(c for c in file.name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                    file_path = os.path.join(self.data_dir, safe_filename)
                    
                    # Avoid overwriting
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(file_path):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        file_path = os.path.join(self.data_dir, safe_filename)
                        counter += 1
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Generate unique ID
                    doc_id = f"doc_{int(time.time()*1000)}_{i}"
                    
                    # Add to metadata
                    st.session_state.doc_meta[doc_id] = {
                        "filename": safe_filename,
                        "original_name": file.name,
                        "upload_date": datetime.now().isoformat(),
                        "size": file.size,
                        "file_path": file_path
                    }
                    
                    self.save_metadata()
                    success += 1
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    st.error(f"❌ Failed to save {file.name}: {str(e)}")
            
            progress_bar.progress(1.0)
            
            if success > 0:
                status_text.text(f"✅ Uploaded {success} files!")
                st.success(f"✅ Successfully uploaded {success} file(s)!")
                st.info("💡 Click 'Update Index' to make them searchable")
                
                # Auto-refresh
                time.sleep(2)
                st.rerun()
            else:
                status_text.text("❌ No files were uploaded")
                
        except Exception as e:
            st.error(f"❌ Upload process failed: {str(e)}")
        finally:
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    def rebuild_index(self):
        """Rebuild search index"""
        try:
            with st.spinner("🔄 Updating search index..."):
                # Remove old index
                if os.path.exists("faiss_store"):
                    import shutil
                    shutil.rmtree("faiss_store")
                
                # Create new system
                self.rag_system = RAGSearch()
                
                stats = self.rag_system.get_vectorstore_stats()
                st.success(f"✅ Index updated! Now have {stats['total_vectors']} searchable chunks")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Update failed: {e}")
    
    def clear_all(self):
        """Clear all documents"""
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
            
        if not st.session_state.confirm_clear:
            if st.button("🔥 Click to Confirm Delete", type="primary", key="confirm_clear_btn"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            if st.button("🗑️ CONFIRM DELETE ALL FILES", type="primary", key="final_clear_btn"):
                try:
                    import shutil
                    if os.path.exists(self.data_dir):
                        shutil.rmtree(self.data_dir)
                    if os.path.exists("faiss_store"):
                        shutil.rmtree("faiss_store")
                    
                    os.makedirs(self.data_dir, exist_ok=True)
                    
                    st.session_state.doc_meta = {}
                    self.save_metadata({})
                    
                    self.rag_system = RAGSearch()
                    
                    st.session_state.confirm_clear = False
                    st.session_state.last_processed_files = set()  # Reset processed files
                    
                    st.success("✅ All data cleared!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Clear failed: {e}")
    
    def display_files(self):
        """Show uploaded files"""
        if not st.session_state.doc_meta:
            st.info("📭 No documents yet. Upload files using the sidebar.")
            return
        
        st.subheader("📁 Your Documents")
        
        for doc_id, meta in list(st.session_state.doc_meta.items()):
            col1, col2 = st.columns([4, 1])
            with col1:
                date = datetime.fromisoformat(meta['upload_date']).strftime("%b %d, %H:%M")
                size_kb = meta['size'] / 1024
                display_name = meta.get('original_name', meta['filename'])
                st.write(f"**{display_name}**")
                st.caption(f"Uploaded: {date} | Size: {size_kb:.1f} KB")
            
            with col2:
                if st.button("Delete", key=f"delete_{doc_id}"):
                    self.delete_file(doc_id, meta)
            
            st.markdown("---")
    
    def delete_file(self, doc_id, meta):
        """Delete a single file"""
        try:
            file_path = meta.get('file_path', os.path.join(self.data_dir, meta['filename']))
            if os.path.exists(file_path):
                os.remove(file_path)
            
            if doc_id in st.session_state.doc_meta:
                del st.session_state.doc_meta[doc_id]
                self.save_metadata()
                
            st.success(f"✅ Deleted {meta.get('original_name', meta['filename'])}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Delete failed: {e}")
    
    def display_search(self):
        """Display search interface"""
        st.subheader("💬 Ask a Question")
        
        query = st.text_input(
            "What would you like to know about your documents?",
            placeholder="e.g., What are the main points? Explain machine learning...",
            key="search_input"
        )
        
        if query and query.strip():
            if self.get_file_count() == 0:
                st.warning("⚠️ Please upload documents first")
                return
            
            self.search_docs(query.strip())
    
    def search_docs(self, query):
        """Search documents and show results"""
        try:
            with st.spinner("🔍 Searching your documents..."):
                response = self.rag_system.search_and_summarize(
                    query=query, 
                    top_k=5, 
                    include_sources=True
                )
            
            st.subheader("🤖 AI Answer")
            st.markdown("---")
            st.write(response)
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful", key="helpful_btn"):
                    st.success("Thanks for your feedback! 🎉")
            with col2:
                if st.button("🔄 New Search", key="new_search_btn"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            st.info("Make sure you have documents uploaded and the index is updated.")
    
    def run(self):
        """Main app runner"""
        if not self.rag_system:
            st.error("❌ System not initialized. Please check if all dependencies are installed.")
            return
        
        self.display_header()
        self.display_sidebar()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            self.display_files()
            self.display_search()

# Run the app
if __name__ == "__main__":
    # Initialize session state
    if 'doc_meta' not in st.session_state:
        st.session_state.doc_meta = {}
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    if 'last_processed_files' not in st.session_state:
        st.session_state.last_processed_files = set()
    if 'init_shown' not in st.session_state:
        st.session_state.init_shown = False
        
    app = SimpleRAGApp()
    app.run()