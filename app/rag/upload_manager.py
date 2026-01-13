# app/rag/upload_manager.py
import os
import shutil
import time
import logging
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfReader

# Core RAG logic imports
from .image_loader import image_to_text
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import create_faiss_index

# Configuration for supported formats
SUPPORTED_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_DOC_EXT = (".pdf",) + SUPPORTED_IMAGE_EXT

# Set up logging for professional production tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_tmp_dir(tmp_dir: str) -> None:
    """
    Safely ensures the temporary directory exists.
    """
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info(f"Created temporary directory at: {tmp_dir}")

def save_uploaded_files(tmp_dir: str, paths: List[str]) -> List[str]:
    """
    Validates and copies user-provided files into a sandbox directory.
    
    Args:
        tmp_dir: The destination directory for the session.
        paths: A list of absolute/relative file paths to upload.
        
    Returns:
        List[str]: Paths of successfully saved files in the tmp directory.
    """
    ensure_tmp_dir(tmp_dir)
    saved_files = []
    
    for src in paths:
        # Check 1: Does the file exist?
        if not os.path.isfile(src):
            logger.warning(f"File not found, skipping: {src}")
            continue
            
        # Check 2: Is the extension supported?
        if not src.lower().endswith(SUPPORTED_DOC_EXT):
            logger.warning(f"Unsupported file type for: {src}")
            continue
            
        try:
            # Copy file to sandbox with original metadata
            dst = os.path.join(tmp_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            saved_files.append(dst)
        except Exception as e:
            logger.error(f"Failed to copy {src}: {str(e)}")
            
    return saved_files

def load_text_from_file(path: str, client: Any) -> Dict[str, Any]:
    """
    Extracts text content based on file type (PDF or Image).
    
    Returns a structured dictionary matching the system's document format.
    """
    filename = os.path.basename(path)
    logger.info(f"Extracting text from: {filename}")
    
    # Logic for PDF processing
    if filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return {
                "text": text.strip(), 
                "source": filename, 
                "type": "upload"
            }
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            return {"text": "", "source": filename, "type": "upload"}

    # Logic for Image processing using Vision AI
    elif filename.lower().endswith(SUPPORTED_IMAGE_EXT):
        try:
            text = image_to_text(path, client)
            return {
                "text": text, 
                "source": filename, 
                "type": "upload"
            }
        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            return {"text": "", "source": filename, "type": "upload"}
            
    else:
        raise ValueError(f"Incompatible file format encountered: {filename}")

def build_temp_index(tmp_dir: str, client: Any) -> Optional[Dict[str, Any]]:
    """
    Orchestrates the creation of a FAISS index from all files in the tmp directory.
    This index lives in memory for the duration of the session.
    """
    if not os.path.isdir(tmp_dir):
        logger.error(f"Temporary directory {tmp_dir} does not exist.")
        return None

    # Step 1: Load all valid documents from the sandbox
    docs = []
    for fn in os.listdir(tmp_dir):
        path = os.path.join(tmp_dir, fn)
        if os.path.isfile(path) and fn.lower().endswith(SUPPORTED_DOC_EXT):
            docs.append(load_text_from_file(path, client))

    if not docs:
        logger.info("No valid documents found to index in temporary storage.")
        return None

    # Step 2: Chunking and Metadata tagging
    timestamp = int(time.time())
    all_chunks = []
    all_metadatas = []
    
    for doc in docs:
        if not doc["text"]: continue # Skip empty extractions
        
        doc_chunks = chunk_text(doc["text"])
        all_chunks.extend(doc_chunks)
        
        # Link each chunk back to its source file and timestamp
        for chunk in doc_chunks:
            all_metadatas.append({
                "source": doc["source"],
                "type": doc["type"],
                "updated_at": timestamp,
                "text_preview": chunk[:100] # Useful for tracing sources in logs
            })

    # Step 3: Embedding and Vector Store creation
    logger.info(f"Generating embeddings for {len(all_chunks)} temporary chunks...")
    vectors = embed_texts(all_chunks)
    
    # Build the FAISS structure
    index = create_faiss_index(
        vectors=vectors, 
        texts=all_chunks, 
        metadatas=all_metadatas
    )
    
    logger.info("Successfully built session-specific temporary index.")
    return index

def clear_tmp_dir(tmp_dir: str) -> None:
    """
    Wipes the temporary directory. Call this on 'exit' or '/clear_uploads'
    to maintain user privacy and free up disk space.
    """
    if os.path.isdir(tmp_dir):
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Temporary directory {tmp_dir} has been cleared.")
        except Exception as e:
            logger.error(f"Error during directory cleanup: {e}")