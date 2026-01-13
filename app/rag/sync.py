# app/rag/sync.py
import os
from typing import List
from .utils import file_hash, load_manifest, save_manifest
from .pdf_loader import load_all_pdfs_text
from .image_reader import load_all_images_text
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import create_faiss_index

def gather_files(pdf_dir: str, img_dir: str) -> List[str]:
    """
    Scans both PDF and Image folders to create a list of all current files.
    """
    files = []
    # Check PDF folder
    if os.path.isdir(pdf_dir):
        for fn in os.listdir(pdf_dir):
            if fn.lower().endswith(".pdf"):
                files.append(os.path.join(pdf_dir, fn))
    
    # Check Image folder
    if os.path.isdir(img_dir):
        for fn in os.listdir(img_dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                files.append(os.path.join(img_dir, fn))
    return sorted(files)

def build_documents_list(pdf_dir: str, img_dir: str, client=None) -> list:
    """
    Aggregates all content from loaders. 
    Assumes loaders return: [{"text": "...", "source": "filename"}, ...]
    """
    pdf_docs = load_all_pdfs_text(pdf_dir)
    image_docs = load_all_images_text(img_dir, client)
    return pdf_docs + image_docs

def sync_and_rebuild(pdf_dir: str, img_dir: str, client) -> bool:
    """
    The main logic: Detects changes and rebuilds the index only if necessary.
    """
    # 1. Load the 'Last Known State' (manifest.json)
    manifest = load_manifest()
    
    # 2. Get the 'Current State' of the folders
    files = gather_files(pdf_dir, img_dir)
    current_map = {}
    for f in files:
        try:
            # Generate a unique fingerprint (hash) based on file content
            current_map[f] = file_hash(f)
        except Exception:
            current_map[f] = None

    # 3. Compare: Has anything changed?
    # Check for new or deleted files
    files_added_or_removed = set(manifest.keys()) != set(current_map.keys())
    
    # Check if existing files were edited
    content_changed = any(manifest.get(k) != current_map.get(k) for k in current_map)

    if not (files_added_or_removed or content_changed):
        print(" Data is in sync. No rebuild needed.")
        return False

    print(" Changes detected! Rebuilding FAISS index...")

    # 4. Save the new state so we don't rebuild again next time
    save_manifest(current_map)

    # 5. Extract all text from files
    docs = build_documents_list(pdf_dir, img_dir, client)
    
    all_chunks = []
    metadatas = []

    # 6. Process each document into chunks
    for doc in docs:
        body = doc["text"]
        source = doc["source"]

        chunks = chunk_text(body)
        for c in chunks:
            all_chunks.append(c)
            # Metadata allows the bot to say "I found this in file X"
            metadatas.append({
                "source": source,
                "text_preview": c[:100] # Useful for debugging
            })

    # 7. Create New Mathematical Vectors
    embeddings = embed_texts(all_chunks)
    
    # 8. Update the FAISS index file
    create_faiss_index(embeddings, all_chunks, metadatas)
    
    print(f" Index rebuilt successfully with {len(all_chunks)} chunks.")
    return True