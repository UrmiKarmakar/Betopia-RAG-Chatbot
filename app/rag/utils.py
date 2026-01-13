# app/rag/utils.py
import os
import hashlib
import json

MANIFEST_PATH = "data/manifest.json"

def file_hash(path: str) -> str:
    """
    Creates a unique fingerprint for a file.
    If even one character changes in a PDF, this hash will be completely different.
    """
    # 'rb' means read binary, which is required for images and PDFs
    with open(path, "rb") as f:
        # We use MD5 for speed; it's perfect for checking if files changed
        return hashlib.md5(f.read()).hexdigest()

def load_manifest() -> dict:
    """
    Reads the manifest.json file from disk. 
    This is the bot's 'memory' of the files from the last run.
    """
    if not os.path.exists(MANIFEST_PATH):
        return {} # Return empty if it's the first time running the bot
    
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_manifest(manifest_data: dict):
    """
    Saves the current state of files to manifest.json.
    """
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=4)

def file_metadata(path: str, version: int = 1):
    """
    Generates a structured dictionary of information about a file.
    This is useful for 'Source Citing' in your chat responses.
    """
    return {
        "doc_id": file_hash(path),
        "doc_name": os.path.basename(path),
        "updated_at": int(os.path.getmtime(path)), # The timestamp of the last edit
        "version": version,
        "priority": version,
    }