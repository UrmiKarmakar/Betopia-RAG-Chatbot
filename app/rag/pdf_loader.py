# app/rag/pdf_loader.py
import os
from PyPDF2 import PdfReader

def load_all_pdfs_text(pdf_dir):
    """
    Scans a folder for PDF files and extracts all text content from them.
    
    Args:
        pdf_dir (str): Path to the folder containing your PDFs (e.g., 'data/pdf').
        
    Returns:
        list: A list of dictionaries, each containing extracted 'text' and the 'source' filename.
    """
    documents = []

    # 1. Check if the directory exists
    if not os.path.exists(pdf_dir):
        print(f" PDF folder not found: {pdf_dir}")
        return documents

    # 2. Iterate through every file in the folder
    for file in os.listdir(pdf_dir):
        
        # 3. Only process files that end with the .pdf extension
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            print(f" Loading PDF: {file}")

            # 4. Initialize the PDF reader
            reader = PdfReader(path)
            text = ""
            
            # 5. Loop through every page and extract text
            for page in reader.pages:
                # We use 'or ""' to handle cases where a page might be empty/unreadable
                text += page.extract_text() or ""

            # 6. Append a structured dictionary to our documents list
            # We use .strip() to remove unnecessary whitespace at the start/end
            if text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": file
                })

    return documents