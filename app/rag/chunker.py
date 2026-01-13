#  app/rag/chunker.py
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Splits a long string into smaller overlapping segments (chunks).
    
    Args:
        text (str): The full raw text extracted from PDFs or Images.
        chunk_size (int): Max number of characters per chunk.
        chunk_overlap (int): Number of characters to repeat from the previous chunk.
    """
    
    # Initialize an empty list to store our final text segments
    chunks = []
    
    # The 'start' variable acts as a cursor or pointer 
    # indicating where the current slice begins.
    start = 0
    
    # Continue looping as long as our starting cursor 
    # hasn't reached the end of the full text.
    while start < len(text):
        
        # Calculate the end position of the current chunk.
        # Python slicing (text[start:end]) is 'exclusive' of the end index.
        end = start + chunk_size
        
        # 'Slice' the string from the current start to the calculated end.
        # This extracts a sub-string of approximately 'chunk_size' characters.
        chunk = text[start:end]
        
        # Save this specific piece of text into our list.
        chunks.append(chunk)
        
        # Move the 'start' cursor forward. 
        # Instead of moving by the full chunk_size, we subtract the overlap.
        # This ensures the next chunk begins 'chunk_overlap' characters 
        # BEFORE the current one ended.
        start += (chunk_size - chunk_overlap)
        
    # Return the completed list of segments to be converted into embeddings.
    return chunks
