# app/rag/vector_store.py
import faiss
import numpy as np

def create_faiss_index(vectors, texts, metadatas):
    """
    Creates a high-speed search index.
    
    Args:
        vectors (list): The list of numerical embeddings (math versions of your text).
        texts (list): The actual human-readable text chunks.
        metadatas (list): Info about the source (e.g., filename, page number).
        
    Returns:
        dict: A bundle containing the FAISS search object and the corresponding data.
    """

    # 1. Safety Check
    # If there is no data to index, we stop early to prevent errors.
    if not vectors:
        raise ValueError("No vectors provided. Please check your PDF/Image folders.")

    # 2. Define the Dimensions
    # 'dim' is the length of the vector (e.g., 1536 for OpenAI embeddings).
    # All vectors in the index must have the exact same length.
    dim = len(vectors[0])

    # 3. Choose the Index Type
    # IndexFlatL2 calculates the straight-line distance (Euclidean) between vectors.
    # It is very accurate for small-to-medium sized datasets like yours.
    index = faiss.IndexFlatL2(dim)

    # 4. Add Data to the Index
    # We stack the vectors into a matrix and convert them to 'float32', 
    # which is the specific format FAISS requires for high-speed math.
    index.add(np.vstack(vectors).astype("float32"))

    # 5. Return the Knowledge Bundle
    # We return a dictionary so the 'Retriever' knows which text 
    # belongs to which mathematical vector.
    return {
        "faiss": index,
        "texts": texts,
        "metadatas": metadatas
    }
