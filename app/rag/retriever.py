# app/rag/retriever.py
import numpy as np

def retrieve_chunks(query, index, embed_func, top_k=3):
    """
    Finds the most relevant pieces of text from the FAISS index.
    
    Args:
        query (str): The user's natural language question.
        index (dict): A dictionary containing the 'faiss' object, 'texts', and 'metadatas'.
        embed_func (function): The function that converts text into math vectors.
        top_k (int): How many relevant chunks to return (default is 3).
        
    Returns:
        list: A list of dictionaries containing text and source metadata.
    """
    
    # 1. Vectorize the User Question
    # We convert the user's text into the same "math language" (vectors) as our PDF chunks.
    # .astype("float32") is required by the FAISS library.
    q_vec = embed_func(query)[0].astype("float32")

    # 2. Mathematical Search
    # index["faiss"].search looks for the k-nearest vectors in the database.
    # D: Distances (how similar the results are).
    # I: Indices (the position IDs of the matching text).
    D, I = index["faiss"].search(
        np.array([q_vec]), 
        top_k
    )

    # 3. Reconstruct the Results
    results = []
    # I[0] contains the IDs of the top matching chunks
    for idx in I[0]:
        # If FAISS finds a match, we grab the actual text and its source metadata
        results.append({
            "text": index["texts"][idx],
            "metadata": index["metadatas"][idx]
        })

    return results
