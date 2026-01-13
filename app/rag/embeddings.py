# app/rag/embeddings.py
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. Configuration & Setup
# load_dotenv() searches for a .env file to load your secret API keys into the system environment.
load_dotenv()

# We fetch the API key from the environment. This keeps your key safe and out of the source code.
api_key = os.getenv("OPENAI_API_KEY")

# Safety Check: If the key is missing, the program stops immediately with a clear error message.
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")

# 2. Client Initialization
# Create an instance of the OpenAI class. This object handles the connection to OpenAI's servers.
client = OpenAI(api_key=api_key)

def embed_texts(texts):
    """
    Converts a list of text chunks into numerical vectors (embeddings).

    Args:
        texts (list[str]): The text pieces created by your chunking function.

    Returns:
        list[np.ndarray]: A list of vectors representing the meaning of the text.
    """
    
    # Initialize an empty list to store the numerical vectors.
    embeddings = []

    # 3. Processing the Chunks
    # We loop through every single text chunk in the list.
    for t in texts:
        try:
            # Request a 'vector' from the OpenAI API.
            # We use "text-embedding-3-small", which is fast and cost-effective.
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=t
            )
            
            # The API returns a list of floats. We convert this into a 'numpy array' 
            # (np.array) because FAISS (your vector store) requires numpy format for fast math.
            vector = np.array(resp.data[0].embedding)
            embeddings.append(vector)
            
        except Exception as e:
            # If the internet fails or the API crashes, we print a snippet of the 
            # text that failed so you can troubleshoot without crashing the whole bot.
            print(f"Error embedding text: {t[:50]}... | {e}")

    # Return the list of numerical vectors to be stored in the FAISS index.
    return embeddings
