# app/rag/image_reader.py
import os
from .image_loader import image_to_text

# Define which image formats the OpenAI Vision model can process
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp")

def load_all_images_text(image_dir, client):
    """
    Scans a folder for images and converts them into text descriptions.
    
    Args:
        image_dir (str): Path to the folder containing your images.
        client: The initialized OpenAI client.
        
    Returns:
        list: A list of dictionaries containing the text and the filename.
    """
    documents = []

    # 1. Check if the directory exists to prevent the app from crashing
    if not os.path.exists(image_dir):
        print(f" Warning: Image directory not found: {image_dir}")
        return documents

    # 2. Loop through every file inside the folder
    for file in os.listdir(image_dir):
        
        # 3. Only process files with supported image extensions
        if file.lower().endswith(SUPPORTED_EXT):
            path = os.path.join(image_dir, file)
            print(f" Loading image: {file}")

            # 4. Use the image_loader to get a text description from GPT-4o-mini
            text = image_to_text(path, client)

            # 5. Store the result as a dictionary
            # Keeping the 'source' allows the bot to cite its sources later
            documents.append({
                "text": text,
                "source": file
            })

    # Return the full list of image-based descriptions
    return documents