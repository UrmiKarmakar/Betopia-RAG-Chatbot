# app/rag/image_loader.py
import base64

def encode_image(path):
    """
    Computers and APIs cannot 'see' a file on your hard drive directly. 
    We must convert the binary image file into a Base64 string (a long string of text)
    so it can be sent inside a standard JSON API request.
    """
    # 'rb' stands for 'read binary' - necessary for non-text files like images
    with open(path, "rb") as image_file:
        # 1. Read the raw bytes
        # 2. Encode those bytes into base64 format
        # 3. Decode into a UTF-8 string so it can be handled as text
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_text(image_path, client):
    """
    This function sends the image to OpenAI's Vision model (GPT-4o-mini)
    and asks the AI to describe it. This description becomes the 'text' 
    version of the image for your RAG database.
    
    Args:
        image_path (str): The local path to your image (e.g., 'data/images/chart.png')
        client: Your initialized OpenAI client
    """
    
    # First, convert the image file into a sendable string format
    image_base64 = encode_image(image_path)

    # Call the GPT-4o-mini Vision model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    # Content item 1: The instruction (Prompt)
                    {
                        "type": "text", 
                        "text": "Describe this image clearly for knowledge retrieval. "
                                "Focus on any text, data, or company facts visible."
                    },
                    # Content item 2: The actual image data
                    {
                        "type": "image_url",
                        "image_url": {
                            # We create a 'Data URL' which tells the API the format 
                            # and includes the encoded image string.
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        # Temperature 0 makes the description factual and consistent
        temperature=0
    )

    # Extract the AI's description of the image and clean up whitespace
    return response.choices[0].message.content.strip()