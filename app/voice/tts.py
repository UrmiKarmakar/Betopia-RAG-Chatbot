# app/voice/tts.py
import tempfile
import pygame
import os
import logging

# Silence pygame welcome message and library logs
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
logging.getLogger("pygame").setLevel(logging.WARNING)

def speak_text(client, text: str):
    """
    Converts text to speech using OpenAI TTS and plays it immediately.
    
    Args:
        client: The OpenAI client instance.
        text (str): The text content to be converted to speech.
    """
    try:
        # 1. Generate audio using the OpenAI TTS model
        # Note: 'tts-1' is the industry standard for real-time applications
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # 2. Create a temporary file to store the audio stream
        # delete=False allows pygame to access the file after closing the handle
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            # write() handles the binary stream from the API response
            temp_audio.write(response.read())
            temp_path = temp_audio.name

        # 3. Initialize pygame mixer and play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        # 4. Block execution while the audio is playing
        # Using a clock tick prevents the CPU from over-working during the loop
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

        # 5. Clean up: Stop the mixer and remove the temporary file
        pygame.mixer.music.unload()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    except Exception as e:
        print(f"⚠️ TTS Playback Error: {e}")