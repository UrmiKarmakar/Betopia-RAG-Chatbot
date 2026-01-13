# app/voice/stt_openai.py
import logging

# Silence logging for this specific module to keep terminal clean
logger = logging.getLogger(__name__)

def speech_to_text(client, audio_path: str) -> str:
    """
    Converts speech from an audio file into text using OpenAI's Whisper-1 model.
    Uses prompt steering to ensure brand names like Betopia and BDCalling are spelled correctly.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                # The prompt helps correct brand names and technical jargon
                prompt="The user is talking about Betopia, BDCalling, and RAG chatbots." 
            )
        
        return transcription.text.strip()

    except Exception as e:
        logger.error(f"Error during Speech-to-Text: {str(e)}")
        return ""