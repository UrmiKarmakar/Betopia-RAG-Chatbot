# app/voice/stt.py
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
import logging

# Set up logger for industry-standard error tracking
logger = logging.getLogger(__name__)

def record_audio(duration: int = 6, samplerate: int = 16000) -> str:
    """
    Records audio from the default microphone and saves it to a temporary WAV file.
    
    Args:
        duration (int): How many seconds to record.
        samplerate (int): The frequency of the audio (16kHz is standard for AI models).
        
    Returns:
        str: The absolute path to the temporary wav file.
    """
    try:
        print(f" 🎙️  Listening for {duration} seconds...")
        
        # Capture the audio data from the microphone
        # int16 is the standard bit-depth for speech recognition
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="int16"
        )
        
        # Block execution until the recording is finished
        sd.wait()
        
        # Create a temporary file that persists after closing for processing
        # delete=False is crucial so the file isn't wiped before Whisper reads it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        
        # Save the numpy array as a standard WAV file
        wav.write(temp_path, samplerate, audio)
        
        return temp_path

    except Exception as e:
        logger.error(f"Failed to record audio: {str(e)}")
        print(" ❌ Error: Microphone access failed or sounddevice error.")
        return ""

def cleanup_audio(file_path: str):
    """
    Utility to remove the temporary audio file once it has been transcribed.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"Cleanup failed for {file_path}: {e}")