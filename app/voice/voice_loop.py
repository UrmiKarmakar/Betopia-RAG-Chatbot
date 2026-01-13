# app/voice/voice_loop.py
import os
import logging
from .stt import record_audio
from .stt_openai import speech_to_text
from .tts import speak_text

# Set up logging for professional error tracking
logger = logging.getLogger(__name__)

def voice_chat_loop(client, ask_rag_fn):
    """
    Manages the continuous Voice-to-Voice interaction loop.
    
    Args:
        client: Initialized OpenAI client.
        ask_rag_fn: The function from main.py that handles RAG retrieval and generation.
    """
    print("\n--- 🎙️ Voice Mode Activated (Say 'exit' to stop) ---")
    
    try:
        while True:
            # 1. Record audio from microphone (returns path to temp .wav)
            audio_path = record_audio()
            
            # 2. Transcribe audio to text
            user_text = speech_to_text(client, audio_path)

            # Cleanup: Delete the temp recording file immediately after transcription
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            # 3. Skip loop if no speech was detected
            if not user_text or not user_text.strip():
                continue

            print(f"\n🎤 You (Voice): {user_text}")

            # 4. Handle exit commands
            if user_text.lower().strip(".") in ["exit", "quit", "stop"]:
                exit_msg = "Goodbye! Returning to text mode."
                print(f"🤖 Bot: {exit_msg}")
                speak_text(client, exit_msg)
                break

            # 5. Get Answer from RAG system
            # ask_rag_fn should be the logic from your main.py loop
            answer = ask_rag_fn(user_text)

            # 6. Output Answer (Print and Speak)
            print(f"🤖 Bot: {answer}")
            speak_text(client, answer)

    except Exception as e:
        logger.error(f"Error in Voice Chat Loop: {e}")
        print(f"⚠️ Voice system encountered an error: {e}")

    finally:
        print("--- ⌨️ Text Mode Reactivated ---")