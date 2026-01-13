import os
import time
import json
import shlex
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. SILENCE LOGGING: Keeps the terminal clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Custom RAG & Voice Imports
from rag.pdf_loader import load_all_pdfs_text
from rag.image_reader import load_all_images_text
from rag.chunker import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import create_faiss_index
from rag.retriever import retrieve_chunks
from rag.prompt import build_prompt
from rag.upload_manager import save_uploaded_files, build_temp_index, clear_tmp_dir
from rag.actions import schedule_meeting 
from voice.stt import record_audio, cleanup_audio
from voice.stt_openai import speech_to_text
from voice.tts import speak_text

# CONFIGURATION
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the Tool Schema
TOOLS = [{
    "type": "function",
    "function": {
        "name": "schedule_meeting",
        "description": "ONLY call this if the user EXPLICITLY asks to book a meeting.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User's full name"},
                "email": {"type": "string", "description": "User's email"},
                "phone": {"type": "string", "description": "User's phone number"}
            },
            "required": ["name", "email", "phone"]
        }
    }
}]

# PATHS & SESSION STATE
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TMP_UPLOAD_DIR = DATA_DIR / "tmp"
MAX_MEMORY_TURNS = 10

conversation_history = []
temp_index = None
meeting_scheduled_in_session = False
voice_output_enabled = False 

# HELPER FUNCTIONS

def show_history():
    """Displays the in-memory session history."""
    if not conversation_history:
        print("\n History is empty for this session.")
        return
    print("\n" + "="*70)
    print(f"{'INDEX':<5} | {'SENDER':<8} | {'MESSAGE'}")
    print("-" * 70)
    for i, turn in enumerate(conversation_history):
        print(f"{i:<5} | {'User':<8} | {turn['user']}")
        bot_short = (turn['assistant'][:60] + '...') if len(turn['assistant']) > 60 else turn['assistant']
        print(f"{' ': <5} | {'Bot':<8} | {bot_short}")
    print("="*70 + "\n")

# STARTUP LOGIC
print("\n Loading Knowledge Base Documents...")
pdf_docs = load_all_pdfs_text(str(DATA_DIR / "pdf"))
image_docs = load_all_images_text(str(DATA_DIR / "images"), client)

documents = []
ts = int(time.time())

for doc in pdf_docs + image_docs:
    documents.append({"text": doc["text"], "metadata": {"source": doc["source"], "updated_at": ts}})

if documents:
    chunks, metadatas = [], []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"])
        chunks.extend(doc_chunks)
        metadatas.extend([doc["metadata"]] * len(doc_chunks))
    index = create_faiss_index(embed_texts(chunks), chunks, metadatas)
    print(f" Loaded {len(pdf_docs)} PDFs and {len(image_docs)} images.")
else:
    index = None

# MAIN INTERACTION LOOP
print("\n" + "="*50)
print("🤖 BETOPIA AI AGENT ONLINE")
print("="*50)
print("COMMANDS:")
print("• [Type Text] + Enter : Normal Chat")
print("• [Empty Enter]      : Voice Input Mode")
print("• /voice             : Toggle Text-to-Voice (On/Off)")
print("• /history           : View session logs")
print("• /upload <path>     : Add temp files")
print("• /clear             : Delete temp uploads")
print("• exit               : Close Assistant")
print("-" * 50)

try:
    while True:
        raw_input = input("\nYou: ").strip()
        is_voice_mode = False
        user_input = raw_input

        # 1. INPUT PROCESSING
        if raw_input == "":
            is_voice_mode = True
            audio_path = record_audio()
            user_input = speech_to_text(client, audio_path)
            cleanup_audio(audio_path) 
            if not user_input or len(user_input.strip()) < 2: continue
            print(f"🗣️  You said: {user_input}")

        if user_input.lower() == "exit":
            print("\n👋 Goodbye! Thanks for chatting with Betopia.")
            break

        # 2. COMMAND HANDLING
        if user_input.lower() == "/voice":
            voice_output_enabled = not voice_output_enabled
            print(f"🔊 Text-to-Voice: {'ENABLED' if voice_output_enabled else 'DISABLED'}")
            continue

        if user_input.lower() == "/history":
            show_history()
            continue

        if user_input.lower() == "/clear":
            clear_tmp_dir(str(TMP_UPLOAD_DIR))
            temp_index = None
            print("🧹 Temporary files cleared.")
            continue

        if user_input.startswith("/upload"):
            try:
                paths = shlex.split(user_input)[1:]
                save_uploaded_files(str(TMP_UPLOAD_DIR), paths)
                temp_index = build_temp_index(str(TMP_UPLOAD_DIR), client)
                print("✨ Temp index updated.")
            except Exception as e:
                print(f" Error: {e}")
            continue

        # 3. AI AGENT LOGIC (RAG + Tools)
        retrieved = []
        if index:
            retrieved.extend(retrieve_chunks(user_input, index, lambda x: embed_texts([x]), top_k=5))
        if temp_index:
            retrieved.extend(retrieve_chunks(user_input, temp_index, lambda x: embed_texts([x]), top_k=3))
        
        context = "\n\n".join(r["text"] for r in retrieved)
        history_pairs = [(h["user"], h["assistant"]) for h in conversation_history]
        prompt = build_prompt(context, user_input, history_pairs, meeting_status=meeting_scheduled_in_session)
        
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto")
        resp_msg = response.choices[0].message
        
        if resp_msg.tool_calls:
            messages.append(resp_msg)
            for tool_call in resp_msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                action_result = schedule_meeting(**args)
                if "SUCCESS" in action_result: meeting_scheduled_in_session = True
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": "schedule_meeting", "content": action_result})
            final_resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            answer = final_resp.choices[0].message.content
        else:
            answer = resp_msg.content

        # 4. OUTPUT
        print(f"\n🤖 Bot: {answer}")
        if is_voice_mode or voice_output_enabled: 
            speak_text(client, answer)
        
        print("-" * 60)
        conversation_history.append({"user": user_input, "assistant": answer})
        if len(conversation_history) > MAX_MEMORY_TURNS:
            conversation_history.pop(0)

except KeyboardInterrupt:
    print("\n👋Session ended. Goodbye!")
# finally:
#     clear_tmp_dir(str(TMP_UPLOAD_DIR))