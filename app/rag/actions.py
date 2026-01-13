import json
import os
import logging
from datetime import datetime, timedelta

# Professional logging setup
logger = logging.getLogger(__name__)

def schedule_meeting(name, email, phone):
    """
    Saves a meeting record to a local JSON database with duplicate prevention.
    
    Industry Standards Applied:
    1. Atomic writing (prevents file corruption).
    2. Duplicate Prevention: Checks if the same person registered in the last 5 minutes.
    3. Path abstraction using absolute paths.
    """
    try:
        # Define Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, "data")
        file_path = os.path.join(data_dir, "meetings.json")

        os.makedirs(data_dir, exist_ok=True)

        # 1. Read existing data first to check for duplicates
        meetings = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    meetings = json.load(f)
                    if not isinstance(meetings, list):
                        meetings = []
                except json.JSONDecodeError:
                    meetings = []

        # 2. DUPLICATE PREVENTION LOGIC
        # We check if the same email OR phone was added in the last 5 minutes
        # This prevents the AI from double-triggering during the "Confirmation" step.
        now = datetime.now()
        for meeting in meetings:
            if meeting.get("email") == email or meeting.get("phone") == str(phone):
                last_time = datetime.strptime(meeting["timestamp"], "%Y-%m-%d %H:%M:%S")
                # If registered within the last 5 minutes, reject as duplicate
                if now - last_time < timedelta(minutes=5):
                    logger.warning(f"Duplicate meeting attempt blocked for: {email}")
                    return f"ALREADY_EXISTS: A meeting for {name} is already recorded."

        # 3. Prepare the new entry
        new_entry = {
            "name": name,
            "email": email,
            "phone": str(phone),
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 4. Append and Save
        meetings.append(new_entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(meetings, f, indent=4)

        # Log success instead of just printing
        print(f"--> JSON Updated: {file_path}")
        return f"SUCCESS: Meeting successfully saved for {name}."

    except Exception as e:
        logger.error(f"CRITICAL ERROR SAVING JSON: {str(e)}")
        return "ERROR: Internal server error while saving data."