import os
import json
import uuid
import datetime

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "settings.json")
FAVORITES_FILE = os.path.join(SCRIPT_DIR, "favorites.json")
CHAT_HISTORY_FILE = os.path.join(SCRIPT_DIR, "chat_history.json")

APP_TITLE = "🤗 Hugging Face Hub Manager & One-Click Downloader"
DEFAULT_WINDOW_SIZE = "1240x840"
MIN_WINDOW_SIZE = (1040, 700)

def load_settings():
    """Loads configuration settings from settings.json."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}
    else:
        settings = {}

    if "download_dir" not in settings:
        settings["download_dir"] = ""
    if "hf_token" not in settings:
        settings["hf_token"] = ""
    if "language" not in settings:
        settings["language"] = "tr"

    return settings

def save_settings(settings):
    """Saves configuration settings to settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")

# ==========================================
# FAVORITES PERSISTENCE LOGIC
# ==========================================
def load_favorites():
    """Loads favorited models from favorites.json."""
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_favorites(favorites):
    """Saves favorited models list to favorites.json."""
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favorites, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving favorites: {e}")

def is_favorite(repo_id):
    """Returns True if the given repo_id is in favorites."""
    favs = load_favorites()
    return any(item.get("repo_id") == repo_id for item in favs)

def add_to_favorites(repo_id, repo_type="model", note=""):
    """Adds a model/dataset to favorites."""
    favs = load_favorites()
    for item in favs:
        if item.get("repo_id") == repo_id:
            item["note"] = note
            item["repo_type"] = repo_type
            save_favorites(favs)
            return favs
    
    prefix = "https://huggingface.co/datasets/" if repo_type == "dataset" else "https://huggingface.co/"
    new_fav = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "url": f"{prefix}{repo_id}",
        "note": note,
        "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    favs.insert(0, new_fav)
    save_favorites(favs)
    return favs

def remove_from_favorites(repo_id):
    """Removes a model/dataset from favorites."""
    favs = load_favorites()
    favs = [item for item in favs if item.get("repo_id") != repo_id]
    save_favorites(favs)
    return favs

def update_favorite_note(repo_id, note):
    """Updates the note for an existing favorite item."""
    favs = load_favorites()
    for item in favs:
        if item.get("repo_id") == repo_id:
            item["note"] = note
            break
    save_favorites(favs)
    return favs

# ==========================================
# CORTEX AI CHAT HISTORY BACKUP LOGIC
# ==========================================
def load_chat_sessions():
    """Loads saved chat sessions from chat_history.json."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_sessions(sessions):
    """Saves chat sessions list to chat_history.json."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def create_chat_session(title="Yeni Sohbet"):
    """Creates a new ChatGPT-style chat session."""
    sessions = load_chat_sessions()
    session_id = str(uuid.uuid4())[:8]
    new_session = {
        "session_id": session_id,
        "title": title,
        "created_at": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
        "messages": [
            {
                "sender": "cortex",
                "text": "Merhaba! Ben Cortex AI Yapay Zeka Asistanınız.\nNasıl bir model veya yapay zeka görevi aradığınızı söyleyin, size en uygun projeleri ve doğrudan indirme bağlantılarını sunayım!",
                "timestamp": datetime.datetime.now().strftime("%H:%M")
            }
        ]
    }
    sessions.insert(0, new_session)
    save_chat_sessions(sessions)
    return new_session

def add_chat_message(session_id, sender, text):
    """Adds a message to a chat session."""
    sessions = load_chat_sessions()
    for sess in sessions:
        if sess["session_id"] == session_id:
            msg = {
                "sender": sender,
                "text": text,
                "timestamp": datetime.datetime.now().strftime("%H:%M")
            }
            sess["messages"].append(msg)
            # Update title if it's default
            if len(sess["messages"]) == 2 and sender == "user":
                clean_title = text[:30].replace("\n", " ").strip()
                if clean_title:
                    sess["title"] = clean_title
            save_chat_sessions(sessions)
            return sess
    return None

def delete_chat_session(session_id):
    """Deletes a chat session from backup."""
    sessions = load_chat_sessions()
    sessions = [s for s in sessions if s["session_id"] != session_id]
    save_chat_sessions(sessions)
    return sessions
