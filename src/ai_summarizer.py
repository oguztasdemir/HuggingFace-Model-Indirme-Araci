import requests

OLLAMA_URL = "http://localhost:11434"

def check_ollama_status():
    """
    Checks if Ollama service is running on http://localhost:11434 and retrieves installed models.
    Returns dict: {"connected": bool, "models": list, "active_model": str, "message": str}
    """
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if res.status_code == 200:
            models_data = res.json()
            models = models_data.get("models", [])
            if models:
                model_names = [m["name"] for m in models]
                active_model = model_names[0]
                for name in model_names:
                    n_lower = name.lower()
                    if any(k in n_lower for k in ["llama", "gemma", "mistral", "qwen"]):
                        active_model = name
                        break
                return {
                    "connected": True,
                    "models": model_names,
                    "active_model": active_model,
                    "message": f"🟢 Ollama (Cortex Engine) Bağlı: {active_model}"
                }
            return {
                "connected": True,
                "models": [],
                "active_model": None,
                "message": "🟡 Ollama Bağlı ancak hiç model yüklenmemiş (ollama pull llama3 çalıştırın)."
            }
    except Exception:
        pass

    return {
        "connected": False,
        "models": [],
        "active_model": None,
        "message": "🔴 Ollama (Cortex Engine) Bağlantısı Yok (localhost:11434 kapalı)."
    }

def generate_summary(readme_content, repo_id):
    """
    Generates a Turkish summary of the model/dataset using local Ollama LLM.
    """
    if not readme_content:
        return "README içeriği alınamadı. Özet oluşturulamıyor."

    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        lines = [line.strip() for line in readme_content.split('\n') if line.strip() and not line.startswith('#')]
        summary = ""
        count = 0
        for line in lines:
            if not line.startswith(('```', '-', '*', '|', '[')):
                summary += line + " "
                count += 1
                if count > 4:
                    break
        summary = summary.strip()
        if not summary:
            summary = readme_content[:400] + "..."

        return (
            "💡 [Ollama / Cortex Engine Çalışmıyor]\n"
            "Yapay zeka özetleri için bilgisayarınızda Ollama'yı (http://localhost:11434) başlatın.\n\n"
            f"📄 Ham README Metni:\n{summary}"
        )

    prompt = (
        f"Aşağıdaki Hugging Face deposunun ({repo_id}) README/içeriğini incele. "
        "Lütfen bu deponun/modülün ne işe yaradığını, ana amacını ve nasıl kullanılacağını Türkçe olarak özetle. "
        "Yanıtın profesyonel, anlaşılır ve kısa (en fazla 250 kelime) olsun.\n\n"
        f"İçerik:\n{readme_content[:10000]}"
    )

    try:
        payload = {
            "model": status["active_model"],
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            ai_response = res.json().get("response", "")
            return f"[🧠 Cortex AI ({status['active_model']}) ile Özetlendi]\n\n{ai_response}"
    except Exception as e:
        print(f"Ollama generation error: {e}")

    return f"Cortex AI ile özet oluşturulurken bir hata oluştu: {status['message']}"

def chat_about_model(repo_id, user_question, readme_content=""):
    """
    Answers user questions about a specific model using Ollama LLM.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Bağlantısı Bulunamadı!\n\n"
            "Lütfen bilgisayarınızda Ollama servisini başlatın (localhost:11434).\n"
            "Örnek çalıştırma: `ollama run llama3`"
        )

    system_context = (
        f"Sen Cortex AI isimli, HuggingFace projeleri konusunda uzman bir yapay zeka asistanısın. "
        f"Kullanıcı şu anda '{repo_id}' isimli HuggingFace modeli hakkında soru soruyor.\n"
        f"Model README özeti / içeriği:\n{readme_content[:4000]}\n\n"
        f"Kullanıcının sorusuna net, yardımcı ve Türkçe olarak cevap ver. "
        f"Eğer bu model kullanıcının istediği işlevi yapamıyorsa, kesinlikle bu işlevi yapan popüler ve alternatif Hugging Face model adlarını (Örn: 'meta-llama/Llama-2-7b', 'runwayml/stable-diffusion-v1-5', 'openai/whisper-large-v3') öner."
    )

    prompt = f"{system_context}\n\nKullanıcının Sorusu: {user_question}"

    try:
        payload = {
            "model": status["active_model"],
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "Yanıt alınamadı.")
    except Exception as e:
        return f"Cortex AI sohbet hatası: {e}"

    return "Cortex AI yanıt veremedi."

def chat_general_ai_assistant(user_prompt, messages_history=None):
    """
    General Cortex AI Chat Assistant for discovering HuggingFace models & answering questions via Ollama.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Servisi Kapalı veya Ulaşılamıyor!\n\n"
            "Lütfen terminalinizden `ollama run llama3` veya `ollama serve` komutuyla Ollama'yı başlatın ve '🔄 Yeniden Bağlan' butonuna basın."
        )

    system_prompt = (
        "Sen Cortex AI adında, ChatGPT kalitesinde çalışan ve Hugging Face platformundaki modelleri, veri kümelerini ve yapay zeka projelerini çok iyi bilen Türkçe rehber yapay zeka asistansın. "
        "Kullanıcının istediği yapay zeka görevlerine uygun Hugging Face model adlarını (Örn: 'stabilityai/stable-diffusion-2-1', 'openai/whisper-large-v3', 'meta-llama/Llama-2-7b-chat-hf') açıkça belirt ve ne işe yaradıklarını Türkçe anlat."
    )

    history_str = ""
    if messages_history:
        for m in messages_history[-6:]:
            sender = "Kullanıcı" if m["sender"] == "user" else "Cortex AI"
            history_str += f"{sender}: {m['text']}\n"

    full_prompt = f"{system_prompt}\n\n{history_str}Kullanıcı: {user_prompt}\nCortex AI:"

    try:
        payload = {
            "model": status["active_model"],
            "prompt": full_prompt,
            "stream": False
        }
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "")
    except Exception as e:
        return f"Cortex AI hatası: {e}"

    return "Yanıt alınamadı."
