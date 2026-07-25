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

def generate_summary(readme_content, repo_id, lang="tr"):
    """
    Generates a summary of the model/dataset using local Ollama LLM.
    """
    if not readme_content:
        return "README içeriği alınamadı. Özet oluşturulamıyor." if lang == "tr" else "No README content found."

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

        if lang == "en":
            return f"💡 [Ollama / Cortex Engine Not Running]\nStart Ollama (http://localhost:11434) for AI summaries.\n\n📄 Raw README:\n{summary}"
        return (
            "💡 [Ollama / Cortex Engine Çalışmıyor]\n"
            "Yapay zeka özetleri için bilgisayarınızda Ollama'yı (http://localhost:11434) başlatın.\n\n"
            f"📄 Ham README Metni:\n{summary}"
        )

    lang_instruction = "Türkçe olarak özetle" if lang == "tr" else "Summarize in English"
    prompt = (
        f"Depo: {repo_id}\n"
        f"Lütfen bu modelin/deponun ne işe yaradığını, ana amacını ve özelliklerini {lang_instruction}.\n\n"
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
            return f"[🧠 Cortex AI ({status['active_model']})]\n\n{ai_response}"
    except Exception as e:
        print(f"Ollama generation error: {e}")

    return f"Cortex AI error: {status['message']}"

def analyze_vram_hardware(readme_content, repo_id, lang="tr"):
    """
    Analyzes minimum/recommended vRAM and hardware specs for the given model.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Bağlantısı Bulunamadı!\nvRAM ve donanım analizi için Ollama servisinin açık olması gereklidir."
            if lang == "tr" else
            "🔴 Cortex Engine (Ollama) Not Connected!\nOllama service must be running for hardware & vRAM estimation."
        )

    lang_instr = "Türkçe yanıt ver" if lang == "tr" else "Respond in English"
    prompt = (
        f"Depo: {repo_id}\n\n"
        f"Aşağıdaki modelin README/yapısını inceleyerek bu modelin çalışması için gereken DONANIM VE vRAM GEREKSİNİMLERİNİ analiz et.\n"
        f"1. Minimum GPU vRAM (Örn: 6 GB / 8 GB / 16 GB / 24 GB vRAM)\n"
        f"2. Önerilen Sistem RAM'i ve İşlemci Gücü\n"
        f"3. Uyumlu Çalıştırma Arayüzleri (Örn: ComfyUI, WebUI, Diffusers, Ollama, PyTorch)\n\n"
        f"{lang_instr}. Kısa, net ve maddeler halinde yaz.\n\n"
        f"İçerik:\n{readme_content[:10000]}"
    )

    try:
        payload = {"model": status["active_model"], "prompt": prompt, "stream": False}
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return f"💻 [Cortex AI Donanım & vRAM Analizi]\n\n{res.json().get('response', '')}"
    except Exception as e:
        return f"vRAM analizi hatası: {e}"

    return "vRAM analizi gerçekleştirilemedi."

def generate_usage_code(readme_content, repo_id, lang="tr"):
    """
    Generates ready-to-run Python code snippet and sample prompts for using the model.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Bağlantısı Bulunamadı!\nÖrnek Python kodları üretmek için Ollama servisini çalıştırın."
            if lang == "tr" else
            "🔴 Cortex Engine (Ollama) Not Connected!\nRun Ollama service to generate Python usage code."
        )

    lang_instr = "Açıklamaları Türkçe yaz" if lang == "tr" else "Write explanations in English"
    prompt = (
        f"Depo: {repo_id}\n\n"
        f"Aşağıdaki model için çalıştırmaya hazır Python kodu (diffusers, transformers veya huggingface_hub kullanarak) ve 2 adet örnek prompt üret.\n"
        f"Kod tam, çalışır durumda ve anlaşılır olsun.\n{lang_instr}.\n\n"
        f"İçerik:\n{readme_content[:10000]}"
    )

    try:
        payload = {"model": status["active_model"], "prompt": prompt, "stream": False}
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return f"📜 [Cortex AI Kullanım Kodu & Prompt Örnekleri]\n\n{res.json().get('response', '')}"
    except Exception as e:
        return f"Kod üretme hatası: {e}"

    return "Kullanım kodu üretilemedi."

def chat_about_model(repo_id, user_question, readme_content="", lang="tr"):
    """
    Answers user questions about a specific model using Ollama LLM.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Bağlantısı Bulunamadı!\n\nLütfen bilgisayarınızda Ollama servisini başlatın (localhost:11434)."
            if lang == "tr" else
            "🔴 Cortex Engine (Ollama) Not Connected!\nPlease start Ollama service on your machine."
        )

    system_context = (
        f"Sen Cortex AI isimli, HuggingFace projeleri konusunda uzman bir yapay zeka asistanısın. "
        f"Kullanıcı şu anda '{repo_id}' isimli HuggingFace modeli hakkında soru soruyor.\n"
        f"Model README özeti:\n{readme_content[:4000]}\n\n"
        f"Kullanıcının sorusuna net ve {'Türkçe' if lang == 'tr' else 'İngilizce'} yanıt ver."
    )

    prompt = f"{system_context}\n\nKullanıcının Sorusu: {user_question}"

    try:
        payload = {"model": status["active_model"], "prompt": prompt, "stream": False}
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "Yanıt alınamadı.")
    except Exception as e:
        return f"Cortex AI sohbet hatası: {e}"

    return "Cortex AI yanıt veremedi."

def chat_general_ai_assistant(user_prompt, messages_history=None, lang="tr"):
    """
    General Cortex AI Chat Assistant for discovering HuggingFace models & answering questions via Ollama.
    """
    status = check_ollama_status()
    if not status["connected"] or not status["active_model"]:
        return (
            "🔴 Cortex Engine (Ollama) Servisi Kapalı veya Ulaşılamıyor!\nLütfen terminalinizden Ollama'yı başlatın."
            if lang == "tr" else
            "🔴 Cortex Engine (Ollama) Service is Offline!\nPlease start Ollama service from your terminal."
        )

    system_prompt = (
        f"Sen Cortex AI adında, Hugging Face platformundaki modelleri, veri kümelerini çok iyi bilen {'Türkçe' if lang == 'tr' else 'İngilizce'} rehber yapay zeka asistansın. "
        f"Kullanıcının istediği yapay zeka görevlerine uygun Hugging Face model adlarını (Örn: 'stabilityai/stable-diffusion-2-1', 'openai/whisper-large-v3') açıkça belirt."
    )

    history_str = ""
    if messages_history:
        for m in messages_history[-6:]:
            sender = "Kullanıcı" if m["sender"] == "user" else "Cortex AI"
            history_str += f"{sender}: {m['text']}\n"

    full_prompt = f"{system_prompt}\n\n{history_str}Kullanıcı: {user_prompt}\nCortex AI:"

    try:
        payload = {"model": status["active_model"], "prompt": full_prompt, "stream": False}
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "")
    except Exception as e:
        return f"Cortex AI hatası: {e}"

    return "Yanıt alınamadı."
