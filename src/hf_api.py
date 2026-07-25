import re
import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi

# Extended Intent Mapping for Natural Language Queries
INTENT_TAG_MAP = {
    # Multimodal / Vision to Video & Image
    "resim çizen": "text-to-image",
    "resim üretme": "text-to-image",
    "fotoğraf üretme": "text-to-image",
    "görselden resim": "text-to-image",
    "text to image": "text-to-image",
    "image to image": "image-to-image",
    "resimden resim": "image-to-image",
    "resmi değiştirme": "image-to-image",
    "text to video": "text-to-video",
    "metinden video": "text-to-video",
    "video üretme": "text-to-video",
    "image to video": "image-to-video",
    "resimden video": "image-to-video",
    "fotoğraftan video": "image-to-video",
    "visual question": "visual-question-answering",
    "görsel soru": "visual-question-answering",
    "image to text": "image-to-text",
    "görsel açıklama": "image-to-text",
    "3d": "text-to-3d",
    # Text / NLP
    "yazı": "text-generation",
    "metin": "text-generation",
    "chat": "text-generation",
    "sohbet": "text-generation",
    "llm": "text-generation",
    "asistan": "text2text-generation",
    "çeviri": "translation",
    "translate": "translation",
    "özet": "summarization",
    "soru": "question-answering",
    "duygu": "text-classification",
    "vektör": "feature-extraction",
    "embedding": "feature-extraction",
    # Vision
    "görüntü sınıflandırma": "image-classification",
    "nesne tespit": "object-detection",
    "segmentasyon": "image-segmentation",
    "derinlik": "depth-estimation",
    # Audio
    "ses tanıma": "automatic-speech-recognition",
    "konuşma tanıma": "automatic-speech-recognition",
    "ses kaydı": "automatic-speech-recognition",
    "whisper": "automatic-speech-recognition",
    "metinden sese": "text-to-speech",
    "ses üretme": "text-to-speech",
    "ses dönüştürme": "audio-to-audio",
    "ses sınıflandırma": "audio-classification",
    # Datasets
    "dataset": "datasets",
    "veri kümesi": "datasets"
}

# ALL Official Hugging Face Task Categories
CATEGORY_OPTIONS = {
    "Tümü (All)": "all",
    
    # Multimodal & Vision-to-X
    "🎨 Metinden Resim (Text-to-Image)": "text-to-image",
    "🖼️ Resimden Resim (Image-to-Image)": "image-to-image",
    "🎬 Metinden Video (Text-to-Video)": "text-to-video",
    "📹 Resimden Video (Image-to-Video)": "image-to-video",
    "💬 Görsel Açıklama / VLM (Image-to-Text)": "image-to-text",
    "👁️ Görsel Soru Yanıtlama (Visual QA)": "visual-question-answering",
    "📦 3D Üretimi (Text-to-3D / Image-to-3D)": "text-to-3d",
    
    # Natural Language Processing (NLP)
    "✍️ Metin Üretimi & LLM (Text Generation)": "text-generation",
    "💬 Sohbet & Metin Dönüştürme (Text2Text)": "text2text-generation",
    "🌍 Çeviri (Translation)": "translation",
    "📝 Özetleme (Summarization)": "summarization",
    "❓ Soru Yanıtlama (Question Answering)": "question-answering",
    "📊 Duygu & Metin Sınıflandırma (Text Classification)": "text-classification",
    "🏷️ Varlık Tanıma (Token Classification)": "token-classification",
    "🔍 Metin Vektörleştirme (Feature Extraction)": "feature-extraction",
    
    # Computer Vision
    "🖼️ Görüntü Sınıflandırma (Image Classification)": "image-classification",
    "🎯 Nesne Tespiti (Object Detection)": "object-detection",
    "✂️ Görüntü Segmentasyonu (Image Segmentation)": "image-segmentation",
    "📐 Derinlik Tahmini (Depth Estimation)": "depth-estimation",
    
    # Audio & Speech
    "🎙️ Ses / Konuşma Tanıma (Speech Recognition)": "automatic-speech-recognition",
    "🔊 Metinden Sese (Text-to-Speech)": "text-to-speech",
    "🎵 Ses Üretimi & Dönüştürme (Audio-to-Audio)": "audio-to-audio",
    "🎧 Ses Sınıflandırma (Audio Classification)": "audio-classification",
    
    # Tabular & RL & Datasets
    "📈 Tablosal Veri Sınıflandırma (Tabular)": "tabular-classification",
    "🤖 Pekiştirmeli Öğrenme (Reinforcement Learning)": "reinforcement-learning",
    "📦 Popüler Datasets": "datasets"
}

def parse_hf_url(url):
    """
    Parses Hugging Face URL.
    Returns (repo_id, repo_type) where repo_type is 'model' or 'dataset'.
    Returns (None, None) if parsing fails.
    """
    url = url.strip().rstrip('/')
    url_clean = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", url)

    if url_clean.startswith("datasets/"):
        repo_id = url_clean[len("datasets/"):]
        return repo_id, "dataset"

    if url_clean.startswith("models/"):
        repo_id = url_clean[len("models/"):]
        return repo_id, "model"

    parts = url_clean.split('/')
    if parts[0] in ["spaces", "posts", "docs", "pricing", "blog"]:
        return None, None

    if len(parts) <= 2:
        return url_clean, "model"

    return None, None

def get_repo_details(repo_id, repo_type="model", token=None):
    """
    Fetches the list of files and metadata for the given repo using optional HF token.
    """
    api = HfApi(token=token if token else None)
    files = []
    total_size = 0
    try:
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, files_metadata=True, token=token if token else None)
        else:
            info = api.model_info(repo_id, files_metadata=True, token=token if token else None)

        if info.siblings:
            for sibling in info.siblings:
                size = getattr(sibling, "size", 0) or 0
                files.append({
                    "path": sibling.rfilename,
                    "size": size
                })
                total_size += size
        return {
            "success": True,
            "files": files,
            "total_size": total_size,
            "author": getattr(info, "author", "Unknown"),
            "downloads": getattr(info, "downloads", 0),
            "likes": getattr(info, "likes", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def scrape_readme(repo_id, repo_type="model", token=None):
    """
    Scrapes the README of the repo using optional token headers.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}/raw/main/README.md"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.text
        url_master = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}/raw/master/README.md"
        res_master = requests.get(url_master, headers=headers, timeout=10)
        if res_master.status_code == 200:
            return res_master.text
    except Exception:
        pass

    try:
        web_url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
        res = requests.get(web_url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)[:2000]
    except Exception:
        pass

    return ""

def search_hf_hub(query="", filter_tag=None, sort_by="downloads", limit=30, token=None):
    """
    Searches models/datasets on Hugging Face Hub using text query, category filters, and optional token.
    Supports intent matching for natural Turkish/English queries.
    """
    api = HfApi(token=token if token else None)
    results = []

    detected_tag = filter_tag
    if not detected_tag or detected_tag == "all":
        query_lower = query.lower()
        for kw, tag in INTENT_TAG_MAP.items():
            if kw in query_lower:
                detected_tag = tag
                break

    try:
        if detected_tag == "datasets":
            datasets = api.list_datasets(search=query if query else None, sort=sort_by, limit=limit)
            for item in datasets:
                author = getattr(item, "author", None) or (item.id.split("/")[0] if "/" in item.id else "Unknown")
                results.append({
                    "repo_id": item.id,
                    "repo_type": "dataset",
                    "author": author,
                    "downloads": getattr(item, "downloads", 0) or 0,
                    "likes": getattr(item, "likes", 0) or 0,
                    "pipeline_tag": "dataset",
                    "last_modified": str(getattr(item, "lastModified", ""))[:10]
                })
        else:
            kwargs = {"sort": sort_by, "limit": limit}
            if query:
                kwargs["search"] = query
            if detected_tag and detected_tag not in ["all", "trending"]:
                kwargs["filter"] = detected_tag

            models = api.list_models(**kwargs)
            for item in models:
                author = getattr(item, "author", None) or (item.id.split("/")[0] if "/" in item.id else "Unknown")
                results.append({
                    "repo_id": item.id,
                    "repo_type": "model",
                    "author": author,
                    "downloads": getattr(item, "downloads", 0) or 0,
                    "likes": getattr(item, "likes", 0) or 0,
                    "pipeline_tag": getattr(item, "pipeline_tag", "model") or "model",
                    "last_modified": str(getattr(item, "lastModified", ""))[:10]
                })
    except Exception as e:
        print(f"Error searching Hugging Face Hub: {e}")

    return results
