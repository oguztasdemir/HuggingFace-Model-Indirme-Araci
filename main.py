import os
import sys
import subprocess

def ensure_dependencies():
    """
    Checks if required dependencies are installed.
    If missing, automatically installs them from requirements.txt via pip.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(req_file):
        return

    required_packages = {
        "customtkinter": "customtkinter",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "huggingface_hub": "huggingface_hub",
        "tkinterweb": "tkinterweb"
    }

    missing_pkgs = []
    for module_name in required_packages.keys():
        try:
            __import__(module_name)
        except ImportError:
            missing_pkgs.append(module_name)

    if missing_pkgs:
        print(f"[HF Downloader] Eksik paketler tespit edildi: {missing_pkgs}")
        print("[HF Downloader] Otomatik kütüphane kurulumu başlatılıyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("[HF Downloader] Tüm kütüphaneler başarıyla yüklendi!")
        except Exception as e:
            print(f"[HF Downloader] Paket kurulumunda uyarı/hata: {e}")

if __name__ == "__main__":
    ensure_dependencies()
    from src.ui.app import HFDownloaderApp
    app = HFDownloaderApp()
    app.mainloop()
