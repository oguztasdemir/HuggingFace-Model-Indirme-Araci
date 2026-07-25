import os
import re
import json
import time
import threading
import webbrowser
from tkinter import filedialog, messagebox
import customtkinter as ctk
from huggingface_hub import hf_hub_download, snapshot_download

from src.config import (
    load_settings, save_settings, APP_TITLE, DEFAULT_WINDOW_SIZE, MIN_WINDOW_SIZE,
    load_favorites, save_favorites, is_favorite, add_to_favorites, remove_from_favorites, update_favorite_note,
    load_chat_sessions, save_chat_sessions, create_chat_session, add_chat_message, delete_chat_session, CHAT_HISTORY_FILE
)
from src import storage
from src import hf_api
from src import ai_summarizer

# Appearance Mode & Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def clean_markdown_text(text):
    """
    Converts raw markdown text (**bold**, `code`, etc.) into cleanly formatted, highly readable display text with emoji markers.
    """
    if not text:
        return ""
    # Convert **bold** to clean text
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Convert `code` to clean text
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
    # Convert markdown headers # ## ###
    cleaned = re.sub(r'^\s*#{1,6}\s*', r'📌 ', cleaned, flags=re.MULTILINE)
    # Convert numbered lists 1. 2. 3.
    cleaned = re.sub(r'^\s*(\d+)\.\s*', r'\n🔹 \1. ', cleaned, flags=re.MULTILINE)
    # Convert bullet points - or *
    cleaned = re.sub(r'^\s*[\-\*]\s*', r'  • ', cleaned, flags=re.MULTILINE)
    # Remove excess blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

class HFDownloaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)

        # Load settings
        self.settings = load_settings()

        # Scan drives for HF path
        scan_results = storage.scan_huggingface_dirs()
        self.detected_drives = scan_results["drives"]
        if not self.settings.get("download_dir"):
            self.settings["download_dir"] = scan_results["default_path"]
            save_settings(self.settings)

        # Current state variables
        self.current_repo_id = None
        self.current_repo_type = None
        self.repo_files = []
        self.total_repo_size_bytes = 0
        self.downloaded_bytes_total = 0
        self.current_readme_content = ""
        self.checkbox_vars = {}
        self.checkbox_widgets = []
        self.local_models_data = []
        self.active_chat_session_id = None
        self.is_thinking = False
        self.thinking_seconds = 0.0
        self.thinking_bubble = None

        # Configure root grid
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Build Sidebar & Main Content
        self.build_sidebar()
        self.build_main_content()

        # Default page
        self.show_page("search")

        # Initial background tasks
        self.after(500, self.refresh_local_models_thread)
        self.after(600, self.refresh_sidebar_disk_info)

    # ==========================================
    # LEFT SIDEBAR PANEL BUILDER
    # ==========================================
    def build_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray15"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="🤗 HF Downloader", 
            font=ctk.CTkFont(family="Inter", size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 20), sticky="w")

        # Nav 1: Model Search & Download
        self.btn_nav_search = ctk.CTkButton(
            self.sidebar_frame, 
            text="🔍 Model İncele & İndir", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("search")
        )
        self.btn_nav_search.grid(row=1, column=0, padx=15, pady=4, sticky="ew")

        # Nav 2: Favorites & Notes
        self.btn_nav_fav = ctk.CTkButton(
            self.sidebar_frame, 
            text="⭐ Favorilerim & Notlarım", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("favorites")
        )
        self.btn_nav_fav.grid(row=2, column=0, padx=15, pady=4, sticky="ew")

        # Nav 3: Cortex AI Assistant (ChatGPT Style + Chat Backup)
        self.btn_nav_ai = ctk.CTkButton(
            self.sidebar_frame, 
            text="🧠 Cortex AI Asistanı", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("ai_chat")
        )
        self.btn_nav_ai.grid(row=3, column=0, padx=15, pady=4, sticky="ew")

        # Nav 4: Explore Catalog & Semantic Search
        self.btn_nav_explore = ctk.CTkButton(
            self.sidebar_frame, 
            text="🌐 HF Katalog & Keşfet", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("explore")
        )
        self.btn_nav_explore.grid(row=4, column=0, padx=15, pady=4, sticky="ew")

        # Nav 5: Local Models
        self.btn_nav_local = ctk.CTkButton(
            self.sidebar_frame, 
            text="📦 Mevcut Modellerim", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("local")
        )
        self.btn_nav_local.grid(row=5, column=0, padx=15, pady=4, sticky="ew")

        # Nav 6: Settings
        self.btn_nav_settings = ctk.CTkButton(
            self.sidebar_frame, 
            text="⚙️ Sürücü & Ayarlar", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("settings")
        )
        self.btn_nav_settings.grid(row=6, column=0, padx=15, pady=4, sticky="ew")

        self.sidebar_divider = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray75", "gray25"))
        self.sidebar_divider.grid(row=7, column=0, padx=15, pady=15, sticky="ew")

        # LIVE DISK STORAGE PANEL
        self.sidebar_disk_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=10, fg_color=("gray80", "gray20"))
        self.sidebar_disk_card.grid(row=9, column=0, padx=15, pady=(0, 20), sticky="ew")
        self.sidebar_disk_card.grid_columnconfigure(0, weight=1)

        self.sb_disk_title = ctk.CTkLabel(
            self.sidebar_disk_card, 
            text="💾 Hedef Sürücü Durumu", 
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.sb_disk_title.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self.sb_drive_path_lbl = ctk.CTkLabel(
            self.sidebar_disk_card,
            text="Konum: Yükleniyor...",
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        )
        self.sb_drive_path_lbl.grid(row=1, column=0, padx=12, pady=0, sticky="w")

        self.sb_disk_progress = ctk.CTkProgressBar(self.sidebar_disk_card, height=12)
        self.sb_disk_progress.grid(row=2, column=0, padx=12, pady=8, sticky="ew")
        self.sb_disk_progress.set(0)

        self.sb_disk_details_lbl = ctk.CTkLabel(
            self.sidebar_disk_card,
            text="Boş: - GB / Toplam: - GB\nHF Önbelleği: -",
            font=ctk.CTkFont(size=11),
            justify="left",
            anchor="w"
        )
        self.sb_disk_details_lbl.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="w")

        self.sb_refresh_btn = ctk.CTkButton(
            self.sidebar_disk_card,
            text="🔄 Disk Alanını Güncelle",
            font=ctk.CTkFont(size=11),
            height=26,
            command=self.refresh_sidebar_disk_info
        )
        self.sb_refresh_btn.grid(row=4, column=0, padx=12, pady=(0, 10), sticky="ew")

    # ==========================================
    # RIGHT MAIN CONTENT BUILDER
    # ==========================================
    def build_main_content(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.page_search = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_favorites = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_ai_chat = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_explore = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_local = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_settings = ctk.CTkFrame(self.main_container, fg_color="transparent")

        self.build_page_search()
        self.build_page_favorites()
        self.build_page_ai_chat()
        self.build_page_explore()
        self.build_page_local()
        self.build_page_settings()

    def show_page(self, page_name):
        self.page_search.grid_forget()
        self.page_favorites.grid_forget()
        self.page_ai_chat.grid_forget()
        self.page_explore.grid_forget()
        self.page_local.grid_forget()
        self.page_settings.grid_forget()

        active_color = ("gray75", "gray30")
        inactive_color = "transparent"

        self.btn_nav_search.configure(fg_color=inactive_color)
        self.btn_nav_fav.configure(fg_color=inactive_color)
        self.btn_nav_ai.configure(fg_color=inactive_color)
        self.btn_nav_explore.configure(fg_color=inactive_color)
        self.btn_nav_local.configure(fg_color=inactive_color)
        self.btn_nav_settings.configure(fg_color=inactive_color)

        if page_name == "search":
            self.page_search.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_search.configure(fg_color=active_color)
        elif page_name == "favorites":
            self.page_favorites.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_fav.configure(fg_color=active_color)
            self.refresh_favorites_ui()
        elif page_name == "ai_chat":
            self.page_ai_chat.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_ai.configure(fg_color=active_color)
            self.check_ai_status_ui()
            self.refresh_chat_sessions_list_ui()
        elif page_name == "explore":
            self.page_explore.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_explore.configure(fg_color=active_color)
            if not getattr(self, "_explore_loaded", False):
                self.start_explore_search_thread()
        elif page_name == "local":
            self.page_local.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_local.configure(fg_color=active_color)
            self.refresh_local_models_thread()
        elif page_name == "settings":
            self.page_settings.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_settings.configure(fg_color=active_color)

    # ==========================================
    # PAGE 1: SEARCH, DOWNLOAD & MODEL QA CHAT
    # ==========================================
    def build_page_search(self):
        self.page_search.grid_columnconfigure(0, weight=1)
        self.page_search.grid_rowconfigure(2, weight=1)

        # TOP BAR
        self.top_web_bar = ctk.CTkFrame(self.page_search, corner_radius=8, fg_color=("gray85", "gray20"))
        self.top_web_bar.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")
        self.top_web_bar.grid_columnconfigure(0, weight=1)

        self.btn_open_external_hf = ctk.CTkButton(
            self.top_web_bar,
            text="🌐 Hugging Face'i Tarayıcıda Aç (huggingface.co)",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=34,
            fg_color=("gray75", "gray30"),
            hover_color=("gray65", "gray40"),
            command=lambda: webbrowser.open("https://huggingface.co/models")
        )
        self.btn_open_external_hf.grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.web_hint_lbl = ctk.CTkLabel(
            self.top_web_bar,
            text="Tarayıcıda beğendiğiniz modelin linkini kopyalayıp aşağıdaki kutuya yapıştırın.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray70"
        )
        self.web_hint_lbl.grid(row=0, column=1, padx=(0, 12), pady=8, sticky="e")

        # URL Input Frame
        self.url_frame = ctk.CTkFrame(self.page_search, corner_radius=8)
        self.url_frame.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.url_frame.grid_columnconfigure(1, weight=1)

        self.url_lbl = ctk.CTkLabel(self.url_frame, text="Hugging Face Linki / Repo:", font=ctk.CTkFont(weight="bold"))
        self.url_lbl.grid(row=0, column=0, padx=(15, 10), pady=12)

        self.url_entry = ctk.CTkEntry(
            self.url_frame, 
            placeholder_text="Örn: https://huggingface.co/gpt2 veya meta-llama/Llama-2-7b veya datasets/imdb",
            height=36
        )
        self.url_entry.grid(row=0, column=1, padx=5, pady=12, sticky="ew")

        self.analyze_btn = ctk.CTkButton(
            self.url_frame, 
            text="🔍 Verileri Getir", 
            width=130, 
            height=36,
            font=ctk.CTkFont(weight="bold"),
            command=self.start_fetch_data_thread
        )
        self.analyze_btn.grid(row=0, column=2, padx=(5, 15), pady=12)

        # Content split
        self.search_body = ctk.CTkFrame(self.page_search, fg_color="transparent")
        self.search_body.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        self.search_body.grid_columnconfigure(0, weight=3)
        self.search_body.grid_columnconfigure(1, weight=2)
        self.search_body.grid_rowconfigure(0, weight=1)

        # LEFT: Files list & Smart Selection Controls
        self.files_frame = ctk.CTkFrame(self.search_body, corner_radius=8)
        self.files_frame.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        self.files_frame.grid_columnconfigure(0, weight=1)
        self.files_frame.grid_rowconfigure(3, weight=1)

        self.files_hdr = ctk.CTkFrame(self.files_frame, fg_color="transparent")
        self.files_hdr.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        self.files_hdr.grid_columnconfigure(0, weight=1)

        self.files_title = ctk.CTkLabel(
            self.files_hdr, 
            text="📂 Depo Dosyaları (Seçilen: 0 B / Toplam Depo: 0 B)", 
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.files_title.grid(row=0, column=0, sticky="w")

        # Toolbar for smart selection & deduplication
        self.toolbar_frame = ctk.CTkFrame(self.files_frame, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, padx=10, pady=(0, 4), sticky="ew")
        self.toolbar_frame.grid_columnconfigure(0, weight=1)

        self.smart_select_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="🎯 Önerilen Modeli Seç",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#27AE60",
            hover_color="#1E8449",
            height=28,
            command=self.select_smart_defaults
        )
        self.smart_select_btn.pack(side="left", padx=(0, 4))

        self.skip_downloaded_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="⚡ Yüklü Olanları Atla",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2980B9",
            hover_color="#1F618D",
            height=28,
            command=self.skip_downloaded_files
        )
        self.skip_downloaded_btn.pack(side="left", padx=4)

        self.select_all_var = ctk.StringVar(value="off")
        self.select_all_cb = ctk.CTkCheckBox(
            self.toolbar_frame, 
            text="Tümünü Seç", 
            variable=self.select_all_var, 
            onvalue="on", 
            offvalue="off",
            font=ctk.CTkFont(size=11),
            command=self.toggle_select_all
        )
        self.select_all_cb.pack(side="left", padx=4)

        self.clear_select_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="🧹 Temizle",
            font=ctk.CTkFont(size=11),
            fg_color=("gray75", "gray30"),
            height=28,
            width=65,
            command=self.clear_all_selections
        )
        self.clear_select_btn.pack(side="left", padx=4)

        self.file_search_entry = ctk.CTkEntry(self.files_frame, placeholder_text="Dosya ismine göre ara...", height=28)
        self.file_search_entry.grid(row=2, column=0, padx=10, pady=4, sticky="ew")
        self.file_search_entry.bind("<KeyRelease>", self.filter_files)

        self.files_scroll_frame = ctk.CTkScrollableFrame(self.files_frame, fg_color="transparent")
        self.files_scroll_frame.grid(row=3, column=0, padx=10, pady=4, sticky="nsew")

        self.download_selected_btn = ctk.CTkButton(
            self.files_frame,
            text="📥 Seçilen Dosyaları İndir",
            height=38,
            command=self.start_download_selected_thread,
            state="disabled",
            font=ctk.CTkFont(weight="bold")
        )
        self.download_selected_btn.grid(row=4, column=0, padx=12, pady=10, sticky="ew")

        # RIGHT: One Click Download, AI Summary & Interactive Model Chat
        self.info_pane = ctk.CTkFrame(self.search_body, corner_radius=8)
        self.info_pane.grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")
        self.info_pane.grid_columnconfigure(0, weight=1)
        self.info_pane.grid_rowconfigure(3, weight=1)

        # One Click Download Card
        self.one_click_frame = ctk.CTkFrame(self.info_pane, fg_color=("gray80", "gray22"), corner_radius=8)
        self.one_click_frame.grid(row=0, column=0, padx=12, pady=10, sticky="ew")
        self.one_click_frame.grid_columnconfigure(0, weight=1)

        self.one_click_btn = ctk.CTkButton(
            self.one_click_frame,
            text="⚡ Tek Tıkla Tüm Modeli İndir",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#27AE60",
            hover_color="#1E8449",
            height=42,
            command=self.start_one_click_download_thread,
            state="disabled"
        )
        self.one_click_btn.grid(row=0, column=0, padx=12, pady=10, sticky="ew")

        self.one_click_hint = ctk.CTkLabel(
            self.one_click_frame, 
            text="Depodaki tüm sürümleri ve dosyaları otomatik olarak tek tıkla indirir.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray70"
        )
        self.one_click_hint.grid(row=1, column=0, padx=12, pady=(0, 8))

        # Metadata & Favorite Card
        self.meta_card = ctk.CTkFrame(self.info_pane, fg_color="transparent")
        self.meta_card.grid(row=1, column=0, padx=12, pady=2, sticky="ew")

        self.meta_info_label = ctk.CTkLabel(
            self.meta_card,
            text="📌 Lütfen bir Hugging Face linki girip '🔍 Verileri Getir' butonuna basın.",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=13)
        )
        self.meta_info_label.pack(side="left", fill="x", expand=True, pady=2)

        self.star_model_btn = ctk.CTkButton(
            self.meta_card,
            text="⭐ Favorilere Ekle",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=28,
            width=110,
            fg_color="#D4AC0D",
            hover_color="#B7950B",
            state="disabled",
            command=self.open_add_favorite_dialog_current
        )
        self.star_model_btn.pack(side="right", padx=5, pady=2)

        # AI Summary Textbox & Optional Button Header
        self.summary_hdr_frame = ctk.CTkFrame(self.info_pane, fg_color="transparent")
        self.summary_hdr_frame.grid(row=2, column=0, padx=12, pady=(6, 2), sticky="ew")
        self.summary_hdr_frame.grid_columnconfigure(0, weight=1)

        self.summary_title = ctk.CTkLabel(
            self.summary_hdr_frame, 
            text="🤖 AI & Model İşlev Özeti", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.summary_title.pack(side="left")

        self.btn_run_ai_analysis = ctk.CTkButton(
            self.summary_hdr_frame,
            text="🤖 Cortex AI ile Modeli Analiz Et",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=28,
            fg_color="#2980B9",
            hover_color="#1F618D",
            state="disabled",
            command=self.start_manual_ai_analysis_thread
        )
        self.btn_run_ai_analysis.pack(side="right", padx=5)

        self.summary_text = ctk.CTkTextbox(self.info_pane, wrap="word", font=ctk.CTkFont(size=13), height=140)
        self.summary_text.grid(row=3, column=0, padx=12, pady=(2, 6), sticky="nsew")
        self.summary_text.insert("0.0", "Hugging Face model verileri getirildikten sonra isteğe bağlı olarak '🤖 Cortex AI ile Modeli Analiz Et' butonuna basarak özet alabilirsiniz.")
        self.summary_text.configure(state="disabled")

        # MODEL SPECIFIC QA CHAT
        self.model_qa_frame = ctk.CTkFrame(self.info_pane, corner_radius=8, fg_color=("gray85", "gray20"))
        self.model_qa_frame.grid(row=4, column=0, padx=12, pady=(4, 10), sticky="ew")
        self.model_qa_frame.grid_columnconfigure(0, weight=1)

        self.qa_title = ctk.CTkLabel(
            self.model_qa_frame, 
            text="💬 Bu Model Hakkında Cortex AI'ya Soru Sor", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.qa_title.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        self.qa_entry = ctk.CTkEntry(
            self.model_qa_frame, 
            placeholder_text="Örn: Bu model Türkçe yapabiliyor mu? Yapmıyorsa ne önerirsin?",
            height=30
        )
        self.qa_entry.grid(row=1, column=0, padx=(10, 4), pady=(0, 8), sticky="ew")
        self.qa_entry.bind("<Return>", lambda e: self.send_model_qa_chat())

        self.qa_btn = ctk.CTkButton(
            self.model_qa_frame, 
            text="Sor", 
            width=65, 
            height=30, 
            font=ctk.CTkFont(weight="bold"),
            command=self.send_model_qa_chat
        )
        self.qa_btn.grid(row=1, column=1, padx=(0, 10), pady=(0, 8))

        # Bottom Progress Bar
        self.bottom_status_frame = ctk.CTkFrame(self.page_search, corner_radius=8)
        self.bottom_status_frame.grid(row=3, column=0, padx=0, pady=(10, 0), sticky="ew")
        self.bottom_status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.bottom_status_frame, text="Sistem Hazır", font=ctk.CTkFont(size=12, weight="bold"))
        self.status_label.grid(row=0, column=0, padx=12, pady=(6, 2), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.bottom_status_frame, height=10)
        self.progress_bar.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        self.progress_bar.set(0)

    # ==========================================
    # PAGE 2: FAVORITES & NOTES
    # ==========================================
    def build_page_favorites(self):
        self.page_favorites.grid_columnconfigure(0, weight=1)
        self.page_favorites.grid_rowconfigure(2, weight=1)

        self.fav_header = ctk.CTkFrame(self.page_favorites, corner_radius=8, fg_color=("gray85", "gray20"))
        self.fav_header.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")

        self.fav_title_lbl = ctk.CTkLabel(
            self.fav_header,
            text="⭐ Favori Modellerim & Kişisel Notlarım",
            font=ctk.CTkFont(size=17, weight="bold")
        )
        self.fav_title_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.fav_sub_lbl = ctk.CTkLabel(
            self.fav_header,
            text="Beğendiğiniz projeleri notlarınızla birlikte kaydedin ve tek tıkla inceleyip indirin.",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.fav_sub_lbl.pack(anchor="w", padx=15, pady=(0, 10))

        self.fav_controls = ctk.CTkFrame(self.page_favorites, corner_radius=8)
        self.fav_controls.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.fav_controls.grid_columnconfigure(0, weight=2)
        self.fav_controls.grid_columnconfigure(1, weight=3)

        self.fav_add_url_entry = ctk.CTkEntry(
            self.fav_controls,
            placeholder_text="HuggingFace URL veya Repo ID (Örn: meta-llama/Llama-2-7b)...",
            height=34
        )
        self.fav_add_url_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")

        self.fav_add_note_entry = ctk.CTkEntry(
            self.fav_controls,
            placeholder_text="Kişisel Notunuz (Örn: Türkçe çeviri için harika, SDXL uyumlu)...",
            height=34
        )
        self.fav_add_note_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.fav_add_btn = ctk.CTkButton(
            self.fav_controls,
            text="➕ Favorilere Ekle",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#D4AC0D",
            hover_color="#B7950B",
            height=34,
            command=self.add_favorite_manual_gui
        )
        self.fav_add_btn.grid(row=0, column=2, padx=(5, 12), pady=10)

        self.fav_scroll_frame = ctk.CTkScrollableFrame(self.page_favorites, fg_color="transparent")
        self.fav_scroll_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        self.fav_scroll_frame.grid_columnconfigure(0, weight=1)

    def refresh_favorites_ui(self):
        for widget in self.fav_scroll_frame.winfo_children():
            widget.destroy()

        favs = load_favorites()

        if not favs:
            lbl = ctk.CTkLabel(
                self.fav_scroll_frame,
                text="Henüz favorilere eklenmiş bir Hugging Face modeli yok.\nArama yaparken ⭐ butonuna basarak veya yukarıdaki kutuyu kullanarak ekleyebilirsiniz.",
                font=ctk.CTkFont(size=14),
                text_color="gray60"
            )
            lbl.pack(pady=40)
            return

        for item in favs:
            card = ctk.CTkFrame(self.fav_scroll_frame, corner_radius=8, fg_color=("gray85", "gray20"))
            card.pack(fill="x", padx=5, pady=6)
            card.grid_columnconfigure(0, weight=1)

            repo_id = item["repo_id"]
            repo_type = item.get("repo_type", "model").upper()
            note = item.get("note", "")
            added_at = item.get("added_at", "")

            hdr_txt = f"⭐ {repo_id}   [{repo_type}]  •  Eklendi: {added_at}"
            title_lbl = ctk.CTkLabel(card, text=hdr_txt, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            title_lbl.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

            note_frame = ctk.CTkFrame(card, fg_color="transparent")
            note_frame.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="ew")
            note_frame.grid_columnconfigure(1, weight=1)

            note_lbl = ctk.CTkLabel(note_frame, text="📝 Not:", font=ctk.CTkFont(size=12, weight="bold"))
            note_lbl.grid(row=0, column=0, padx=(0, 6), sticky="w")

            note_entry = ctk.CTkEntry(note_frame, height=28)
            note_entry.insert(0, note)
            note_entry.grid(row=0, column=1, padx=(0, 6), sticky="ew")

            save_note_btn = ctk.CTkButton(
                note_frame,
                text="💾 Notu Güncelle",
                font=ctk.CTkFont(size=11),
                height=28,
                width=95,
                command=lambda r=repo_id, e=note_entry: self.save_updated_note(r, e.get())
            )
            save_note_btn.grid(row=0, column=2, sticky="e")

            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            inspect_btn = ctk.CTkButton(
                btn_box,
                text="🔍 Analiz Et & İndir",
                font=ctk.CTkFont(weight="bold"),
                height=34,
                command=lambda r=repo_id, t=item.get("repo_type", "model"): self.inspect_from_explore(r, t)
            )
            inspect_btn.pack(side="left", padx=4)

            web_btn = ctk.CTkButton(
                btn_box,
                text="🔗 HF'te Aç",
                height=34,
                width=80,
                fg_color=("gray70", "gray30"),
                command=lambda u=item.get("url", f"https://huggingface.co/{repo_id}"): webbrowser.open(u)
            )
            web_btn.pack(side="left", padx=4)

            remove_btn = ctk.CTkButton(
                btn_box,
                text="❌ Çıkar",
                height=34,
                width=75,
                fg_color="#C0392B",
                hover_color="#922B21",
                command=lambda r=repo_id: self.remove_favorite_action(r)
            )
            remove_btn.pack(side="left", padx=4)

    def add_favorite_manual_gui(self):
        url = self.fav_add_url_entry.get().strip()
        note = self.fav_add_note_entry.get().strip()
        if not url:
            messagebox.showwarning("Eksik Bilgi", "Lütfen bir HuggingFace URL'si veya Repo ID girin.")
            return

        repo_id, repo_type = hf_api.parse_hf_url(url)
        if not repo_id:
            repo_id = url
            repo_type = "model"

        add_to_favorites(repo_id, repo_type, note)
        self.fav_add_url_entry.delete(0, "end")
        self.fav_add_note_entry.delete(0, "end")
        messagebox.showinfo("Favorilere Eklendi", f"'{repo_id}' favorilerinize eklendi!")
        self.refresh_favorites_ui()

    def open_add_favorite_dialog_current(self):
        if not self.current_repo_id:
            return
        
        dialog = ctk.CTkInputDialog(text=f"'{self.current_repo_id}' için bir not yazın:", title="⭐ Favoriye Not Ekle")
        user_note = dialog.get_input()
        if user_note is not None:
            add_to_favorites(self.current_repo_id, self.current_repo_type, user_note.strip())
            messagebox.showinfo("Başarılı", f"'{self.current_repo_id}' favorilerinize eklendi!")
            if is_favorite(self.current_repo_id):
                self.star_model_btn.configure(text="⭐ Favorilerde", fg_color="#B7950B")

    def save_updated_note(self, repo_id, new_note):
        update_favorite_note(repo_id, new_note.strip())
        messagebox.showinfo("Güncellendi", "Notunuz başarıyla kaydedildi!")

    def remove_favorite_action(self, repo_id):
        remove_from_favorites(repo_id)
        self.refresh_favorites_ui()

    # ==========================================
    # PAGE 3: CORTEX AI ASSISTANT (CHATGPT STYLE + CHAT BACKUP + LIVE THINKING)
    # ==========================================
    def build_page_ai_chat(self):
        self.page_ai_chat.grid_columnconfigure(0, weight=0, minsize=260) # Left ChatGPT-style Sessions Sidebar
        self.page_ai_chat.grid_columnconfigure(1, weight=1)             # Right Chat View Area
        self.page_ai_chat.grid_rowconfigure(0, weight=1)

        # LEFT SUB-SIDEBAR: CHAT SESSIONS HISTORY & BACKUPS
        self.chat_sidebar = ctk.CTkFrame(self.page_ai_chat, corner_radius=8, fg_color=("gray85", "gray18"))
        self.chat_sidebar.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        self.chat_sidebar.grid_columnconfigure(0, weight=1)
        self.chat_sidebar.grid_rowconfigure(2, weight=1)

        self.btn_new_chat = ctk.CTkButton(
            self.chat_sidebar,
            text="➕ Yeni Sohbet Başlat",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
            fg_color="#2980B9",
            hover_color="#1F618D",
            command=self.start_new_chat_session_gui
        )
        self.btn_new_chat.grid(row=0, column=0, padx=10, pady=(12, 6), sticky="ew")

        self.btn_export_chat = ctk.CTkButton(
            self.chat_sidebar,
            text="💾 Sohbet Yedeğini İndir (JSON)",
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color=("gray75", "gray30"),
            command=self.export_chat_backups_gui
        )
        self.btn_export_chat.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")

        self.chat_sessions_scroll = ctk.CTkScrollableFrame(self.chat_sidebar, fg_color="transparent")
        self.chat_sessions_scroll.grid(row=2, column=0, padx=5, pady=0, sticky="nsew")
        self.chat_sessions_scroll.grid_columnconfigure(0, weight=1)

        # RIGHT MAIN CHAT AREA
        self.chat_main_area = ctk.CTkFrame(self.page_ai_chat, fg_color="transparent")
        self.chat_main_area.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        self.chat_main_area.grid_columnconfigure(0, weight=1)
        self.chat_main_area.grid_rowconfigure(1, weight=1)

        # Top Cortex AI Status Bar
        self.ai_status_bar = ctk.CTkFrame(self.chat_main_area, corner_radius=8, fg_color=("gray85", "gray20"))
        self.ai_status_bar.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")
        self.ai_status_bar.grid_columnconfigure(0, weight=1)

        self.ai_status_lbl = ctk.CTkLabel(
            self.ai_status_bar, 
            text="⏳ Cortex Engine (Ollama) durumu denetleniyor...", 
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.ai_status_lbl.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.ai_reconnect_btn = ctk.CTkButton(
            self.ai_status_bar,
            text="🔄 Yeniden Bağlan",
            width=130,
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.check_ai_status_ui
        )
        self.ai_reconnect_btn.grid(row=0, column=1, padx=15, pady=10)

        # Scrollable Active Chat Messages Log
        self.ai_chat_scroll = ctk.CTkScrollableFrame(self.chat_main_area, fg_color="transparent")
        self.ai_chat_scroll.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.ai_chat_scroll.grid_columnconfigure(0, weight=1)

        # Bottom Input Bar
        self.ai_input_frame = ctk.CTkFrame(self.chat_main_area, corner_radius=8)
        self.ai_input_frame.grid(row=2, column=0, padx=0, pady=(8, 0), sticky="ew")
        self.ai_input_frame.grid_columnconfigure(0, weight=1)

        self.ai_chat_entry = ctk.CTkEntry(
            self.ai_input_frame,
            placeholder_text="Cortex AI Asistanına sorun (Örn: Bana video üreten modelleri öner, Türkçe LLM'leri listele)...",
            height=40
        )
        self.ai_chat_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")
        self.ai_chat_entry.bind("<Return>", lambda e: self.send_ai_chat_message())

        self.ai_send_btn = ctk.CTkButton(
            self.ai_input_frame,
            text="Gönder 🚀",
            font=ctk.CTkFont(weight="bold"),
            width=110,
            height=40,
            command=self.send_ai_chat_message
        )
        self.ai_send_btn.grid(row=0, column=1, padx=(5, 12), pady=10)

    def scroll_chat_to_bottom(self):
        """Auto-scrolls the AI Chat scrollable canvas to the bottom."""
        try:
            self.after(20, lambda: self.ai_chat_scroll._parent_canvas.yview_moveto(1.0))
            self.after(100, lambda: self.ai_chat_scroll._parent_canvas.yview_moveto(1.0))
        except Exception:
            pass

    def start_thinking_ui(self):
        """Shows live AI logo, thinking/searching status indicator and timer."""
        self.thinking_seconds = 0.0
        self.is_thinking = True

        self.thinking_bubble = ctk.CTkFrame(
            self.ai_chat_scroll, 
            corner_radius=12, 
            fg_color=("gray80", "#24282E")
        )
        self.thinking_bubble.pack(fill="x", padx=5, pady=6, anchor="w")
        self.thinking_bubble.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkLabel(
            self.thinking_bubble, 
            text="🧠 Cortex AI Engine", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            text_color="#2ECC71"
        )
        hdr.pack(anchor="w", padx=14, pady=(10, 4))

        self.thinking_status_lbl = ctk.CTkLabel(
            self.thinking_bubble, 
            text="🔍 Hugging Face verileri taranıyor & Cortex AI düşünüyor... ⏱️ (0.0 saniye)", 
            font=ctk.CTkFont(family="Segoe UI", size=13, slant="italic"),
            text_color="#F39C12",
            justify="left", 
            anchor="w"
        )
        self.thinking_status_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        self.scroll_chat_to_bottom()
        self.tick_thinking_timer()

    def tick_thinking_timer(self):
        if getattr(self, "is_thinking", False):
            self.thinking_seconds += 0.2
            if hasattr(self, "thinking_status_lbl") and self.thinking_status_lbl.winfo_exists():
                if self.thinking_seconds < 2.0:
                    phase = "🔍 Hugging Face kataloğundaki veriler taranıyor..."
                else:
                    phase = "🧠 Cortex AI düşünüyor ve yanıtı oluşturuyor..."

                self.thinking_status_lbl.configure(
                    text=f"{phase}  ⏱️ ({self.thinking_seconds:.1f} saniye)"
                )
            self.scroll_chat_to_bottom()
            self.after(200, self.tick_thinking_timer)

    def stop_thinking_ui(self):
        self.is_thinking = False
        if hasattr(self, "thinking_bubble") and self.thinking_bubble:
            try:
                self.thinking_bubble.destroy()
            except Exception:
                pass

    def refresh_chat_sessions_list_ui(self):
        for widget in self.chat_sessions_scroll.winfo_children():
            widget.destroy()

        sessions = load_chat_sessions()
        if not sessions:
            new_sess = create_chat_session("Yeni Sohbet")
            sessions = [new_sess]

        if not self.active_chat_session_id and sessions:
            self.active_chat_session_id = sessions[0]["session_id"]

        for sess in sessions:
            sid = sess["session_id"]
            title = sess.get("title", "Sohbet")
            is_active = (sid == self.active_chat_session_id)

            btn_frame = ctk.CTkFrame(
                self.chat_sessions_scroll,
                corner_radius=6,
                fg_color=("gray75", "gray30") if is_active else ("gray85", "gray22")
            )
            btn_frame.pack(fill="x", padx=2, pady=3)
            btn_frame.grid_columnconfigure(0, weight=1)

            btn_title = ctk.CTkButton(
                btn_frame,
                text=f"💬 {title}",
                font=ctk.CTkFont(size=12, weight="bold" if is_active else "normal"),
                height=32,
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                command=lambda s=sid: self.load_active_chat_session(s)
            )
            btn_title.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

            del_btn = ctk.CTkButton(
                btn_frame,
                text="🗑️",
                width=28,
                height=28,
                fg_color="transparent",
                hover_color="#C0392B",
                command=lambda s=sid: self.delete_chat_session_gui(s)
            )
            del_btn.grid(row=0, column=1, padx=4, pady=2)

        self.render_active_chat_messages()

    def start_new_chat_session_gui(self):
        new_sess = create_chat_session("Yeni Sohbet")
        self.active_chat_session_id = new_sess["session_id"]
        self.refresh_chat_sessions_list_ui()

    def load_active_chat_session(self, session_id):
        self.active_chat_session_id = session_id
        self.refresh_chat_sessions_list_ui()

    def delete_chat_session_gui(self, session_id):
        delete_chat_session(session_id)
        sessions = load_chat_sessions()
        if sessions:
            self.active_chat_session_id = sessions[0]["session_id"]
        else:
            self.active_chat_session_id = None
        self.refresh_chat_sessions_list_ui()

    def render_active_chat_messages(self):
        for widget in self.ai_chat_scroll.winfo_children():
            widget.destroy()

        sessions = load_chat_sessions()
        active_sess = next((s for s in sessions if s["session_id"] == self.active_chat_session_id), None)
        if not active_sess:
            return

        for m in active_sess.get("messages", []):
            sender_name = "👤 Siz" if m["sender"] == "user" else "🧠 Cortex AI"
            self.render_ai_chat_bubble(sender_name, m["text"], m.get("timestamp", ""))

        self.scroll_chat_to_bottom()

    def export_chat_backups_gui(self):
        if not os.path.exists(CHAT_HISTORY_FILE):
            messagebox.showwarning("Yedek Yok", "Henüz kaydedilmiş sohbet geçmişi yok.")
            return

        target_file = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Dosyası", "*.json")],
            initialfile="cortex_chat_history_backup.json"
        )
        if target_file:
            try:
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as src, open(target_file, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Başarılı", f"Sohbet yedeğiniz kaydedildi:\n{target_file}")
            except Exception as e:
                messagebox.showerror("Hata", f"Yedekleme sırasında hata oluştu: {e}")

    def check_ai_status_ui(self):
        status = ai_summarizer.check_ollama_status()
        self.ai_status_lbl.configure(text=status["message"])
        if status["connected"]:
            self.ai_status_bar.configure(fg_color=("gray85", "#1E3A2B"))
        else:
            self.ai_status_bar.configure(fg_color=("gray85", "#4A1D1D"))

    def send_ai_chat_message(self):
        msg = self.ai_chat_entry.get().strip()
        if not msg:
            return

        if not self.active_chat_session_id:
            new_sess = create_chat_session("Yeni Sohbet")
            self.active_chat_session_id = new_sess["session_id"]

        self.ai_chat_entry.delete(0, "end")
        add_chat_message(self.active_chat_session_id, "user", msg)
        self.refresh_chat_sessions_list_ui()

        self.ai_send_btn.configure(state="disabled")
        self.start_thinking_ui()

        # Get message history for context
        sessions = load_chat_sessions()
        active_sess = next((s for s in sessions if s["session_id"] == self.active_chat_session_id), None)
        history = active_sess.get("messages", []) if active_sess else []

        threading.Thread(target=self.process_ai_chat_reply, args=(msg, history), daemon=True).start()

    def process_ai_chat_reply(self, user_msg, history):
        reply = ai_summarizer.chat_general_ai_assistant(user_msg, history)
        self.safe_gui_update(self.on_ai_chat_reply_received, reply)

    def on_ai_chat_reply_received(self, reply):
        elapsed = getattr(self, "thinking_seconds", 0.0)
        self.stop_thinking_ui()
        self.ai_send_btn.configure(state="normal")

        if elapsed > 0:
            reply_with_time = f"{reply}\n\n⏱️ (Cortex AI {elapsed:.1f} saniyede düşündü ve yanıtladı)"
        else:
            reply_with_time = reply

        if self.active_chat_session_id:
            add_chat_message(self.active_chat_session_id, "cortex", reply_with_time)
        self.refresh_chat_sessions_list_ui()
        self.scroll_chat_to_bottom()

    def render_ai_chat_bubble(self, sender, message, timestamp=""):
        is_user = ("Siz" in sender)
        
        # Sleek dark ChatGPT-style bubble colors & corners
        bubble_bg = ("gray85", "#1E2A38") if is_user else ("gray80", "#24282E")
        sender_color = "#64B5F6" if is_user else "#2ECC71"

        bubble = ctk.CTkFrame(
            self.ai_chat_scroll, 
            corner_radius=12, 
            fg_color=bubble_bg
        )
        bubble.pack(fill="x", padx=5, pady=6, anchor="w")
        bubble.grid_columnconfigure(0, weight=1)

        hdr_txt = f"{sender}  •  {timestamp}" if timestamp else sender
        sender_lbl = ctk.CTkLabel(
            bubble, 
            text=hdr_txt, 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            text_color=sender_color
        )
        sender_lbl.pack(anchor="w", padx=14, pady=(10, 4))

        # Clean markdown formatting characters (**bold**, `code`, etc.)
        formatted_text = clean_markdown_text(message)

        msg_lbl = ctk.CTkLabel(
            bubble, 
            text=formatted_text, 
            font=ctk.CTkFont(family="Segoe UI", size=13),
            justify="left", 
            anchor="w", 
            wraplength=660
        )
        msg_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        # Extract mentioned HuggingFace Repo IDs & render individual Model Action Cards
        potential_repos = set(re.findall(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+', message))
        if potential_repos and not is_user:
            actions_hdr = ctk.CTkLabel(
                bubble, 
                text="⚡ Önerilen Modeller ve Hızlı Eylemler:", 
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#2ECC71"
            )
            actions_hdr.pack(anchor="w", padx=14, pady=(2, 4))

            for repo_id in list(potential_repos)[:4]: # Max 4 models per message
                card_item = ctk.CTkFrame(bubble, fg_color=("gray75", "#1A1D20"), corner_radius=8)
                card_item.pack(fill="x", padx=14, pady=4)
                card_item.grid_columnconfigure(0, weight=1)

                lbl_repo = ctk.CTkLabel(
                    card_item, 
                    text=f"📦 {repo_id}", 
                    font=ctk.CTkFont(size=12, weight="bold")
                )
                lbl_repo.grid(row=0, column=0, padx=10, pady=8, sticky="w")

                btn_box = ctk.CTkFrame(card_item, fg_color="transparent")
                btn_box.grid(row=0, column=1, padx=6, pady=4, sticky="e")

                btn_inspect = ctk.CTkButton(
                    btn_box,
                    text="🔍 Analiz Et & İndir",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    height=28,
                    fg_color="#27AE60",
                    hover_color="#1E8449",
                    command=lambda r=repo_id: self.inspect_from_explore(r, "model")
                )
                btn_inspect.pack(side="left", padx=3)

                btn_star = ctk.CTkButton(
                    btn_box,
                    text="⭐ Favorile",
                    font=ctk.CTkFont(size=11),
                    height=28,
                    width=80,
                    fg_color="#D4AC0D",
                    hover_color="#B7950B",
                    command=lambda r=repo_id: self.quick_star_repo(r)
                )
                btn_star.pack(side="left", padx=3)

    def quick_star_repo(self, repo_id):
        add_to_favorites(repo_id, "model", "Cortex AI önerisiyle eklendi.")
        messagebox.showinfo("Favoriye Eklendi", f"'{repo_id}' favorilerinize eklendi!")

    def send_model_qa_chat(self):
        q = self.qa_entry.get().strip()
        if not q:
            return

        if not self.current_repo_id:
            messagebox.showwarning("Model Seçilmedi", "Lütfen önce bir model linki girip '🔍 Verileri Getir' butonuna basın.")
            return

        self.qa_entry.delete(0, "end")
        self.summary_text.configure(state="normal")
        self.summary_text.insert("end", f"\n\n❓ Soru: {q}\n⏳ Cortex AI bu modeli inceliyor ve düşünüyor...\n")
        self.summary_text.see("end")
        self.summary_text.configure(state="disabled")

        threading.Thread(
            target=self.process_model_qa, 
            args=(self.current_repo_id, q, self.current_readme_content), 
            daemon=True
        ).start()

    def process_model_qa(self, repo_id, question, readme):
        reply = ai_summarizer.chat_about_model(repo_id, question, readme)
        self.safe_gui_update(self.on_model_qa_received, reply)

    def on_model_qa_received(self, reply):
        cleaned_reply = clean_markdown_text(reply)
        self.summary_text.configure(state="normal")
        self.summary_text.insert("end", f"🧠 Cortex AI Yanıtı:\n{cleaned_reply}\n")
        self.summary_text.see("end")
        self.summary_text.configure(state="disabled")

    # ==========================================
    # PAGE 4: EXPLORE CATALOG & INTENT SEARCH
    # ==========================================
    def build_page_explore(self):
        self.page_explore.grid_columnconfigure(0, weight=1)
        self.page_explore.grid_rowconfigure(2, weight=1)

        self.explore_header = ctk.CTkFrame(self.page_explore, corner_radius=8, fg_color=("gray85", "gray20"))
        self.explore_header.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")
        
        self.exp_hdr_lbl = ctk.CTkLabel(
            self.explore_header,
            text="🌐 Hugging Face Model & Görev Kataloğu",
            font=ctk.CTkFont(size=17, weight="bold")
        )
        self.exp_hdr_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.exp_hdr_sub = ctk.CTkLabel(
            self.explore_header,
            text="Ne yapmak istediğinizi açıklayın (Örn: 'Text to image', 'Image to video', 'Çeviri modeli') veya kategorileri filtreleyin.",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.exp_hdr_sub.pack(anchor="w", padx=15, pady=(0, 10))

        self.explore_controls = ctk.CTkFrame(self.page_explore, corner_radius=8)
        self.explore_controls.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.explore_controls.grid_columnconfigure(0, weight=1)

        self.explore_search_entry = ctk.CTkEntry(
            self.explore_controls,
            placeholder_text="🔍 Ne yapmak istediğinizi yazın (Örn: Resim çizen model, Text to video, ses tanıma)...",
            height=38
        )
        self.explore_search_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")
        self.explore_search_entry.bind("<Return>", lambda e: self.start_explore_search_thread())

        self.explore_btn = ctk.CTkButton(
            self.explore_controls,
            text="🔍 Bul",
            width=100,
            height=38,
            font=ctk.CTkFont(weight="bold"),
            command=self.start_explore_search_thread
        )
        self.explore_btn.grid(row=0, column=1, padx=5, pady=10)

        # ALL HuggingFace Categories Dropdown Menu
        self.explore_cat_menu = ctk.CTkOptionMenu(
            self.explore_controls,
            values=list(hf_api.CATEGORY_OPTIONS.keys()),
            command=lambda choice: self.start_explore_search_thread(),
            height=38,
            width=240
        )
        self.explore_cat_menu.grid(row=0, column=2, padx=(5, 12), pady=10)

        # Scrollable Cards List
        self.explore_scroll_frame = ctk.CTkScrollableFrame(self.page_explore, fg_color="transparent")
        self.explore_scroll_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        self.explore_scroll_frame.grid_columnconfigure(0, weight=1)

    def start_explore_search_thread(self):
        self._explore_loaded = True
        for widget in self.explore_scroll_frame.winfo_children():
            widget.destroy()

        loading_lbl = ctk.CTkLabel(
            self.explore_scroll_frame,
            text="⏳ Hugging Face kataloğu taranıyor ve uygun modeller getiriliyor...",
            font=ctk.CTkFont(size=14)
        )
        loading_lbl.pack(pady=40)

        query = self.explore_search_entry.get().strip()
        cat_display = self.explore_cat_menu.get()
        filter_tag = hf_api.CATEGORY_OPTIONS.get(cat_display, "all")

        threading.Thread(target=self.fetch_explore_results, args=(query, filter_tag), daemon=True).start()

    def fetch_explore_results(self, query, filter_tag):
        results = hf_api.search_hf_hub(query=query, filter_tag=filter_tag, limit=30)
        self.safe_gui_update(self.render_explore_results, results)

    def render_explore_results(self, results):
        for widget in self.explore_scroll_frame.winfo_children():
            widget.destroy()

        if not results:
            lbl = ctk.CTkLabel(
                self.explore_scroll_frame,
                text="Aramanıza uygun proje bulunamadı. Farklı kelimeler deneyin.",
                font=ctk.CTkFont(size=14),
                text_color="gray60"
            )
            lbl.pack(pady=40)
            return

        for item in results:
            card = ctk.CTkFrame(self.explore_scroll_frame, corner_radius=8, fg_color=("gray85", "gray20"))
            card.pack(fill="x", padx=5, pady=6)
            card.grid_columnconfigure(0, weight=1)

            repo_id = item["repo_id"]
            repo_type = item["repo_type"]
            tag = item["pipeline_tag"]
            likes = item["likes"]
            downloads = item["downloads"]
            author = item["author"]

            type_badge = "DATASET" if repo_type == "dataset" else "MODEL"
            title_text = f"🌐 {repo_id}   [{type_badge}]"
            details_text = f"🏷️ Kategori/Görev: {tag}  |  👤 Yazar: {author}\n❤️ Beğeni: {likes:,}  |  📥 İndirilme: {downloads:,}"

            title_lbl = ctk.CTkLabel(card, text=title_text, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            title_lbl.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

            det_lbl = ctk.CTkLabel(card, text=details_text, font=ctk.CTkFont(size=11), text_color="gray70", justify="left", anchor="w")
            det_lbl.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            star_btn = ctk.CTkButton(
                btn_box,
                text="⭐ Favorile",
                font=ctk.CTkFont(size=11),
                height=34,
                width=80,
                fg_color="#D4AC0D",
                hover_color="#B7950B",
                command=lambda r=repo_id, t=repo_type: self.quick_star_repo_with_type(r, t)
            )
            star_btn.pack(side="left", padx=4)

            inspect_btn = ctk.CTkButton(
                btn_box,
                text="🔍 Analiz Et & İndir",
                font=ctk.CTkFont(weight="bold"),
                height=34,
                command=lambda r_id=repo_id, r_type=repo_type: self.inspect_from_explore(r_id, r_type)
            )
            inspect_btn.pack(side="left", padx=4)

            web_btn = ctk.CTkButton(
                btn_box,
                text="🔗 HF'te Aç",
                height=34,
                width=75,
                fg_color=("gray70", "gray30"),
                command=lambda r_id=repo_id, r_type=repo_type: webbrowser.open(f"https://huggingface.co/{'datasets/' if r_type=='dataset' else ''}{r_id}")
            )
            web_btn.pack(side="left", padx=4)

    def quick_star_repo_with_type(self, repo_id, repo_type):
        dialog = ctk.CTkInputDialog(text=f"'{repo_id}' için bir not yazın:", title="⭐ Favoriye Ekle")
        user_note = dialog.get_input()
        if user_note is not None:
            add_to_favorites(repo_id, repo_type, user_note.strip())
            messagebox.showinfo("Başarılı", f"'{repo_id}' favorilerinize eklendi!")

    def inspect_from_explore(self, repo_id, repo_type):
        prefix = "https://huggingface.co/datasets/" if repo_type == "dataset" else "https://huggingface.co/"
        full_url = f"{prefix}{repo_id}"
        
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, full_url)
        self.show_page("search")
        self.start_fetch_data_thread()

    # ==========================================
    # PAGE 5: LOCAL MODELS
    # ==========================================
    def build_page_local(self):
        self.page_local.grid_columnconfigure(0, weight=1)
        self.page_local.grid_rowconfigure(1, weight=1)

        self.local_top_frame = ctk.CTkFrame(self.page_local, corner_radius=8)
        self.local_top_frame.grid(row=0, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.local_top_frame.grid_columnconfigure(1, weight=1)

        self.cache_size_lbl = ctk.CTkLabel(
            self.local_top_frame, 
            text="Toplam Önbellek Boyutu: Yükleniyor...", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.cache_size_lbl.grid(row=0, column=0, padx=15, pady=10)

        self.local_search_entry = ctk.CTkEntry(
            self.local_top_frame, 
            placeholder_text="Yerel modellerde ara...", 
            height=32
        )
        self.local_search_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.local_search_entry.bind("<KeyRelease>", self.filter_local_models)

        self.refresh_local_btn = ctk.CTkButton(
            self.local_top_frame,
            text="🔄 Yenile",
            width=90,
            height=32,
            command=self.refresh_local_models_thread
        )
        self.refresh_local_btn.grid(row=0, column=2, padx=(0, 15), pady=10)

        self.local_scroll_frame = ctk.CTkScrollableFrame(self.page_local, fg_color="transparent")
        self.local_scroll_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.local_scroll_frame.grid_columnconfigure(0, weight=1)

    # ==========================================
    # PAGE 6: SETTINGS & DRIVE SELECTOR
    # ==========================================
    def build_page_settings(self):
        self.page_settings.grid_columnconfigure(0, weight=1)

        self.settings_card = ctk.CTkFrame(self.page_settings, corner_radius=8)
        self.settings_card.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        self.settings_card.grid_columnconfigure(1, weight=1)

        self.sett_title = ctk.CTkLabel(self.settings_card, text="⚙️ Sürücü ve Genel Ayarlar", font=ctk.CTkFont(size=16, weight="bold"))
        self.sett_title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        self.sett_dir_lbl = ctk.CTkLabel(self.settings_card, text="Varsayılan İndirme Klasörü (Sabit):", font=ctk.CTkFont(weight="bold"))
        self.sett_dir_lbl.grid(row=1, column=0, padx=15, pady=15, sticky="w")

        self.sett_dir_entry = ctk.CTkEntry(self.settings_card, height=34)
        self.sett_dir_entry.grid(row=1, column=1, padx=10, pady=15, sticky="ew")
        self.sett_dir_entry.insert(0, self.settings.get("download_dir", ""))

        self.sett_dir_btn = ctk.CTkButton(self.settings_card, text="📁 Klasör Seç", width=100, height=34, command=self.browse_dir)
        self.sett_dir_btn.grid(row=1, column=2, padx=(0, 15), pady=15)

        self.sett_save_btn = ctk.CTkButton(self.settings_card, text="💾 Klasör Tercihini Kaydet", font=ctk.CTkFont(weight="bold"), height=34, command=self.save_settings_gui)
        self.sett_save_btn.grid(row=1, column=3, padx=(5, 15), pady=15)

        self.sett_drive_info = ctk.CTkLabel(
            self.settings_card,
            text=f"Sistemdeki Sürücüler: {', '.join(self.detected_drives)}\nÖneri: E: diski takılı ise varsayılan olarak E:\\huggingface klasörü seçilmesi önerilir.\nYapay Zeka Özetleri & Cortex AI: Bilgisayarınızda çalışan yerel Ollama (http://localhost:11434) LLM servisi kullanılarak üretilir.",
            font=ctk.CTkFont(size=12, slant="italic"),
            justify="left",
            anchor="w",
            text_color="gray70"
        )
        self.sett_drive_info.grid(row=2, column=0, columnspan=4, padx=15, pady=(5, 15), sticky="w")

    # ==========================================
    # DRIVE & DISK VALIDATION LOGIC
    # ==========================================
    def validate_download_destination(self, required_bytes=0):
        target_dir = self.settings.get("download_dir", "").strip()

        is_accessible, drive_letter, access_msg = storage.check_drive_accessibility(target_dir)
        if not is_accessible:
            messagebox.showerror(
                "Sürücü Bulunamadı / Takılı Değil",
                f"{access_msg}\n\nLütfen harici sürücünüzü takın veya Ayarlar sekmesinden geçerli bir indirme konumu seçin."
            )
            return False

        if required_bytes > 0:
            is_sufficient, free_bytes, free_str, space_msg = storage.check_sufficient_space(target_dir, required_bytes)
            if not is_sufficient:
                messagebox.showwarning("Yetersiz Disk Alanı", space_msg)
                return False

        return True

    def refresh_sidebar_disk_info(self):
        target_dir = self.settings.get("download_dir", "C:\\")
        info = storage.get_disk_info(target_dir)

        if info["success"]:
            self.sb_drive_path_lbl.configure(text=f"Hedef: {target_dir}")
            self.sb_disk_progress.set(info["percent_used"] / 100.0)

            cache_info = storage.scan_local_models(cache_dir=target_dir if target_dir else None)
            detail_txt = f"Boş: {info['free_str']}  /  Toplam: {info['total_str']}\nHF Önbelleği: {cache_info['total_cache_size_str']}"
            self.sb_disk_details_lbl.configure(text=detail_txt)
        else:
            self.sb_drive_path_lbl.configure(text=f"Erişilemiyor: {target_dir}")
            self.sb_disk_progress.set(0)
            self.sb_disk_details_lbl.configure(text="Sürücü takılı değil veya erişilemiyor!")

    # ==========================================
    # LOGIC & EVENT HANDLERS
    # ==========================================
    def safe_gui_update(self, func, *args, **kwargs):
        self.after(0, lambda: func(*args, **kwargs))

    def browse_dir(self):
        selected = filedialog.askdirectory(initialdir=self.settings.get("download_dir"))
        if selected:
            selected = os.path.normpath(selected)
            self.sett_dir_entry.delete(0, "end")
            self.sett_dir_entry.insert(0, selected)
            self.settings["download_dir"] = selected
            save_settings(self.settings)
            messagebox.showinfo("Başarılı", f"İndirme konumu sabitleyerek kaydedildi:\n{selected}")
            self.refresh_sidebar_disk_info()

    def save_settings_gui(self):
        self.settings["download_dir"] = self.sett_dir_entry.get().strip()
        save_settings(self.settings)
        self.refresh_sidebar_disk_info()
        messagebox.showinfo("Kaydedildi", "İndirme klasörü tercihiniz kaydedildi.")

    def start_fetch_data_thread(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Eksik Bilgi", "Lütfen bir Hugging Face URL'si veya Repo ID girin.")
            return

        self.status_label.configure(text="⚡ Model verileri ve dosya listesi anında getiriliyor...")
        self.progress_bar.set(0.3)
        self.analyze_btn.configure(state="disabled")

        threading.Thread(target=self.fetch_repo_data, args=(url,), daemon=True).start()

    def fetch_repo_data(self, url):
        repo_id, repo_type = hf_api.parse_hf_url(url)
        if not repo_id:
            self.safe_gui_update(self.on_fetch_failed, "Geçersiz Hugging Face URL formatı.")
            return

        self.current_repo_id = repo_id
        self.current_repo_type = repo_type

        details = hf_api.get_repo_details(repo_id, repo_type)
        if not details["success"]:
            self.safe_gui_update(self.on_fetch_failed, f"Depo bilgisi alınamadı: {details.get('error')}")
            return

        self.repo_files = details["files"]
        self.total_repo_size_bytes = details["total_size"]

        # Check local disk deduplication status for each file
        self.check_local_files_status()

        self.safe_gui_update(self.on_fetch_success, details, self.total_repo_size_bytes)

    def check_local_files_status(self):
        """Checks if files already exist in local target folder to avoid re-downloading."""
        target_dir = self.settings.get("download_dir", "").strip()
        safe_folder = self.current_repo_id.replace("/", "--")
        project_dir = os.path.join(target_dir, f"{self.current_repo_type}s--{safe_folder}")

        self.downloaded_bytes_total = 0
        for f in self.repo_files:
            local_fp = os.path.join(project_dir, f["path"])
            if os.path.exists(local_fp) and os.path.getsize(local_fp) == f["size"]:
                f["is_downloaded"] = True
                self.downloaded_bytes_total += f["size"]
            else:
                f["is_downloaded"] = False

    def on_fetch_failed(self, error_msg):
        self.status_label.configure(text="❌ Veri getirme başarısız.")
        self.progress_bar.set(0)
        self.analyze_btn.configure(state="normal")
        self.download_selected_btn.configure(state="disabled")
        self.one_click_btn.configure(state="disabled")
        self.star_model_btn.configure(state="disabled")
        self.btn_run_ai_analysis.configure(state="disabled")
        messagebox.showerror("Hata", error_msg)

    def on_fetch_success(self, details, total_bytes):
        self.status_label.configure(text="⚡ Model verileri anında getirildi. İndirmeye hazır!")
        self.progress_bar.set(1.0)
        self.analyze_btn.configure(state="normal")
        self.download_selected_btn.configure(state="normal")
        self.one_click_btn.configure(state="normal")
        self.star_model_btn.configure(state="normal")
        self.btn_run_ai_analysis.configure(state="normal")

        if is_favorite(self.current_repo_id):
            self.star_model_btn.configure(text="⭐ Favorilerde", fg_color="#B7950B")
        else:
            self.star_model_btn.configure(text="⭐ Favorilere Ekle", fg_color="#D4AC0D")

        type_str = "Dataset" if self.current_repo_type == "dataset" else "Model"
        info_text = (
            f"📌 Repo ID: {self.current_repo_id}\n"
            f"🏷️ Tür: {type_str} | 👤 Yazar: {details['author']}\n"
            f"📦 Toplam Depo Boyutu (Tüm Sürümler): {storage.format_size(total_bytes)}\n"
            f"❤️ Beğeni: {details['likes']} | 📥 İndirilme: {details['downloads']}"
        )
        self.meta_info_label.configure(text=info_text)

        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert(
            "0.0", 
            "⚡ Model dosyaları ve verileri başarıyla getirildi.\n\nİsterseniz yukarıdaki '🤖 Cortex AI ile Modeli Analiz Et' butonuna tıklayarak modelin işlevi hakkında detaylı Türkçe özet alabilirsiniz."
        )
        self.summary_text.configure(state="disabled")

        self.one_click_btn.configure(
            text=f"⚡ Tek Tıkla Tüm Modelleri & Sürümleri İndir ({storage.format_size(total_bytes)})"
        )

        self.clear_file_list()
        self.checkbox_vars.clear()
        self.populate_file_list(self.repo_files)
        self.select_smart_defaults()

    def start_manual_ai_analysis_thread(self):
        if not self.current_repo_id:
            return

        self.btn_run_ai_analysis.configure(state="disabled")
        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert(
            "0.0", 
            "⏳ Cortex AI (Ollama) model açıklamasını (README) okuyor ve Türkçe özet hazırlıyor...\nLütfen bekleyin."
        )
        self.summary_text.configure(state="disabled")

        threading.Thread(target=self.process_manual_ai_analysis, daemon=True).start()

    def process_manual_ai_analysis(self):
        readme = hf_api.scrape_readme(self.current_repo_id, self.current_repo_type)
        self.current_readme_content = readme
        summary = ai_summarizer.generate_summary(readme, self.current_repo_id)
        self.safe_gui_update(self.on_manual_ai_analysis_complete, summary)

    def on_manual_ai_analysis_complete(self, summary):
        self.btn_run_ai_analysis.configure(state="normal")
        cleaned_summary = clean_markdown_text(summary)
        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert("0.0", cleaned_summary)
        self.summary_text.configure(state="disabled")

    def clear_file_list(self):
        for widget in self.checkbox_widgets:
            widget.destroy()
        self.checkbox_widgets.clear()

    def populate_file_list(self, files_list):
        for f in files_list:
            var = ctk.StringVar(value="off")
            self.checkbox_vars[f["path"]] = var
            size_str = storage.format_size(f["size"])

            is_dl = f.get("is_downloaded", False)
            if is_dl:
                cb_text = f"{f['path']}  ({size_str})   [✓ İndirildi / Diskinde Var]"
                text_color = "#2ECC71"
            else:
                cb_text = f"{f['path']}  ({size_str})"
                text_color = ("gray10", "gray90")

            cb = ctk.CTkCheckBox(
                self.files_scroll_frame,
                text=cb_text,
                variable=var,
                onvalue="on",
                offvalue="off",
                font=ctk.CTkFont(size=12),
                text_color=text_color,
                command=self.update_selected_size_label
            )
            cb.pack(fill="x", anchor="w", padx=5, pady=3)
            self.checkbox_widgets.append(cb)

    def select_smart_defaults(self):
        """
        Universal Smart File Selection Engine:
        1. Always includes small config/tokenizer files.
        2. Filters out demo assets, preview videos, gifs, and sample images.
        3. Supports sharded models (model-00001-of-00005): selects all shards for complete model.
        4. Supports multi-component diffusers pipelines (transformer, vae, unet, text_encoder).
        5. Supports GGUF quantization repos: selects medium Q4_K_M / Q5_K_M by default.
        6. Supports multi-version single-file repos: selects latest primary model file.
        7. Deduplication: Automatically unchecks files that ALREADY exist locally on disk!
        """
        if not self.repo_files:
            return

        def is_useless_asset(path):
            p = path.lower()
            if any(p.startswith(x) for x in ["assets/", "demo/", "media/", "images/", "sample_images/", "eval/"]):
                return True
            if any(p.endswith(ext) for ext in [".gif", ".mp4", ".png", ".jpg", ".jpeg"]):
                return True
            return False

        large_files = [f for f in self.repo_files if f["size"] > 50 * 1024 * 1024 and not is_useless_asset(f["path"])]
        small_files = [f for f in self.repo_files if f["size"] <= 50 * 1024 * 1024 and not is_useless_asset(f["path"])]

        selected_paths = set(f["path"] for f in small_files)

        if large_files:
            comp_map = {}
            for f in large_files:
                parts = f["path"].split("/")
                comp = parts[0] if len(parts) > 1 else ""
                comp_map.setdefault(comp, []).append(f)

            for comp, files in comp_map.items():
                gguf_files = [f for f in files if f["path"].endswith(".gguf")]
                if gguf_files:
                    q4_or_q5 = [f for f in gguf_files if "q4_k_m" in f["path"].lower() or "q5_k_m" in f["path"].lower() or "q4_0" in f["path"].lower()]
                    selected_paths.add((q4_or_q5 or gguf_files)[0]["path"])
                else:
                    safetensors = [f for f in files if f["path"].endswith(".safetensors")]
                    candidates = safetensors if safetensors else files

                    shards = [f for f in candidates if re.search(r'-\d{5}-of-\d{5}', f["path"])]
                    if shards:
                        for sf in shards:
                            selected_paths.add(sf["path"])
                    else:
                        def version_key(f):
                            v_match = re.search(r'v?(\d+(\.\d+)+)', f["path"])
                            if v_match:
                                try:
                                    return [int(p) for p in v_match.group(1).split('.')]
                                except Exception:
                                    pass
                            return [0, f["size"]]

                        candidates.sort(key=version_key, reverse=True)
                        if candidates:
                            selected_paths.add(candidates[0]["path"])

        # Deduplication: Uncheck files that ALREADY exist on disk!
        for path, var in self.checkbox_vars.items():
            is_dl = any(f["is_downloaded"] for f in self.repo_files if f["path"] == path)
            if path in selected_paths and not is_dl:
                var.set("on")
            else:
                var.set("off")

        self.update_selected_size_label()

    def skip_downloaded_files(self):
        for f in self.repo_files:
            if f.get("is_downloaded", False):
                if f["path"] in self.checkbox_vars:
                    self.checkbox_vars[f["path"]].set("off")
        self.update_selected_size_label()

    def update_selected_size_label(self):
        if not self.repo_files:
            return

        selected_bytes = sum(f["size"] for f in self.repo_files if self.checkbox_vars.get(f["path"]) and self.checkbox_vars[f["path"]].get() == "on")
        downloaded_bytes = getattr(self, "downloaded_bytes_total", 0)

        selected_str = storage.format_size(selected_bytes)
        downloaded_str = storage.format_size(downloaded_bytes)
        total_str = storage.format_size(self.total_repo_size_bytes)

        if downloaded_bytes > 0:
            header_txt = f"📂 Depo Dosyaları  (☑️ Seçilen: {selected_str} / 💾 Diskinde Var: {downloaded_str} / 🌐 Depo: {total_str})"
        else:
            header_txt = f"📂 Depo Dosyaları  (☑️ Seçilen: {selected_str} / Toplam Depo: {total_str})"

        self.files_title.configure(text=header_txt)

        if selected_bytes > 0:
            self.download_selected_btn.configure(
                text=f"📥 Seçilen Eksik Dosyaları İndir ({selected_str})",
                state="normal"
            )
        else:
            self.download_selected_btn.configure(
                text="📥 Seçilen Dosyaları İndir (0 B)",
                state="disabled"
            )

    def filter_files(self, event=None):
        query = self.file_search_entry.get().strip().lower()
        self.clear_file_list()
        filtered = [f for f in self.repo_files if query in f["path"].lower()]

        for f in filtered:
            if f["path"] not in self.checkbox_vars:
                self.checkbox_vars[f["path"]] = ctk.StringVar(value="off")
            var = self.checkbox_vars[f["path"]]
            size_str = storage.format_size(f["size"])

            is_dl = f.get("is_downloaded", False)
            if is_dl:
                cb_text = f"{f['path']}  ({size_str})   [✓ İndirildi / Diskinde Var]"
                text_color = "#2ECC71"
            else:
                cb_text = f"{f['path']}  ({size_str})"
                text_color = ("gray10", "gray90")

            cb = ctk.CTkCheckBox(
                self.files_scroll_frame,
                text=cb_text,
                variable=var,
                onvalue="on",
                offvalue="off",
                font=ctk.CTkFont(size=12),
                text_color=text_color,
                command=self.update_selected_size_label
            )
            cb.pack(fill="x", anchor="w", padx=5, pady=3)
            self.checkbox_widgets.append(cb)

        self.update_selected_size_label()

    def toggle_select_all(self):
        val = self.select_all_var.get()
        for var in self.checkbox_vars.values():
            var.set(val)
        self.update_selected_size_label()

    def clear_all_selections(self):
        for var in self.checkbox_vars.values():
            var.set("off")
        self.select_all_var.set("off")
        self.update_selected_size_label()

    def start_one_click_download_thread(self):
        if not self.current_repo_id:
            return

        if not self.validate_download_destination(required_bytes=self.total_repo_size_bytes):
            return

        target_dir = self.settings.get("download_dir", "").strip()

        self.one_click_btn.configure(state="disabled")
        self.download_selected_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text=f"⚡ Tek Tıkla Tüm {self.current_repo_id} Deposu indiriliyor...")

        threading.Thread(target=self.download_snapshot_bg, args=(target_dir,), daemon=True).start()

    def download_snapshot_bg(self, target_dir):
        try:
            safe_folder = self.current_repo_id.replace("/", "--")
            project_dir = os.path.join(target_dir, f"{self.current_repo_type}s--{safe_folder}")
            os.makedirs(project_dir, exist_ok=True)

            snapshot_download(
                repo_id=self.current_repo_id,
                repo_type=self.current_repo_type,
                local_dir=project_dir,
                local_dir_use_symlinks=False
            )
            self.safe_gui_update(self.on_download_complete, True, project_dir)
        except Exception as e:
            self.safe_gui_update(self.on_download_complete, False, str(e))

    def start_download_selected_thread(self):
        selected_files = [path for path, var in self.checkbox_vars.items() if var.get() == "on"]
        if not selected_files:
            messagebox.showwarning("Seçim Yapılmadı", "Lütfen indirmek için en az bir dosya seçin.")
            return

        req_bytes = sum(f["size"] for f in self.repo_files if f["path"] in selected_files)

        if not self.validate_download_destination(required_bytes=req_bytes):
            return

        target_dir = self.settings.get("download_dir", "").strip()

        self.download_selected_btn.configure(state="disabled")
        self.one_click_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="İndirme başlatılıyor...")

        threading.Thread(target=self.download_files_bg, args=(selected_files, target_dir), daemon=True).start()

    def download_files_bg(self, selected_files, target_dir):
        try:
            safe_folder = self.current_repo_id.replace("/", "--")
            project_dir = os.path.join(target_dir, f"{self.current_repo_type}s--{safe_folder}")
            os.makedirs(project_dir, exist_ok=True)
            total = len(selected_files)

            for idx, file_path in enumerate(selected_files):
                self.safe_gui_update(
                    self.status_label.configure,
                    text=f"İndiriliyor ({idx+1}/{total}): {os.path.basename(file_path)}..."
                )
                self.safe_gui_update(self.progress_bar.set, idx / total)

                hf_hub_download(
                    repo_id=self.current_repo_id,
                    filename=file_path,
                    repo_type=self.current_repo_type,
                    local_dir=project_dir,
                    local_dir_use_symlinks=False
                )

            self.safe_gui_update(self.on_download_complete, True, project_dir)
        except Exception as e:
            self.safe_gui_update(self.on_download_complete, False, str(e))

    def on_download_complete(self, success, result_path):
        self.download_selected_btn.configure(state="normal")
        self.one_click_btn.configure(state="normal")
        self.progress_bar.set(1.0)

        if success:
            self.status_label.configure(text="🎉 İndirme başarıyla tamamlandı!")
            messagebox.showinfo("Başarılı", f"İndirme tamamlandı.\nKonum: {result_path}")
            self.refresh_local_models_thread()
            self.refresh_sidebar_disk_info()
        else:
            self.status_label.configure(text="❌ İndirme hatası.")
            messagebox.showerror("Hata", f"İndirme sırasında bir hata oluştu:\n{result_path}")

    # --- LOCAL MODELS LOGIC ---
    def refresh_local_models_thread(self):
        threading.Thread(target=self.refresh_local_models, daemon=True).start()

    def refresh_local_models(self):
        target_dir = self.settings.get("download_dir", "")
        res = storage.scan_local_models(cache_dir=target_dir if target_dir else None)
        self.local_models_data = res["models"]
        total_str = res["total_cache_size_str"]

        self.safe_gui_update(self.update_local_models_ui, total_str)

    def update_local_models_ui(self, total_cache_str):
        self.cache_size_lbl.configure(text=f"💾 Toplam Önbellek / Model Boyutu: {total_cache_str}")
        self.render_local_models_list(self.local_models_data)

    def render_local_models_list(self, models_list):
        for widget in self.local_scroll_frame.winfo_children():
            widget.destroy()

        if not models_list:
            lbl = ctk.CTkLabel(
                self.local_scroll_frame, 
                text="Henüz indirilmiş yerel bir Hugging Face modeli bulunamadı.", 
                font=ctk.CTkFont(size=14),
                text_color="gray60"
            )
            lbl.pack(pady=30)
            return

        for item in models_list:
            card = ctk.CTkFrame(self.local_scroll_frame, corner_radius=8, fg_color=("gray85", "gray20"))
            card.pack(fill="x", padx=5, pady=5)
            card.grid_columnconfigure(0, weight=1)

            repo_id = item["repo_id"]
            repo_type = item["repo_type"].upper()
            size_str = item["size_str"]
            local_path = item["local_path"]
            files_cnt = item["files_count"]

            title_str = f"📦 {repo_id}  [{repo_type}]"
            sub_str = f"Boyut: {size_str}  |  Dosya Sayısı: {files_cnt}\nKonum: {local_path}"

            lbl_title = ctk.CTkLabel(card, text=title_str, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            lbl_title.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

            lbl_sub = ctk.CTkLabel(card, text=sub_str, font=ctk.CTkFont(size=11), text_color="gray70", anchor="w", justify="left")
            lbl_sub.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            open_btn = ctk.CTkButton(
                btn_frame, 
                text="📂 Klasörde Aç", 
                width=100, 
                height=30,
                command=lambda path=local_path: self.open_folder(path)
            )
            open_btn.pack(side="left", padx=5)

            del_btn = ctk.CTkButton(
                btn_frame, 
                text="🗑️ Modeli Sil", 
                width=90, 
                height=30,
                fg_color="#C0392B",
                hover_color="#922B21",
                command=lambda itm=item: self.delete_model_action(itm)
            )
            del_btn.pack(side="left", padx=5)

    def filter_local_models(self, event=None):
        query = self.local_search_entry.get().strip().lower()
        filtered = [m for m in self.local_models_data if query in m["repo_id"].lower() or query in m["local_path"].lower()]
        self.render_local_models_list(filtered)

    def open_folder(self, path):
        if os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showerror("Hata", "Klasör bulunamadı.")

    def delete_model_action(self, item):
        confirm = messagebox.askyesno(
            "Model Silme Onayı", 
            f"'{item['repo_id']}' modelini diskten silmek istediğinize emin misiniz?\nKaplanan Alan: {item['size_str']}"
        )
        if confirm:
            target_dir = self.settings.get("download_dir", "")
            res = storage.delete_local_model(item, cache_dir=target_dir if target_dir else None)
            if res["success"]:
                messagebox.showinfo("Silindi", "Model başarıyla silindi.")
                self.refresh_local_models_thread()
                self.refresh_sidebar_disk_info()
            else:
                messagebox.showerror("Hata", f"Silme işlemi başarısız: {res.get('error')}")

if __name__ == "__main__":
    app = HFDownloaderApp()
    app.mainloop()
