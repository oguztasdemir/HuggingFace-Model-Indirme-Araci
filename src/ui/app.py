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
from src.download_manager import DownloadQueueManager
from src.i18n import t, set_lang, get_lang

# Appearance Mode & Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def format_seconds(sec):
    if not sec or sec <= 0:
        return "00s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def clean_markdown_text(text):
    """
    Converts raw markdown text (**bold**, `code`, etc.) into cleanly formatted display text.
    """
    if not text:
        return ""
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
    cleaned = re.sub(r'^\s*#{1,6}\s*', r'📌 ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*(\d+)\.\s*', r'\n🔹 \1. ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*[\-\*]\s*', r'  • ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

class HFDownloaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Load settings
        self.settings = load_settings()
        set_lang(self.settings.get("language", "tr"))

        self.title(t("app_title"))
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)

        # Initialize Download Manager Queue Engine
        self.download_queue = DownloadQueueManager()

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
        self.after(500, self.update_download_manager_ui_tick)

    # ==========================================
    # LEFT SIDEBAR PANEL BUILDER
    # ==========================================
    def build_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray15"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="🤗 HF Downloader", 
            font=ctk.CTkFont(family="Inter", size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 20), sticky="w")

        # Nav Buttons
        self.btn_nav_search = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_search"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("search")
        )
        self.btn_nav_search.grid(row=1, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_fav = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_favorites"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("favorites")
        )
        self.btn_nav_fav.grid(row=2, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_downloads = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_downloads"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("downloads")
        )
        self.btn_nav_downloads.grid(row=3, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_ai = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_ai_chat"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("ai_chat")
        )
        self.btn_nav_ai.grid(row=4, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_explore = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_explore"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("explore")
        )
        self.btn_nav_explore.grid(row=5, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_local = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_local"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("local")
        )
        self.btn_nav_local.grid(row=6, column=0, padx=15, pady=4, sticky="ew")

        self.btn_nav_settings = ctk.CTkButton(
            self.sidebar_frame, text=t("nav_settings"), font=ctk.CTkFont(size=14, weight="bold"), height=40, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray75", "gray25"),
            command=lambda: self.show_page("settings")
        )
        self.btn_nav_settings.grid(row=7, column=0, padx=15, pady=4, sticky="ew")

        self.sidebar_divider = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray75", "gray25"))
        self.sidebar_divider.grid(row=8, column=0, padx=15, pady=15, sticky="ew")

        # LIVE DISK STORAGE PANEL
        self.sidebar_disk_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=10, fg_color=("gray80", "gray20"))
        self.sidebar_disk_card.grid(row=10, column=0, padx=15, pady=(0, 20), sticky="ew")
        self.sidebar_disk_card.grid_columnconfigure(0, weight=1)

        self.sb_disk_title = ctk.CTkLabel(self.sidebar_disk_card, text=t("target_drive_status"), font=ctk.CTkFont(size=13, weight="bold"))
        self.sb_disk_title.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self.sb_drive_path_lbl = ctk.CTkLabel(self.sidebar_disk_card, text="Konum: Yükleniyor...", font=ctk.CTkFont(size=11), text_color="gray70")
        self.sb_drive_path_lbl.grid(row=1, column=0, padx=12, pady=0, sticky="w")

        self.sb_disk_progress = ctk.CTkProgressBar(self.sidebar_disk_card, height=12)
        self.sb_disk_progress.grid(row=2, column=0, padx=12, pady=8, sticky="ew")
        self.sb_disk_progress.set(0)

        self.sb_disk_details_lbl = ctk.CTkLabel(self.sidebar_disk_card, text="Boş: - GB / Toplam: - GB\nHF Önbelleği: -", font=ctk.CTkFont(size=11), justify="left", anchor="w")
        self.sb_disk_details_lbl.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="w")

        self.sb_refresh_btn = ctk.CTkButton(self.sidebar_disk_card, text=t("refresh_disk"), font=ctk.CTkFont(size=11), height=26, command=self.refresh_sidebar_disk_info)
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
        self.page_downloads = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_ai_chat = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_explore = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_local = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.page_settings = ctk.CTkFrame(self.main_container, fg_color="transparent")

        self.build_page_search()
        self.build_page_favorites()
        self.build_page_downloads()
        self.build_page_ai_chat()
        self.build_page_explore()
        self.build_page_local()
        self.build_page_settings()

    def show_page(self, page_name):
        self.page_search.grid_forget()
        self.page_favorites.grid_forget()
        self.page_downloads.grid_forget()
        self.page_ai_chat.grid_forget()
        self.page_explore.grid_forget()
        self.page_local.grid_forget()
        self.page_settings.grid_forget()

        active_color = ("gray75", "gray30")
        inactive_color = "transparent"

        self.btn_nav_search.configure(fg_color=inactive_color)
        self.btn_nav_fav.configure(fg_color=inactive_color)
        self.btn_nav_downloads.configure(fg_color=inactive_color)
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
        elif page_name == "downloads":
            self.page_downloads.grid(row=0, column=0, sticky="nsew")
            self.btn_nav_downloads.configure(fg_color=active_color)
            self.render_download_manager_state()
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
            self.top_web_bar, text=t("open_in_browser"), font=ctk.CTkFont(size=13, weight="bold"), height=34,
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
            command=lambda: webbrowser.open("https://huggingface.co/models")
        )
        self.btn_open_external_hf.grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.web_hint_lbl = ctk.CTkLabel(self.top_web_bar, text=t("web_hint"), font=ctk.CTkFont(size=11, slant="italic"), text_color="gray70")
        self.web_hint_lbl.grid(row=0, column=1, padx=(0, 12), pady=8, sticky="e")

        # URL Input Frame
        self.url_frame = ctk.CTkFrame(self.page_search, corner_radius=8)
        self.url_frame.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.url_frame.grid_columnconfigure(1, weight=1)

        self.url_lbl = ctk.CTkLabel(self.url_frame, text=t("url_label"), font=ctk.CTkFont(weight="bold"))
        self.url_lbl.grid(row=0, column=0, padx=(15, 10), pady=12)

        self.url_entry = ctk.CTkEntry(self.url_frame, placeholder_text=t("url_placeholder"), height=36)
        self.url_entry.grid(row=0, column=1, padx=5, pady=12, sticky="ew")

        self.analyze_btn = ctk.CTkButton(
            self.url_frame, text=t("fetch_data"), width=130, height=36, font=ctk.CTkFont(weight="bold"),
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

        self.files_title = ctk.CTkLabel(self.files_hdr, text=f"📂 {t('repo_files')} (0 B)", font=ctk.CTkFont(size=13, weight="bold"))
        self.files_title.grid(row=0, column=0, sticky="w")

        # Toolbar for smart selection & deduplication
        self.toolbar_frame = ctk.CTkFrame(self.files_frame, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, padx=10, pady=(0, 4), sticky="ew")

        self.smart_select_btn = ctk.CTkButton(
            self.toolbar_frame, text=t("smart_select"), font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#27AE60", hover_color="#1E8449", height=28, command=self.select_smart_defaults
        )
        self.smart_select_btn.pack(side="left", padx=(0, 4))

        self.skip_downloaded_btn = ctk.CTkButton(
            self.toolbar_frame, text=t("skip_downloaded"), font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2980B9", hover_color="#1F618D", height=28, command=self.skip_downloaded_files
        )
        self.skip_downloaded_btn.pack(side="left", padx=4)

        self.select_all_var = ctk.StringVar(value="off")
        self.select_all_cb = ctk.CTkCheckBox(
            self.toolbar_frame, text=t("select_all"), variable=self.select_all_var, onvalue="on", offvalue="off",
            font=ctk.CTkFont(size=11), command=self.toggle_select_all
        )
        self.select_all_cb.pack(side="left", padx=4)

        self.clear_select_btn = ctk.CTkButton(
            self.toolbar_frame, text=t("clear"), font=ctk.CTkFont(size=11), fg_color=("gray75", "gray30"),
            height=28, width=65, command=self.clear_all_selections
        )
        self.clear_select_btn.pack(side="left", padx=4)

        self.file_search_entry = ctk.CTkEntry(self.files_frame, placeholder_text=t("filter_placeholder"), height=28)
        self.file_search_entry.grid(row=2, column=0, padx=10, pady=4, sticky="ew")
        self.file_search_entry.bind("<KeyRelease>", self.filter_files)

        self.files_scroll_frame = ctk.CTkScrollableFrame(self.files_frame, fg_color="transparent")
        self.files_scroll_frame.grid(row=3, column=0, padx=10, pady=4, sticky="nsew")

        self.download_selected_btn = ctk.CTkButton(
            self.files_frame, text=t("download_selected"), height=38, command=self.start_download_selected_thread,
            state="disabled", font=ctk.CTkFont(weight="bold")
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
            self.one_click_frame, text=t("one_click_download"), font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#27AE60", hover_color="#1E8449", height=42, command=self.start_one_click_download_thread, state="disabled"
        )
        self.one_click_btn.grid(row=0, column=0, padx=12, pady=10, sticky="ew")

        self.one_click_hint = ctk.CTkLabel(self.one_click_frame, text=t("one_click_hint"), font=ctk.CTkFont(size=11, slant="italic"), text_color="gray70")
        self.one_click_hint.grid(row=1, column=0, padx=12, pady=(0, 8))

        # Metadata & Favorite Card
        self.meta_card = ctk.CTkFrame(self.info_pane, fg_color="transparent")
        self.meta_card.grid(row=1, column=0, padx=12, pady=2, sticky="ew")

        self.meta_info_label = ctk.CTkLabel(
            self.meta_card, text="📌 Lütfen bir Hugging Face linki girip '🔍 Verileri Getir' butonuna basın.",
            justify="left", anchor="w", font=ctk.CTkFont(size=13)
        )
        self.meta_info_label.pack(side="left", fill="x", expand=True, pady=2)

        self.star_model_btn = ctk.CTkButton(
            self.meta_card, text=t("star_add"), font=ctk.CTkFont(size=11, weight="bold"), height=28, width=110,
            fg_color="#D4AC0D", hover_color="#B7950B", state="disabled", command=self.open_add_favorite_dialog_current
        )
        self.star_model_btn.pack(side="right", padx=5, pady=2)

        # AI Summary Textbox & Action Buttons Header
        self.summary_hdr_frame = ctk.CTkFrame(self.info_pane, fg_color="transparent")
        self.summary_hdr_frame.grid(row=2, column=0, padx=12, pady=(6, 2), sticky="ew")

        self.summary_title = ctk.CTkLabel(self.summary_hdr_frame, text=t("ai_summary_title"), font=ctk.CTkFont(size=13, weight="bold"))
        self.summary_title.pack(side="left")

        # 3 AI Action Buttons: Analysis, vRAM, Usage Code
        self.btn_run_ai_analysis = ctk.CTkButton(
            self.summary_hdr_frame, text=t("run_ai_analysis"), font=ctk.CTkFont(size=11, weight="bold"),
            height=28, fg_color="#2980B9", hover_color="#1F618D", state="disabled", command=self.start_manual_ai_analysis_thread
        )
        self.btn_run_ai_analysis.pack(side="right", padx=3)

        self.btn_run_vram_analysis = ctk.CTkButton(
            self.summary_hdr_frame, text=t("analyze_vram"), font=ctk.CTkFont(size=11, weight="bold"),
            height=28, fg_color="#8E44AD", hover_color="#71368A", state="disabled", command=self.start_vram_analysis_thread
        )
        self.btn_run_vram_analysis.pack(side="right", padx=3)

        self.btn_run_code_gen = ctk.CTkButton(
            self.summary_hdr_frame, text=t("generate_code"), font=ctk.CTkFont(size=11, weight="bold"),
            height=28, fg_color="#D35400", hover_color="#A04000", state="disabled", command=self.start_code_gen_thread
        )
        self.btn_run_code_gen.pack(side="right", padx=3)

        self.summary_text = ctk.CTkTextbox(self.info_pane, wrap="word", font=ctk.CTkFont(size=13), height=140)
        self.summary_text.grid(row=3, column=0, padx=12, pady=(2, 6), sticky="nsew")
        self.summary_text.insert("0.0", "Hugging Face model verileri getirildikten sonra yukaridaki AI butonlarindan özet, vRAM analizi ve Python kodlari uretebilirsiniz.")
        self.summary_text.configure(state="disabled")

        # MODEL SPECIFIC QA CHAT
        self.model_qa_frame = ctk.CTkFrame(self.info_pane, corner_radius=8, fg_color=("gray85", "gray20"))
        self.model_qa_frame.grid(row=4, column=0, padx=12, pady=(4, 10), sticky="ew")
        self.model_qa_frame.grid_columnconfigure(0, weight=1)

        self.qa_title = ctk.CTkLabel(self.model_qa_frame, text=t("ask_cortex_title"), font=ctk.CTkFont(size=12, weight="bold"))
        self.qa_title.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        self.qa_entry = ctk.CTkEntry(self.model_qa_frame, placeholder_text=t("ask_placeholder"), height=30)
        self.qa_entry.grid(row=0, column=0, padx=(10, 4), pady=(0, 8), sticky="ew")
        self.qa_entry.bind("<Return>", lambda e: self.send_model_qa_chat())

        self.qa_btn = ctk.CTkButton(self.model_qa_frame, text=t("ask_btn"), width=65, height=30, font=ctk.CTkFont(weight="bold"), command=self.send_model_qa_chat)
        self.qa_btn.grid(row=0, column=1, padx=(0, 10), pady=(0, 8))

        # Bottom Progress & Speed Bar
        self.bottom_status_frame = ctk.CTkFrame(self.page_search, corner_radius=8)
        self.bottom_status_frame.grid(row=3, column=0, padx=0, pady=(10, 0), sticky="ew")
        self.bottom_status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.bottom_status_frame, text=t("status_ready"), font=ctk.CTkFont(size=12, weight="bold"))
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

        self.fav_title_lbl = ctk.CTkLabel(self.fav_header, text=t("fav_title"), font=ctk.CTkFont(size=17, weight="bold"))
        self.fav_title_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.fav_sub_lbl = ctk.CTkLabel(self.fav_header, text=t("fav_subtitle"), font=ctk.CTkFont(size=12), text_color="gray70")
        self.fav_sub_lbl.pack(anchor="w", padx=15, pady=(0, 10))

        self.fav_controls = ctk.CTkFrame(self.page_favorites, corner_radius=8)
        self.fav_controls.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.fav_controls.grid_columnconfigure(0, weight=2)
        self.fav_controls.grid_columnconfigure(1, weight=3)

        self.fav_add_url_entry = ctk.CTkEntry(self.fav_controls, placeholder_text=t("fav_url_placeholder"), height=34)
        self.fav_add_url_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")

        self.fav_add_note_entry = ctk.CTkEntry(self.fav_controls, placeholder_text=t("fav_note_placeholder"), height=34)
        self.fav_add_note_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.fav_add_btn = ctk.CTkButton(
            self.fav_controls, text=t("fav_add_btn"), font=ctk.CTkFont(weight="bold"), fg_color="#D4AC0D", hover_color="#B7950B",
            height=34, command=self.add_favorite_manual_gui
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
                text="Henüz favorilere eklenmiş bir model yok.",
                font=ctk.CTkFont(size=14), text_color="gray60"
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

            hdr_txt = f"⭐ {repo_id}   [{repo_type}]  •  {added_at}"
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
                note_frame, text="💾 Kaydet", font=ctk.CTkFont(size=11), height=28, width=80,
                command=lambda r=repo_id, e=note_entry: self.save_updated_note(r, e.get())
            )
            save_note_btn.grid(row=0, column=2, sticky="e")

            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            inspect_btn = ctk.CTkButton(
                btn_box, text="🔍 İncele", font=ctk.CTkFont(weight="bold"), height=34,
                command=lambda r=repo_id, t=item.get("repo_type", "model"): self.inspect_from_explore(r, t)
            )
            inspect_btn.pack(side="left", padx=4)

            remove_btn = ctk.CTkButton(
                btn_box, text="❌ Çıkar", height=34, width=75, fg_color="#C0392B", hover_color="#922B21",
                command=lambda r=repo_id: self.remove_favorite_action(r)
            )
            remove_btn.pack(side="left", padx=4)

    def add_favorite_manual_gui(self):
        url = self.fav_add_url_entry.get().strip()
        note = self.fav_add_note_entry.get().strip()
        if not url:
            messagebox.showwarning("Eksik Bilgi", "Lütfen bir URL veya Repo ID girin.")
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
                self.star_model_btn.configure(text=t("star_added"), fg_color="#B7950B")

    def save_updated_note(self, repo_id, new_note):
        update_favorite_note(repo_id, new_note.strip())
        messagebox.showinfo("Güncellendi", "Notunuz başarıyla kaydedildi!")

    def remove_favorite_action(self, repo_id):
        remove_from_favorites(repo_id)
        self.refresh_favorites_ui()

    # ==========================================
    # PAGE: DOWNLOAD MANAGER & QUEUE (STEAM STYLE)
    # ==========================================
    def build_page_downloads(self):
        self.page_downloads.grid_columnconfigure(0, weight=1)
        self.page_downloads.grid_rowconfigure(1, weight=1)

        # Header
        self.dl_header = ctk.CTkFrame(self.page_downloads, corner_radius=8, fg_color=("gray85", "gray20"))
        self.dl_header.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")

        self.dl_title_lbl = ctk.CTkLabel(self.dl_header, text=t("downloads_title"), font=ctk.CTkFont(size=17, weight="bold"))
        self.dl_title_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.dl_sub_lbl = ctk.CTkLabel(self.dl_header, text=t("downloads_subtitle"), font=ctk.CTkFont(size=12), text_color="gray70")
        self.dl_sub_lbl.pack(anchor="w", padx=15, pady=(0, 10))

        # Main scrollable body for active task & queue
        self.dl_scroll = ctk.CTkScrollableFrame(self.page_downloads, fg_color="transparent")
        self.dl_scroll.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.dl_scroll.grid_columnconfigure(0, weight=1)

        # -------------------------------------------------------------
        # 1. ACTIVE DOWNLOAD CARD (STEAM STYLE PERFORMANCE METRICS + LINE GRAPH)
        # -------------------------------------------------------------
        self.dl_active_card = ctk.CTkFrame(self.dl_scroll, corner_radius=10, fg_color=("gray85", "gray18"))
        self.dl_active_card.pack(fill="x", padx=5, pady=(0, 10))
        self.dl_active_card.grid_columnconfigure(0, weight=1)

        # Active Title Row
        self.dl_active_title_frame = ctk.CTkFrame(self.dl_active_card, fg_color="transparent")
        self.dl_active_title_frame.pack(fill="x", padx=15, pady=(12, 6))

        self.dl_active_repo_lbl = ctk.CTkLabel(
            self.dl_active_title_frame, text=t("no_active_download"),
            font=ctk.CTkFont(family="Inter", size=16, weight="bold"), anchor="w"
        )
        self.dl_active_repo_lbl.pack(side="left")

        self.dl_active_badge = ctk.CTkLabel(
            self.dl_active_title_frame, text="", font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="gray30", text_color="white", corner_radius=6, padx=8, pady=2
        )
        self.dl_active_badge.pack(side="right")

        # Live Metrics Grid (Current Mbps, Peak Mbps, Elapsed, ETA)
        self.dl_metrics_frame = ctk.CTkFrame(self.dl_active_card, fg_color="transparent")
        self.dl_metrics_frame.pack(fill="x", padx=15, pady=4)
        for i in range(4):
            self.dl_metrics_frame.grid_columnconfigure(i, weight=1)

        # Speed Card
        self.m_card_speed = ctk.CTkFrame(self.dl_metrics_frame, corner_radius=8, fg_color=("gray80", "gray25"))
        self.m_card_speed.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(self.m_card_speed, text=f"🚀 {t('current_speed')}", font=ctk.CTkFont(size=11), text_color="gray70").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_val_speed = ctk.CTkLabel(self.m_card_speed, text="0.0 Mbps", font=ctk.CTkFont(size=15, weight="bold"), text_color="#00F0FF")
        self.lbl_val_speed.pack(anchor="w", padx=10, pady=(0, 6))

        # Peak Card
        self.m_card_peak = ctk.CTkFrame(self.dl_metrics_frame, corner_radius=8, fg_color=("gray80", "gray25"))
        self.m_card_peak.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(self.m_card_peak, text=f"⚡ {t('peak_speed')}", font=ctk.CTkFont(size=11), text_color="gray70").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_val_peak = ctk.CTkLabel(self.m_card_peak, text="0.0 Mbps", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F39C12")
        self.lbl_val_peak.pack(anchor="w", padx=10, pady=(0, 6))

        # Elapsed Card
        self.m_card_elapsed = ctk.CTkFrame(self.dl_metrics_frame, corner_radius=8, fg_color=("gray80", "gray25"))
        self.m_card_elapsed.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(self.m_card_elapsed, text=f"⏱️ {t('elapsed_time')}", font=ctk.CTkFont(size=11), text_color="gray70").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_val_elapsed = ctk.CTkLabel(self.m_card_elapsed, text="00m 00s", font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_val_elapsed.pack(anchor="w", padx=10, pady=(0, 6))

        # ETA Card
        self.m_card_eta = ctk.CTkFrame(self.dl_metrics_frame, corner_radius=8, fg_color=("gray80", "gray25"))
        self.m_card_eta.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(self.m_card_eta, text=f"⏳ {t('remaining_time')}", font=ctk.CTkFont(size=11), text_color="gray70").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_val_eta = ctk.CTkLabel(self.m_card_eta, text="--", font=ctk.CTkFont(size=15, weight="bold"), text_color="#2ECC71")
        self.lbl_val_eta.pack(anchor="w", padx=10, pady=(0, 6))

        # Canvas Speed Line Graph (Steam Style)
        self.graph_frame = ctk.CTkFrame(self.dl_active_card, corner_radius=8, fg_color="#0F172A")
        self.graph_frame.pack(fill="x", padx=15, pady=8)

        self.canvas_graph = ctk.CTkCanvas(
            self.graph_frame, height=140, bg="#0F172A", highlightthickness=0, bd=0
        )
        self.canvas_graph.pack(fill="both", expand=True, padx=8, pady=8)

        # Progress Bar & Size Text
        self.dl_progress_bar = ctk.CTkProgressBar(self.dl_active_card, height=12)
        self.dl_progress_bar.pack(fill="x", padx=15, pady=(4, 2))
        self.dl_progress_bar.set(0)

        self.dl_size_lbl = ctk.CTkLabel(self.dl_active_card, text="0 B / 0 B (0%)", font=ctk.CTkFont(size=12), text_color="gray70")
        self.dl_size_lbl.pack(anchor="w", padx=15, pady=(0, 10))

        # Active Action Buttons (Pause / Resume / Cancel)
        self.dl_active_btns = ctk.CTkFrame(self.dl_active_card, fg_color="transparent")
        self.dl_active_btns.pack(anchor="e", padx=15, pady=(0, 12))

        self.btn_active_pause = ctk.CTkButton(
            self.dl_active_btns, text=t("pause"), font=ctk.CTkFont(weight="bold"), height=32, width=100,
            fg_color="#D35400", hover_color="#A04000", command=self.pause_active_download_gui
        )
        self.btn_active_pause.pack(side="left", padx=4)

        self.btn_active_resume = ctk.CTkButton(
            self.dl_active_btns, text=t("resume"), font=ctk.CTkFont(weight="bold"), height=32, width=100,
            fg_color="#27AE60", hover_color="#1E8449", command=self.resume_active_download_gui
        )
        self.btn_active_resume.pack(side="left", padx=4)

        self.btn_active_cancel = ctk.CTkButton(
            self.dl_active_btns, text=t("cancel"), font=ctk.CTkFont(weight="bold"), height=32, width=90,
            fg_color="#C0392B", hover_color="#922B21", command=self.cancel_active_download_gui
        )
        self.btn_active_cancel.pack(side="left", padx=4)

        # -------------------------------------------------------------
        # 2. QUEUE & RECENT LIST CARD
        # -------------------------------------------------------------
        self.dl_queue_card = ctk.CTkFrame(self.dl_scroll, corner_radius=10, fg_color=("gray85", "gray18"))
        self.dl_queue_card.pack(fill="both", expand=True, padx=5, pady=5)
        self.dl_queue_card.grid_columnconfigure(0, weight=1)

        self.dl_queue_hdr = ctk.CTkFrame(self.dl_queue_card, fg_color="transparent")
        self.dl_queue_hdr.pack(fill="x", padx=15, pady=(12, 6))
        self.dl_queue_hdr.grid_columnconfigure(0, weight=1)

        self.dl_queue_title = ctk.CTkLabel(self.dl_queue_hdr, text=t("download_queue"), font=ctk.CTkFont(size=15, weight="bold"))
        self.dl_queue_title.grid(row=0, column=0, sticky="w")

        self.btn_clear_finished = ctk.CTkButton(
            self.dl_queue_hdr, text=t("clear_finished"), font=ctk.CTkFont(size=11), height=28,
            fg_color=("gray75", "gray30"), command=self.clear_finished_downloads_gui
        )
        self.btn_clear_finished.grid(row=0, column=1, sticky="e")

        self.queue_items_frame = ctk.CTkFrame(self.dl_queue_card, fg_color="transparent")
        self.queue_items_frame.pack(fill="x", padx=10, pady=(0, 10))

    def draw_speed_graph(self, speed_history, current_mbps, peak_mbps):
        c = self.canvas_graph
        c.delete("all")

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        max_val = max(max(speed_history) if speed_history else 0, peak_mbps, 10.0)

        # Grid lines
        grid_steps = 3
        for i in range(grid_steps + 1):
            y = h - (i / grid_steps) * (h - 24) - 12
            val = (i / grid_steps) * max_val
            c.create_line(45, y, w - 10, y, fill="#1E293B", dash=(3, 3))
            c.create_text(25, y, text=f"{val:.0f}", fill="#64748B", font=("Segoe UI", 9))

        c.create_text(25, 10, text="Mbps", fill="#94A3B8", font=("Segoe UI", 9, "bold"))

        if not speed_history or len(speed_history) < 2:
            return

        pts = []
        num_pts = len(speed_history)
        x_step = (w - 55) / max(num_pts - 1, 1)
        for idx, spd in enumerate(speed_history):
            x = 45 + idx * x_step
            y = h - (spd / max_val) * (h - 24) - 12
            pts.append((x, y))

        poly_pts = [(45, h - 12)] + pts + [(pts[-1][0], h - 12)]
        flat_poly = [coord for pt in poly_pts for coord in pt]
        c.create_polygon(flat_poly, fill="#0F384C", outline="")

        flat_line = [coord for pt in pts for coord in pt]
        c.create_line(flat_line, fill="#00F0FF", width=2, smooth=True)

        last_x, last_y = pts[-1]
        c.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill="#00F0FF", outline="#FFFFFF", width=2)

    def update_download_manager_ui_tick(self):
        try:
            self.render_download_manager_state()
        except Exception:
            pass
        finally:
            self.after(500, self.update_download_manager_ui_tick)

    def render_download_manager_state(self):
        tasks = self.download_queue.get_tasks()
        active = self.download_queue.get_active_task()

        if active:
            self.dl_active_repo_lbl.configure(text=f"🚀 {active.repo_id}")
            st_key = f"status_{active.status}"
            self.dl_active_badge.configure(text=t(st_key), fg_color="#27AE60" if active.status == "downloading" else "#D35400")

            speed_mb = active.current_speed_mbps
            speed_mbs = (active.current_speed_bps / (1024 * 1024))
            self.lbl_val_speed.configure(text=f"{speed_mb:.1f} Mbps\n({speed_mbs:.2f} MB/s)")
            self.lbl_val_peak.configure(text=f"{active.peak_speed_mbps:.1f} Mbps")

            self.lbl_val_elapsed.configure(text=format_seconds(active.elapsed_seconds))
            self.lbl_val_eta.configure(text=format_seconds(active.eta_seconds) if active.eta_seconds > 0 else "--")

            pct = (active.downloaded_bytes / active.total_bytes) if active.total_bytes > 0 else 0
            self.dl_progress_bar.set(pct)
            size_txt = f"{storage.format_size(active.downloaded_bytes)} / {storage.format_size(active.total_bytes)}  ({pct*100:.1f}%)"
            self.dl_size_lbl.configure(text=size_txt)

            self.draw_speed_graph(list(active.speed_history), active.current_speed_mbps, active.peak_speed_mbps)

            self.btn_active_pause.configure(state="normal" if not active.pause_requested else "disabled")
            self.btn_active_resume.configure(state="disabled")
            self.btn_active_cancel.configure(state="normal")
        else:
            queued_or_paused = [t_item for t_item in tasks if t_item.status in ["queued", "paused"]]
            if queued_or_paused:
                first = queued_or_paused[0]
                self.dl_active_repo_lbl.configure(text=f"🚀 {first.repo_id}")
                st_key = f"status_{first.status}"
                self.dl_active_badge.configure(text=t(st_key), fg_color="#D35400" if first.status == "paused" else "#2980B9")
                
                self.lbl_val_speed.configure(text="0.0 Mbps")
                self.lbl_val_peak.configure(text=f"{first.peak_speed_mbps:.1f} Mbps")
                self.lbl_val_elapsed.configure(text=format_seconds(first.elapsed_seconds))
                self.lbl_val_eta.configure(text="--")

                pct = (first.downloaded_bytes / first.total_bytes) if first.total_bytes > 0 else 0
                self.dl_progress_bar.set(pct)
                size_txt = f"{storage.format_size(first.downloaded_bytes)} / {storage.format_size(first.total_bytes)}  ({pct*100:.1f}%)"
                self.dl_size_lbl.configure(text=size_txt)
                
                self.draw_speed_graph(list(first.speed_history), 0.0, first.peak_speed_mbps)

                if first.status == "paused":
                    self.btn_active_pause.configure(state="disabled")
                    self.btn_active_resume.configure(state="normal")
                else:
                    self.btn_active_pause.configure(state="normal")
                    self.btn_active_resume.configure(state="disabled")
                self.btn_active_cancel.configure(state="normal")
            else:
                self.dl_active_repo_lbl.configure(text=t("no_active_download"))
                self.dl_active_badge.configure(text="", fg_color="transparent")
                self.lbl_val_speed.configure(text="0.0 Mbps")
                self.lbl_val_peak.configure(text="0.0 Mbps")
                self.lbl_val_elapsed.configure(text="00m 00s")
                self.lbl_val_eta.configure(text="--")
                self.dl_progress_bar.set(0)
                self.dl_size_lbl.configure(text="0 B / 0 B (0%)")
                self.draw_speed_graph([0.0]*60, 0.0, 0.0)

                self.btn_active_pause.configure(state="disabled")
                self.btn_active_resume.configure(state="disabled")
                self.btn_active_cancel.configure(state="disabled")

        # Render Queue list
        for widget in self.queue_items_frame.winfo_children():
            widget.destroy()

        if not tasks:
            lbl = ctk.CTkLabel(self.queue_items_frame, text="İndirme sırasında henüz öge yok.", font=ctk.CTkFont(size=13), text_color="gray60")
            lbl.pack(pady=20)
            return

        for idx, t_item in enumerate(tasks):
            card = ctk.CTkFrame(self.queue_items_frame, corner_radius=8, fg_color=("gray80", "gray22"))
            card.pack(fill="x", padx=2, pady=4)
            card.grid_columnconfigure(0, weight=1)

            st_key = f"status_{t_item.status}"
            st_text = t(st_key)

            if t_item.status == "downloading":
                badge_color = "#27AE60"
            elif t_item.status == "paused":
                badge_color = "#D35400"
            elif t_item.status == "completed":
                badge_color = "#2980B9"
            elif t_item.status in ["failed", "cancelled"]:
                badge_color = "#C0392B"
            else: # queued
                badge_color = "#8E44AD"

            title_txt = f"{idx+1}. {t_item.repo_id}   [{t_item.mode.upper()}]  •  {storage.format_size(t_item.total_bytes)}"
            lbl_t = ctk.CTkLabel(card, text=title_txt, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
            lbl_t.grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

            status_lbl = ctk.CTkLabel(card, text=st_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=badge_color, anchor="w")
            status_lbl.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

            # Actions
            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.grid(row=0, column=1, rowspan=2, padx=10, pady=6, sticky="e")

            if t_item.status in ["queued", "paused"]:
                up_btn = ctk.CTkButton(btn_box, text="⬆️", width=32, height=28, command=lambda tid=t_item.task_id: self.download_queue.move_task_up(tid))
                up_btn.pack(side="left", padx=2)

                dn_btn = ctk.CTkButton(btn_box, text="⬇️", width=32, height=28, command=lambda tid=t_item.task_id: self.download_queue.move_task_down(tid))
                dn_btn.pack(side="left", padx=2)

            if t_item.status == "downloading":
                p_btn = ctk.CTkButton(btn_box, text="⏸️", width=34, height=28, fg_color="#D35400", command=lambda tid=t_item.task_id: self.download_queue.pause_task(tid))
                p_btn.pack(side="left", padx=2)
            elif t_item.status == "paused":
                r_btn = ctk.CTkButton(btn_box, text="▶️", width=34, height=28, fg_color="#27AE60", command=lambda tid=t_item.task_id: self.download_queue.resume_task(tid))
                r_btn.pack(side="left", padx=2)

            if t_item.status in ["queued", "downloading", "paused"]:
                c_btn = ctk.CTkButton(btn_box, text="❌", width=34, height=28, fg_color="#C0392B", command=lambda tid=t_item.task_id: self.download_queue.cancel_task(tid))
                c_btn.pack(side="left", padx=2)

    def pause_active_download_gui(self):
        active = self.download_queue.get_active_task()
        if active:
            self.download_queue.pause_task(active.task_id)

    def resume_active_download_gui(self):
        tasks = self.download_queue.get_tasks()
        paused = [t_item for t_item in tasks if t_item.status == "paused"]
        if paused:
            self.download_queue.resume_task(paused[0].task_id)

    def cancel_active_download_gui(self):
        active = self.download_queue.get_active_task()
        if active:
            self.download_queue.cancel_task(active.task_id)
        else:
            tasks = self.download_queue.get_tasks()
            queued = [t_item for t_item in tasks if t_item.status in ["queued", "paused"]]
            if queued:
                self.download_queue.cancel_task(queued[0].task_id)

    def clear_finished_downloads_gui(self):
        self.download_queue.clear_finished()
        self.render_download_manager_state()

    # ==========================================
    # PAGE 3: CORTEX AI ASSISTANT
    # ==========================================
    def build_page_ai_chat(self):
        self.page_ai_chat.grid_columnconfigure(0, weight=0, minsize=260)
        self.page_ai_chat.grid_columnconfigure(1, weight=1)
        self.page_ai_chat.grid_rowconfigure(0, weight=1)

        self.chat_sidebar = ctk.CTkFrame(self.page_ai_chat, corner_radius=8, fg_color=("gray85", "gray18"))
        self.chat_sidebar.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        self.chat_sidebar.grid_columnconfigure(0, weight=1)
        self.chat_sidebar.grid_rowconfigure(2, weight=1)

        self.btn_new_chat = ctk.CTkButton(
            self.chat_sidebar, text=t("ai_new_chat"), font=ctk.CTkFont(size=13, weight="bold"), height=36,
            fg_color="#2980B9", hover_color="#1F618D", command=self.start_new_chat_session_gui
        )
        self.btn_new_chat.grid(row=0, column=0, padx=10, pady=(12, 6), sticky="ew")

        self.btn_export_chat = ctk.CTkButton(
            self.chat_sidebar, text=t("ai_export_chat"), font=ctk.CTkFont(size=11), height=28,
            fg_color=("gray75", "gray30"), command=self.export_chat_backups_gui
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

        self.ai_status_bar = ctk.CTkFrame(self.chat_main_area, corner_radius=8, fg_color=("gray85", "gray20"))
        self.ai_status_bar.grid(row=0, column=0, padx=0, pady=(0, 8), sticky="ew")
        self.ai_status_bar.grid_columnconfigure(0, weight=1)

        self.ai_status_lbl = ctk.CTkLabel(self.ai_status_bar, text=t("ai_status_checking"), font=ctk.CTkFont(size=13, weight="bold"))
        self.ai_status_lbl.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.ai_reconnect_btn = ctk.CTkButton(
            self.ai_status_bar, text=t("ai_reconnect"), width=130, height=32, font=ctk.CTkFont(size=12, weight="bold"),
            command=self.check_ai_status_ui
        )
        self.ai_reconnect_btn.grid(row=0, column=1, padx=15, pady=10)

        self.ai_chat_scroll = ctk.CTkScrollableFrame(self.chat_main_area, fg_color="transparent")
        self.ai_chat_scroll.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.ai_chat_scroll.grid_columnconfigure(0, weight=1)

        self.ai_input_frame = ctk.CTkFrame(self.chat_main_area, corner_radius=8)
        self.ai_input_frame.grid(row=2, column=0, padx=0, pady=(8, 0), sticky="ew")
        self.ai_input_frame.grid_columnconfigure(0, weight=1)

        self.ai_chat_entry = ctk.CTkEntry(self.ai_input_frame, placeholder_text=t("ai_input_placeholder"), height=40)
        self.ai_chat_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")
        self.ai_chat_entry.bind("<Return>", lambda e: self.send_ai_chat_message())

        self.ai_send_btn = ctk.CTkButton(
            self.ai_input_frame, text=t("send"), font=ctk.CTkFont(weight="bold"), width=110, height=40, command=self.send_ai_chat_message
        )
        self.ai_send_btn.grid(row=0, column=1, padx=(5, 12), pady=10)

    def scroll_chat_to_bottom(self):
        try:
            self.after(20, lambda: self.ai_chat_scroll._parent_canvas.yview_moveto(1.0))
            self.after(100, lambda: self.ai_chat_scroll._parent_canvas.yview_moveto(1.0))
        except Exception:
            pass

    def start_thinking_ui(self):
        self.thinking_seconds = 0.0
        self.is_thinking = True

        self.thinking_bubble = ctk.CTkFrame(self.ai_chat_scroll, corner_radius=12, fg_color=("gray80", "#24282E"))
        self.thinking_bubble.pack(fill="x", padx=5, pady=6, anchor="w")
        self.thinking_bubble.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkLabel(self.thinking_bubble, text="🧠 Cortex AI Engine", font=ctk.CTkFont(family="Inter", size=12, weight="bold"), text_color="#2ECC71")
        hdr.pack(anchor="w", padx=14, pady=(10, 4))

        self.thinking_status_lbl = ctk.CTkLabel(
            self.thinking_bubble, text="🔍 Cortex AI taranıyor ve düşünüyor... ⏱️ (0.0 s)", font=ctk.CTkFont(family="Segoe UI", size=13, slant="italic"),
            text_color="#F39C12", justify="left", anchor="w"
        )
        self.thinking_status_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        self.scroll_chat_to_bottom()
        self.tick_thinking_timer()

    def tick_thinking_timer(self):
        if getattr(self, "is_thinking", False):
            self.thinking_seconds += 0.2
            if hasattr(self, "thinking_status_lbl") and self.thinking_status_lbl.winfo_exists():
                phase = "🔍 Hugging Face kataloğu taranıyor..." if self.thinking_seconds < 2.0 else "🧠 Cortex AI yanıtı oluşturuyor..."
                self.thinking_status_lbl.configure(text=f"{phase}  ⏱️ ({self.thinking_seconds:.1f} s)")
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

            btn_frame = ctk.CTkFrame(self.chat_sessions_scroll, corner_radius=6, fg_color=("gray75", "gray30") if is_active else ("gray85", "gray22"))
            btn_frame.pack(fill="x", padx=2, pady=3)
            btn_frame.grid_columnconfigure(0, weight=1)

            btn_title = ctk.CTkButton(
                btn_frame, text=f"💬 {title}", font=ctk.CTkFont(size=12, weight="bold" if is_active else "normal"),
                height=32, anchor="w", fg_color="transparent", text_color=("gray10", "gray90"),
                command=lambda s=sid: self.load_active_chat_session(s)
            )
            btn_title.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

            del_btn = ctk.CTkButton(
                btn_frame, text="🗑️", width=28, height=28, fg_color="transparent", hover_color="#C0392B",
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
        self.active_chat_session_id = sessions[0]["session_id"] if sessions else None
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

        target_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="cortex_chat_backup.json")
        if target_file:
            try:
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as src, open(target_file, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Başarılı", f"Sohbet yedeği kaydedildi:\n{target_file}")
            except Exception as e:
                messagebox.showerror("Hata", f"Yedekleme hatası: {e}")

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

        sessions = load_chat_sessions()
        active_sess = next((s for s in sessions if s["session_id"] == self.active_chat_session_id), None)
        history = active_sess.get("messages", []) if active_sess else []

        threading.Thread(target=self.process_ai_chat_reply, args=(msg, history), daemon=True).start()

    def process_ai_chat_reply(self, user_msg, history):
        reply = ai_summarizer.chat_general_ai_assistant(user_msg, history, lang=get_lang())
        self.safe_gui_update(self.on_ai_chat_reply_received, reply)

    def on_ai_chat_reply_received(self, reply):
        elapsed = getattr(self, "thinking_seconds", 0.0)
        self.stop_thinking_ui()
        self.ai_send_btn.configure(state="normal")

        reply_with_time = f"{reply}\n\n⏱️ (Cortex AI {elapsed:.1f} s)" if elapsed > 0 else reply
        if self.active_chat_session_id:
            add_chat_message(self.active_chat_session_id, "cortex", reply_with_time)
        self.refresh_chat_sessions_list_ui()
        self.scroll_chat_to_bottom()

    def render_ai_chat_bubble(self, sender, message, timestamp=""):
        is_user = ("Siz" in sender or "You" in sender)
        bubble_bg = ("gray85", "#1E2A38") if is_user else ("gray80", "#24282E")
        sender_color = "#64B5F6" if is_user else "#2ECC71"

        bubble = ctk.CTkFrame(self.ai_chat_scroll, corner_radius=12, fg_color=bubble_bg)
        bubble.pack(fill="x", padx=5, pady=6, anchor="w")
        bubble.grid_columnconfigure(0, weight=1)

        hdr_txt = f"{sender}  •  {timestamp}" if timestamp else sender
        sender_lbl = ctk.CTkLabel(bubble, text=hdr_txt, font=ctk.CTkFont(family="Inter", size=12, weight="bold"), text_color=sender_color)
        sender_lbl.pack(anchor="w", padx=14, pady=(10, 4))

        formatted_text = clean_markdown_text(message)
        msg_lbl = ctk.CTkLabel(bubble, text=formatted_text, font=ctk.CTkFont(family="Segoe UI", size=13), justify="left", anchor="w", wraplength=660)
        msg_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        potential_repos = set(re.findall(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+', message))
        if potential_repos and not is_user:
            actions_hdr = ctk.CTkLabel(bubble, text="⚡ Önerilen Modeller ve Hızlı Eylemler:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#2ECC71")
            actions_hdr.pack(anchor="w", padx=14, pady=(2, 4))

            for repo_id in list(potential_repos)[:4]:
                card_item = ctk.CTkFrame(bubble, fg_color=("gray75", "#1A1D20"), corner_radius=8)
                card_item.pack(fill="x", padx=14, pady=4)
                card_item.grid_columnconfigure(0, weight=1)

                lbl_repo = ctk.CTkLabel(card_item, text=f"📦 {repo_id}", font=ctk.CTkFont(size=12, weight="bold"))
                lbl_repo.grid(row=0, column=0, padx=10, pady=8, sticky="w")

                btn_box = ctk.CTkFrame(card_item, fg_color="transparent")
                btn_box.grid(row=0, column=1, padx=6, pady=4, sticky="e")

                btn_inspect = ctk.CTkButton(
                    btn_box, text="🔍 Analiz Et & İndir", font=ctk.CTkFont(size=11, weight="bold"), height=28,
                    fg_color="#27AE60", hover_color="#1E8449", command=lambda r=repo_id: self.inspect_from_explore(r, "model")
                )
                btn_inspect.pack(side="left", padx=3)

    def send_model_qa_chat(self):
        q = self.qa_entry.get().strip()
        if not q:
            return

        if not self.current_repo_id:
            messagebox.showwarning("Model Seçilmedi", "Lütfen önce bir model linki girip '🔍 Verileri Getir' butonuna basın.")
            return

        self.qa_entry.delete(0, "end")
        self.summary_text.configure(state="normal")
        self.summary_text.insert("end", f"\n\n❓ Soru: {q}\n⏳ Cortex AI bu modeli inceliyor...\n")
        self.summary_text.see("end")
        self.summary_text.configure(state="disabled")

        threading.Thread(
            target=self.process_model_qa, 
            args=(self.current_repo_id, q, self.current_readme_content), 
            daemon=True
        ).start()

    def process_model_qa(self, repo_id, question, readme):
        reply = ai_summarizer.chat_about_model(repo_id, question, readme, lang=get_lang())
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
        
        self.exp_hdr_lbl = ctk.CTkLabel(self.explore_header, text=t("explore_title"), font=ctk.CTkFont(size=17, weight="bold"))
        self.exp_hdr_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.exp_hdr_sub = ctk.CTkLabel(self.explore_header, text=t("explore_subtitle"), font=ctk.CTkFont(size=12), text_color="gray70")
        self.exp_hdr_sub.pack(anchor="w", padx=15, pady=(0, 10))

        self.explore_controls = ctk.CTkFrame(self.page_explore, corner_radius=8)
        self.explore_controls.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        self.explore_controls.grid_columnconfigure(0, weight=1)

        self.explore_search_entry = ctk.CTkEntry(self.explore_controls, placeholder_text=t("explore_search_placeholder"), height=38)
        self.explore_search_entry.grid(row=0, column=0, padx=(12, 5), pady=10, sticky="ew")
        self.explore_search_entry.bind("<Return>", lambda e: self.start_explore_search_thread())

        self.explore_btn = ctk.CTkButton(self.explore_controls, text=t("explore_find"), width=100, height=38, font=ctk.CTkFont(weight="bold"), command=self.start_explore_search_thread)
        self.explore_btn.grid(row=0, column=1, padx=5, pady=10)

        self.explore_cat_menu = ctk.CTkOptionMenu(
            self.explore_controls, values=list(hf_api.CATEGORY_OPTIONS.keys()),
            command=lambda choice: self.start_explore_search_thread(), height=38, width=240
        )
        self.explore_cat_menu.grid(row=0, column=2, padx=(5, 12), pady=10)

        self.explore_scroll_frame = ctk.CTkScrollableFrame(self.page_explore, fg_color="transparent")
        self.explore_scroll_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        self.explore_scroll_frame.grid_columnconfigure(0, weight=1)

    def start_explore_search_thread(self):
        self._explore_loaded = True
        for widget in self.explore_scroll_frame.winfo_children():
            widget.destroy()

        loading_lbl = ctk.CTkLabel(self.explore_scroll_frame, text="⏳ Hugging Face kataloğu taranıyor...", font=ctk.CTkFont(size=14))
        loading_lbl.pack(pady=40)

        query = self.explore_search_entry.get().strip()
        cat_display = self.explore_cat_menu.get()
        filter_tag = hf_api.CATEGORY_OPTIONS.get(cat_display, "all")
        token = self.settings.get("hf_token", "").strip()

        threading.Thread(target=self.fetch_explore_results, args=(query, filter_tag, token), daemon=True).start()

    def fetch_explore_results(self, query, filter_tag, token):
        results = hf_api.search_hf_hub(query=query, filter_tag=filter_tag, limit=30, token=token)
        self.safe_gui_update(self.render_explore_results, results)

    def render_explore_results(self, results):
        for widget in self.explore_scroll_frame.winfo_children():
            widget.destroy()

        if not results:
            lbl = ctk.CTkLabel(self.explore_scroll_frame, text="Aramanıza uygun proje bulunamadı.", font=ctk.CTkFont(size=14), text_color="gray60")
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
            details_text = f"🏷️ Görev: {tag}  |  👤 Yazar: {author}\n❤️ Beğeni: {likes:,}  |  📥 İndirilme: {downloads:,}"

            title_lbl = ctk.CTkLabel(card, text=title_text, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            title_lbl.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

            det_lbl = ctk.CTkLabel(card, text=details_text, font=ctk.CTkFont(size=11), text_color="gray70", justify="left", anchor="w")
            det_lbl.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            inspect_btn = ctk.CTkButton(
                btn_box, text="🔍 İncele & İndir", font=ctk.CTkFont(weight="bold"), height=34,
                command=lambda r_id=repo_id, r_type=repo_type: self.inspect_from_explore(r_id, r_type)
            )
            inspect_btn.pack(side="left", padx=4)

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

        self.cache_size_lbl = ctk.CTkLabel(self.local_top_frame, text=t("local_cache_title", size="..."), font=ctk.CTkFont(size=14, weight="bold"))
        self.cache_size_lbl.grid(row=0, column=0, padx=15, pady=10)

        self.local_search_entry = ctk.CTkEntry(self.local_top_frame, placeholder_text=t("local_search_placeholder"), height=32)
        self.local_search_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.local_search_entry.bind("<KeyRelease>", self.filter_local_models)

        self.refresh_local_btn = ctk.CTkButton(self.local_top_frame, text=t("refresh"), width=90, height=32, command=self.refresh_local_models_thread)
        self.refresh_local_btn.grid(row=0, column=2, padx=(0, 15), pady=10)

        self.local_scroll_frame = ctk.CTkScrollableFrame(self.page_local, fg_color="transparent")
        self.local_scroll_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.local_scroll_frame.grid_columnconfigure(0, weight=1)

    # ==========================================
    # PAGE 6: SETTINGS, DRIVE, TOKEN & LANG
    # ==========================================
    def build_page_settings(self):
        self.page_settings.grid_columnconfigure(0, weight=1)

        self.settings_card = ctk.CTkFrame(self.page_settings, corner_radius=8)
        self.settings_card.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        self.settings_card.grid_columnconfigure(1, weight=1)

        self.sett_title = ctk.CTkLabel(self.settings_card, text=t("settings_title"), font=ctk.CTkFont(size=16, weight="bold"))
        self.sett_title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # Row 1: Folder Selection
        self.sett_dir_lbl = ctk.CTkLabel(self.settings_card, text=t("settings_dir_label"), font=ctk.CTkFont(weight="bold"))
        self.sett_dir_lbl.grid(row=1, column=0, padx=15, pady=10, sticky="w")

        self.sett_dir_entry = ctk.CTkEntry(self.settings_card, height=34)
        self.sett_dir_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.sett_dir_entry.insert(0, self.settings.get("download_dir", ""))

        self.sett_dir_btn = ctk.CTkButton(self.settings_card, text=t("settings_browse"), width=100, height=34, command=self.browse_dir)
        self.sett_dir_btn.grid(row=1, column=2, padx=(0, 15), pady=10)

        # Row 2: HuggingFace API Token (For Gated Models)
        self.sett_token_lbl = ctk.CTkLabel(self.settings_card, text=t("settings_token_label"), font=ctk.CTkFont(weight="bold"))
        self.sett_token_lbl.grid(row=2, column=0, padx=15, pady=10, sticky="w")

        self.sett_token_entry = ctk.CTkEntry(self.settings_card, placeholder_text=t("settings_token_placeholder"), height=34, show="*")
        self.sett_token_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        self.sett_token_entry.insert(0, self.settings.get("hf_token", ""))

        # Row 3: Language Toggle Menu (TR / EN)
        self.sett_lang_lbl = ctk.CTkLabel(self.settings_card, text=t("settings_lang_label"), font=ctk.CTkFont(weight="bold"))
        self.sett_lang_lbl.grid(row=3, column=0, padx=15, pady=10, sticky="w")

        current_l = self.settings.get("language", "tr")
        lang_display = "Türkçe 🇹🇷" if current_l == "tr" else "English 🇬🇧"
        self.sett_lang_menu = ctk.CTkOptionMenu(
            self.settings_card, values=["Türkçe 🇹🇷", "English 🇬🇧"], height=34, width=180,
            command=self.change_language_gui
        )
        self.sett_lang_menu.set(lang_display)
        self.sett_lang_menu.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.sett_save_btn = ctk.CTkButton(
            self.settings_card, text=t("settings_save"), font=ctk.CTkFont(weight="bold"), height=36,
            fg_color="#27AE60", hover_color="#1E8449", command=self.save_settings_gui
        )
        self.sett_save_btn.grid(row=4, column=0, columnspan=3, padx=15, pady=(15, 20), sticky="ew")

    def change_language_gui(self, choice):
        new_lang = "tr" if "Türkçe" in choice else "en"
        set_lang(new_lang)
        self.settings["language"] = new_lang
        save_settings(self.settings)
        self.title(t("app_title"))
        self.btn_nav_search.configure(text=t("nav_search"))
        self.btn_nav_fav.configure(text=t("nav_favorites"))
        self.btn_nav_downloads.configure(text=t("nav_downloads"))
        self.btn_nav_ai.configure(text=t("nav_ai_chat"))
        self.btn_nav_explore.configure(text=t("nav_explore"))
        self.btn_nav_local.configure(text=t("nav_local"))
        self.btn_nav_settings.configure(text=t("nav_settings"))
        self.sb_disk_title.configure(text=t("target_drive_status"))
        self.sb_refresh_btn.configure(text=t("refresh_disk"))
        messagebox.showinfo("Language Changed", f"Language set to: {choice}")

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
            self.refresh_sidebar_disk_info()

    def save_settings_gui(self):
        self.settings["download_dir"] = self.sett_dir_entry.get().strip()
        self.settings["hf_token"] = self.sett_token_entry.get().strip()
        save_settings(self.settings)
        self.refresh_sidebar_disk_info()
        messagebox.showinfo("Kaydedildi", "Ayarlarınız başarıyla kaydedildi!")

    def start_fetch_data_thread(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Eksik Bilgi", "Lütfen bir Hugging Face URL'si veya Repo ID girin.")
            return

        self.status_label.configure(text=t("status_fetching"))
        self.progress_bar.set(0.3)
        self.analyze_btn.configure(state="disabled")

        token = self.settings.get("hf_token", "").strip()
        threading.Thread(target=self.fetch_repo_data, args=(url, token), daemon=True).start()

    def fetch_repo_data(self, url, token=None):
        repo_id, repo_type = hf_api.parse_hf_url(url)
        if not repo_id:
            self.safe_gui_update(self.on_fetch_failed, "Geçersiz Hugging Face URL formatı.")
            return

        self.current_repo_id = repo_id
        self.current_repo_type = repo_type

        details = hf_api.get_repo_details(repo_id, repo_type, token=token)
        if not details["success"]:
            self.safe_gui_update(self.on_fetch_failed, f"Depo bilgisi alınamadı: {details.get('error')}")
            return

        self.repo_files = details["files"]
        self.total_repo_size_bytes = details["total_size"]

        self.check_local_files_status()
        self.safe_gui_update(self.on_fetch_success, details, self.total_repo_size_bytes)

    def check_local_files_status(self):
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
        self.btn_run_vram_analysis.configure(state="disabled")
        self.btn_run_code_gen.configure(state="disabled")
        messagebox.showerror("Hata", error_msg)

    def on_fetch_success(self, details, total_bytes):
        self.status_label.configure(text=t("status_success"))
        self.progress_bar.set(1.0)
        self.analyze_btn.configure(state="normal")
        self.download_selected_btn.configure(state="normal")
        self.one_click_btn.configure(state="normal")
        self.star_model_btn.configure(state="normal")
        self.btn_run_ai_analysis.configure(state="normal")
        self.btn_run_vram_analysis.configure(state="normal")
        self.btn_run_code_gen.configure(state="normal")

        if is_favorite(self.current_repo_id):
            self.star_model_btn.configure(text=t("star_added"), fg_color="#B7950B")
        else:
            self.star_model_btn.configure(text=t("star_add"), fg_color="#D4AC0D")

        type_str = "Dataset" if self.current_repo_type == "dataset" else "Model"
        info_text = (
            f"📌 Repo ID: {self.current_repo_id}\n"
            f"🏷️ Tür: {type_str} | 👤 Yazar: {details['author']}\n"
            f"📦 Toplam Depo Boyutu: {storage.format_size(total_bytes)}\n"
            f"❤️ Beğeni: {details['likes']} | 📥 İndirilme: {details['downloads']}"
        )
        self.meta_info_label.configure(text=info_text)

        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert("0.0", "⚡ Model verileri getirildi. Yukarıdaki butonlardan özet, vRAM analizi ve kod üretebilirsiniz.")
        self.summary_text.configure(state="disabled")

        self.one_click_btn.configure(text=f"⚡ Tek Tıkla Tüm Modeli İndir ({storage.format_size(total_bytes)})")

        self.clear_file_list()
        self.checkbox_vars.clear()
        self.populate_file_list(self.repo_files)
        self.select_smart_defaults()

    def start_manual_ai_analysis_thread(self):
        if not self.current_repo_id: return
        self.btn_run_ai_analysis.configure(state="disabled")
        self.update_summary_textbox("⏳ Cortex AI model açıklamasını okuyor ve Türkçe özet hazırlıyor...")
        token = self.settings.get("hf_token", "").strip()
        threading.Thread(target=self.process_manual_ai_analysis, args=(token,), daemon=True).start()

    def process_manual_ai_analysis(self, token):
        readme = hf_api.scrape_readme(self.current_repo_id, self.current_repo_type, token=token)
        self.current_readme_content = readme
        summary = ai_summarizer.generate_summary(readme, self.current_repo_id, lang=get_lang())
        self.safe_gui_update(self.on_ai_action_complete, self.btn_run_ai_analysis, summary)

    def start_vram_analysis_thread(self):
        if not self.current_repo_id: return
        self.btn_run_vram_analysis.configure(state="disabled")
        self.update_summary_textbox("⏳ Cortex AI model yapısını ve vRAM gereksinimlerini hesaplıyor...")
        token = self.settings.get("hf_token", "").strip()
        threading.Thread(target=self.process_vram_analysis, args=(token,), daemon=True).start()

    def process_vram_analysis(self, token):
        readme = hf_api.scrape_readme(self.current_repo_id, self.current_repo_type, token=token)
        res = ai_summarizer.analyze_vram_hardware(readme, self.current_repo_id, lang=get_lang())
        self.safe_gui_update(self.on_ai_action_complete, self.btn_run_vram_analysis, res)

    def start_code_gen_thread(self):
        if not self.current_repo_id: return
        self.btn_run_code_gen.configure(state="disabled")
        self.update_summary_textbox("⏳ Cortex AI çalıştırmaya hazır Python kodu ve örnek promptlar üretiyor...")
        token = self.settings.get("hf_token", "").strip()
        threading.Thread(target=self.process_code_gen, args=(token,), daemon=True).start()

    def process_code_gen(self, token):
        readme = hf_api.scrape_readme(self.current_repo_id, self.current_repo_type, token=token)
        code_res = ai_summarizer.generate_usage_code(readme, self.current_repo_id, lang=get_lang())
        self.safe_gui_update(self.on_ai_action_complete, self.btn_run_code_gen, code_res)

    def update_summary_textbox(self, text):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert("0.0", text)
        self.summary_text.configure(state="disabled")

    def on_ai_action_complete(self, btn_widget, text_result):
        btn_widget.configure(state="normal")
        cleaned = clean_markdown_text(text_result)
        self.update_summary_textbox(cleaned)

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
                self.files_scroll_frame, text=cb_text, variable=var, onvalue="on", offvalue="off",
                font=ctk.CTkFont(size=12), text_color=text_color, command=self.update_selected_size_label
            )
            cb.pack(fill="x", anchor="w", padx=5, pady=3)
            self.checkbox_widgets.append(cb)

    def select_smart_defaults(self):
        if not self.repo_files: return

        def is_useless_asset(path):
            p = path.lower()
            if any(p.startswith(x) for x in ["assets/", "demo/", "media/", "images/", "sample_images/", "eval/"]): return True
            if any(p.endswith(ext) for ext in [".gif", ".mp4", ".png", ".jpg", ".jpeg"]): return True
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
                        for sf in shards: selected_paths.add(sf["path"])
                    else:
                        def version_key(f):
                            v_match = re.search(r'v?(\d+(\.\d+)+)', f["path"])
                            if v_match:
                                try: return [int(p) for p in v_match.group(1).split('.')]
                                except Exception: pass
                            return [0, f["size"]]
                        candidates.sort(key=version_key, reverse=True)
                        if candidates: selected_paths.add(candidates[0]["path"])

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
        if not self.repo_files: return
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
            self.download_selected_btn.configure(text=f"📥 Seçilen Eksik Dosyaları İndir ({selected_str})", state="normal")
        else:
            self.download_selected_btn.configure(text="📥 Seçilen Dosyaları İndir (0 B)", state="disabled")

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
            cb_text = f"{f['path']}  ({size_str})   [✓ İndirildi]" if is_dl else f"{f['path']}  ({size_str})"
            text_color = "#2ECC71" if is_dl else ("gray10", "gray90")

            cb = ctk.CTkCheckBox(
                self.files_scroll_frame, text=cb_text, variable=var, onvalue="on", offvalue="off",
                font=ctk.CTkFont(size=12), text_color=text_color, command=self.update_selected_size_label
            )
            cb.pack(fill="x", anchor="w", padx=5, pady=3)
            self.checkbox_widgets.append(cb)

        self.update_selected_size_label()

    def toggle_select_all(self):
        val = self.select_all_var.get()
        for var in self.checkbox_vars.values(): var.set(val)
        self.update_selected_size_label()

    def clear_all_selections(self):
        for var in self.checkbox_vars.values(): var.set("off")
        self.select_all_var.set("off")
        self.update_selected_size_label()

    # --- DOWNLOAD ENGINE INTEGRATION WITH QUEUE MANAGER ---
    def start_one_click_download_thread(self):
        if not self.current_repo_id: return
        if not self.validate_download_destination(required_bytes=self.total_repo_size_bytes): return

        target_dir = self.settings.get("download_dir", "").strip()
        token = self.settings.get("hf_token", "").strip()

        self.download_queue.add_task(
            repo_id=self.current_repo_id,
            repo_type=self.current_repo_type,
            mode="snapshot",
            total_bytes=self.total_repo_size_bytes,
            target_dir=target_dir,
            token=token
        )

        messagebox.showinfo("İndirme Sırasına Eklendi", t("added_to_queue", repo=self.current_repo_id))
        self.show_page("downloads")

    def start_download_selected_thread(self):
        selected_files = [path for path, var in self.checkbox_vars.items() if var.get() == "on"]
        if not selected_files:
            messagebox.showwarning("Seçim Yapılmadı", "Lütfen en az bir dosya seçin.")
            return

        req_bytes = sum(f["size"] for f in self.repo_files if f["path"] in selected_files)
        if not self.validate_download_destination(required_bytes=req_bytes): return

        target_dir = self.settings.get("download_dir", "").strip()
        token = self.settings.get("hf_token", "").strip()

        selected_files_info = [f for f in self.repo_files if f["path"] in selected_files]

        self.download_queue.add_task(
            repo_id=self.current_repo_id,
            repo_type=self.current_repo_type,
            mode="selected",
            selected_files=selected_files_info,
            total_bytes=req_bytes,
            target_dir=target_dir,
            token=token
        )

        messagebox.showinfo("İndirme Sırasına Eklendi", t("added_to_queue", repo=self.current_repo_id))
        self.show_page("downloads")

    # --- LOCAL MODELS LOGIC ---
    def refresh_local_models_thread(self):
        threading.Thread(target=self.refresh_local_models, daemon=True).start()

    def refresh_local_models(self):
        target_dir = self.settings.get("download_dir", "")
        res = storage.scan_local_models(cache_dir=target_dir if target_dir else None)
        self.local_models_data = res["models"]
        self.safe_gui_update(self.update_local_models_ui, res["total_cache_size_str"])

    def update_local_models_ui(self, total_cache_str):
        self.cache_size_lbl.configure(text=t("local_cache_title", size=total_cache_str))
        self.render_local_models_list(self.local_models_data)

    def render_local_models_list(self, models_list):
        for widget in self.local_scroll_frame.winfo_children(): widget.destroy()

        if not models_list:
            lbl = ctk.CTkLabel(self.local_scroll_frame, text="Henüz indirilmiş yerel bir model bulunamadı.", font=ctk.CTkFont(size=14), text_color="gray60")
            lbl.pack(pady=30)
            return

        for item in models_list:
            card = ctk.CTkFrame(self.local_scroll_frame, corner_radius=8, fg_color=("gray85", "gray20"))
            card.pack(fill="x", padx=5, pady=5)
            card.grid_columnconfigure(0, weight=1)

            repo_id, repo_type, size_str, local_path = item["repo_id"], item["repo_type"].upper(), item["size_str"], item["local_path"]
            title_str = f"📦 {repo_id}  [{repo_type}]"
            sub_str = f"Boyut: {size_str}  |  Konum: {local_path}"

            lbl_title = ctk.CTkLabel(card, text=title_str, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            lbl_title.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

            lbl_sub = ctk.CTkLabel(card, text=sub_str, font=ctk.CTkFont(size=11), text_color="gray70", anchor="w", justify="left")
            lbl_sub.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.grid(row=0, column=1, rowspan=2, padx=12, pady=10, sticky="e")

            open_btn = ctk.CTkButton(btn_frame, text=t("open_folder"), width=100, height=30, command=lambda path=local_path: self.open_folder(path))
            open_btn.pack(side="left", padx=5)

            del_btn = ctk.CTkButton(btn_frame, text=t("delete_model"), width=90, height=30, fg_color="#C0392B", hover_color="#922B21", command=lambda itm=item: self.delete_model_action(itm))
            del_btn.pack(side="left", padx=5)

    def filter_local_models(self, event=None):
        query = self.local_search_entry.get().strip().lower()
        filtered = [m for m in self.local_models_data if query in m["repo_id"].lower() or query in m["local_path"].lower()]
        self.render_local_models_list(filtered)

    def open_folder(self, path):
        if os.path.exists(path): os.startfile(path)
        else: messagebox.showerror("Hata", "Klasör bulunamadı.")

    def delete_model_action(self, item):
        confirm = messagebox.askyesno("Silme Onayı", f"'{item['repo_id']}' silinecek. Emin misiniz?\nBoyut: {item['size_str']}")
        if confirm:
            target_dir = self.settings.get("download_dir", "")
            res = storage.delete_local_model(item, cache_dir=target_dir if target_dir else None)
            if res["success"]:
                messagebox.showinfo("Silindi", "Model başarıyla silindi.")
                self.refresh_local_models_thread()
                self.refresh_sidebar_disk_info()
            else:
                messagebox.showerror("Hata", f"Silme hatası: {res.get('error')}")

    def validate_download_destination(self, required_bytes=0):
        target_dir = self.settings.get("download_dir", "").strip()
        is_accessible, drive_letter, access_msg = storage.check_drive_accessibility(target_dir)
        if not is_accessible:
            messagebox.showerror("Sürücü Hata", access_msg)
            return False
        if required_bytes > 0:
            is_sufficient, free_bytes, free_str, space_msg = storage.check_sufficient_space(target_dir, required_bytes)
            if not is_sufficient:
                messagebox.showwarning("Yetersiz Disk", space_msg)
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
            self.sb_disk_details_lbl.configure(text="Sürücü erişilemiyor!")

if __name__ == "__main__":
    app = HFDownloaderApp()
    app.mainloop()
