import os
import shutil
from huggingface_hub import scan_cache_dir

def format_size(size_bytes):
    """Formats bytes into human readable format (KB, MB, GB, TB)."""
    if not size_bytes or size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"

def scan_huggingface_dirs():
    """
    Scans available drives to find or suggest a huggingface folder.
    Prioritizes E: drive on Windows if available, otherwise defaults to user home cache.
    Returns a dict with 'found_paths', 'default_path', and 'available_drives'.
    """
    drives = []
    if os.name == 'nt':
        try:
            import string
            from ctypes import windll
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(f"{letter}:\\")
                bitmask >>= 1
        except Exception:
            drives = ["C:\\"]
    else:
        drives = ["/"]

    found_paths = []
    e_drive_hf = None

    for drive in drives:
        paths_to_check = [
            os.path.join(drive, "huggingface"),
            os.path.join(drive, ".cache", "huggingface"),
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                found_paths.append(path)
                if drive.upper().startswith("E:"):
                    e_drive_hf = path

    user_home = os.path.expanduser("~")
    home_cache = os.path.join(user_home, ".cache", "huggingface")
    if os.path.exists(home_cache):
        found_paths.append(home_cache)

    if e_drive_hf:
        default_path = e_drive_hf
    elif found_paths:
        default_path = found_paths[0]
    elif "E:\\" in drives:
        default_path = "E:\\huggingface"
    else:
        default_path = home_cache

    return {
        "found_paths": list(set(found_paths)),
        "default_path": default_path,
        "drives": drives
    }

def get_disk_info(path=None):
    """
    Returns disk usage statistics for the given path or drive.
    """
    if not path or not os.path.exists(path):
        path = os.path.expanduser("~")

    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        percent_used = (usage.used / usage.total) * 100 if usage.total > 0 else 0
        return {
            "path": path,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "total_str": format_size(usage.total),
            "used_str": format_size(usage.used),
            "free_str": format_size(usage.free),
            "percent_used": round(percent_used, 1),
            "success": True
        }
    except Exception as e:
        return {
            "path": path,
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "total_str": "N/A",
            "used_str": "N/A",
            "free_str": "N/A",
            "percent_used": 0,
            "success": False,
            "error": str(e)
        }

def scan_local_models(cache_dir=None):
    """
    Scans Hugging Face local cache directory using scan_cache_dir
    AND scans custom project subfolders in download directory.
    """
    models = []
    total_cache_size = 0
    scanned_paths = set()

    # 1. Standard scan_cache_dir
    try:
        if cache_dir and os.path.exists(cache_dir):
            cache_info = scan_cache_dir(cache_dir=cache_dir)
        else:
            cache_info = scan_cache_dir()

        total_cache_size += cache_info.size_on_disk

        for repo in cache_info.repos:
            repo_id = repo.repo_id
            repo_type = repo.repo_type
            size_on_disk = repo.size_on_disk
            revisions = len(repo.revisions)
            nb_files = sum(len(rev.files) for rev in repo.revisions)
            local_path = str(repo.repo_path)
            scanned_paths.add(os.path.normpath(local_path))

            models.append({
                "repo_id": repo_id,
                "repo_type": repo_type,
                "size_bytes": size_on_disk,
                "size_str": format_size(size_on_disk),
                "revisions_count": revisions,
                "files_count": nb_files,
                "local_path": local_path,
                "last_modified": repo.last_modified,
                "repo_object": repo
            })

    except Exception as e:
        print(f"Info scan_cache_dir: {e}")

    # 2. Scan custom project subfolders (models--author--name)
    if cache_dir and os.path.exists(cache_dir):
        try:
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                norm_item_path = os.path.normpath(item_path)

                if norm_item_path in scanned_paths or item.startswith('.') or item == "hub":
                    continue

                if os.path.isdir(item_path):
                    dir_size = 0
                    file_cnt = 0
                    for root, _, files in os.walk(item_path):
                        for f in files:
                            fp = os.path.join(root, f)
                            if os.path.exists(fp):
                                dir_size += os.path.getsize(fp)
                                file_cnt += 1

                    if file_cnt > 0:
                        clean_repo_id = item
                        repo_type = "model"
                        if clean_repo_id.startswith("models--"):
                            clean_repo_id = clean_repo_id[len("models--"):].replace("--", "/")
                            repo_type = "model"
                        elif clean_repo_id.startswith("datasets--"):
                            clean_repo_id = clean_repo_id[len("datasets--"):].replace("--", "/")
                            repo_type = "dataset"
                        else:
                            clean_repo_id = clean_repo_id.replace("--", "/")

                        total_cache_size += dir_size
                        scanned_paths.add(norm_item_path)
                        models.append({
                            "repo_id": clean_repo_id,
                            "repo_type": repo_type,
                            "size_bytes": dir_size,
                            "size_str": format_size(dir_size),
                            "revisions_count": 1,
                            "files_count": file_cnt,
                            "local_path": item_path,
                            "last_modified": int(os.path.getmtime(item_path)),
                            "repo_object": None
                        })
        except Exception as e:
            print(f"Error scanning custom project subfolders: {e}")

    return {
        "models": models,
        "total_cache_size_bytes": total_cache_size,
        "total_cache_size_str": format_size(total_cache_size)
    }

def delete_local_model(repo_item, cache_dir=None):
    """
    Deletes a local model from Hugging Face cache using scan_cache_dir revision deletion or directory removal.
    """
    try:
        if repo_item.get("repo_object"):
            repo = repo_item["repo_object"]
            revision_hashes = [rev.commit_hash for rev in repo.revisions]
            if cache_dir and os.path.exists(cache_dir):
                cache_info = scan_cache_dir(cache_dir=cache_dir)
            else:
                cache_info = scan_cache_dir()
            delete_strategy = cache_info.delete_revisions(*revision_hashes)
            delete_strategy.execute()
            return {"success": True, "freed_bytes": delete_strategy.freed_size}
        elif repo_item.get("local_path") and os.path.exists(repo_item["local_path"]):
            shutil.rmtree(repo_item["local_path"])
            return {"success": True, "freed_bytes": repo_item.get("size_bytes", 0)}
        return {"success": False, "error": "Model konumu bulunamadı."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_drive_accessibility(target_path):
    """
    Checks if the target path or drive letter exists and is accessible/writable.
    """
    if not target_path:
        return False, "", "İndirme klasörü yolu boş."

    drive, _ = os.path.splitdrive(target_path)
    drive_letter = drive.upper() if drive else ""

    if drive_letter and not os.path.exists(f"{drive_letter}\\"):
        return False, drive_letter, f"Seçilen {drive_letter} sürücüsü takılı değil veya erişilebilir değil!\nLütfen diski takın veya farklı bir indirme konumu seçin."

    try:
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        return True, drive_letter, "Sürücü erişilebilir."
    except Exception as e:
        return False, drive_letter, f"Sürücüye veya klasöre erişilemiyor ({target_path}):\n{e}"

def check_sufficient_space(target_path, required_bytes):
    """
    Checks if the drive containing target_path has at least required_bytes free space.
    """
    if not target_path:
        return True, 0, "Bilinmiyor", "Yol boş."

    drive, _ = os.path.splitdrive(target_path)
    check_path = f"{drive}\\" if drive and os.path.exists(f"{drive}\\") else os.path.expanduser("~")
    if os.path.exists(target_path):
        check_path = target_path

    try:
        usage = shutil.disk_usage(check_path)
        free_bytes = usage.free
        free_str = format_size(free_bytes)
        req_str = format_size(required_bytes)

        if required_bytes > 0 and free_bytes < required_bytes:
            msg = f"⚠️ Yetersiz Disk Alanı!\n\nİndirilecek Boyut: {req_str}\nHedef Sürücüdeki Boş Alan: {free_str}\n\nLütfen yer açın veya başka bir sürücü seçin."
            return False, free_bytes, free_str, msg

        return True, free_bytes, free_str, "Disk alanı yeterli."
    except Exception as e:
        return True, 0, "Bilinmiyor", f"Disk alanı kontrolü yapılamadı: {e}"
