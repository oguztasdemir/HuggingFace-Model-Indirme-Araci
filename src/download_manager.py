import os
import time
import threading
from collections import deque
from huggingface_hub import hf_hub_download, snapshot_download
from src import storage

class DownloadTask:
    def __init__(self, task_id, repo_id, repo_type="model", mode="selected", selected_files=None, total_bytes=0, target_dir="", token=None):
        self.task_id = task_id
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.mode = mode  # "selected" or "snapshot"
        self.selected_files = selected_files or []  # list of {"path": ..., "size": ...}
        self.total_bytes = total_bytes
        self.target_dir = target_dir
        self.token = token

        self.status = "queued"  # "queued", "downloading", "paused", "completed", "failed", "cancelled"
        self.downloaded_bytes = 0
        self.current_speed_bps = 0.0
        self.current_speed_mbps = 0.0
        self.peak_speed_mbps = 0.0
        self.speed_history = deque(maxlen=60)  # Stores last 60 speed samples in Mbps
        self.start_time = None
        self.elapsed_seconds = 0.0
        self.eta_seconds = 0.0
        self.error_message = ""
        self.cancel_requested = False
        self.pause_requested = False

        # Pre-fill initial speed graph history with zeros
        for _ in range(60):
            self.speed_history.append(0.0)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "mode": self.mode,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "current_speed_mbps": self.current_speed_mbps,
            "peak_speed_mbps": self.peak_speed_mbps,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "error_message": self.error_message
        }


class DownloadQueueManager:
    def __init__(self):
        self.tasks = []
        self.lock = threading.Lock()
        self.active_task = None
        self.worker_thread = None
        self.running = True
        self._start_worker()

    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._queue_worker_loop, daemon=True)
        self.worker_thread.start()

    def add_task(self, repo_id, repo_type="model", mode="selected", selected_files=None, total_bytes=0, target_dir="", token=None):
        with self.lock:
            task_id = f"task_{int(time.time() * 1000)}"
            task = DownloadTask(
                task_id=task_id,
                repo_id=repo_id,
                repo_type=repo_type,
                mode=mode,
                selected_files=selected_files,
                total_bytes=total_bytes,
                target_dir=target_dir,
                token=token
            )
            self.tasks.append(task)
            return task

    def get_tasks(self):
        with self.lock:
            return list(self.tasks)

    def get_active_task(self):
        with self.lock:
            return self.active_task

    def pause_task(self, task_id):
        with self.lock:
            for task in self.tasks:
                if task.task_id == task_id:
                    if task.status == "downloading":
                        task.pause_requested = True
                    elif task.status == "queued":
                        task.status = "paused"
                    break

    def resume_task(self, task_id):
        with self.lock:
            for task in self.tasks:
                if task.task_id == task_id and task.status == "paused":
                    task.status = "queued"
                    task.pause_requested = False
                    break

    def cancel_task(self, task_id):
        with self.lock:
            for task in self.tasks:
                if task.task_id == task_id:
                    if task.status == "downloading":
                        task.cancel_requested = True
                    else:
                        task.status = "cancelled"
                    break

    def move_task_up(self, task_id):
        with self.lock:
            idx = next((i for i, t in enumerate(self.tasks) if t.task_id == task_id), -1)
            if idx > 0 and self.tasks[idx].status in ["queued", "paused"]:
                if self.tasks[idx - 1].status in ["queued", "paused"]:
                    self.tasks[idx], self.tasks[idx - 1] = self.tasks[idx - 1], self.tasks[idx]

    def move_task_down(self, task_id):
        with self.lock:
            idx = next((i for i, t in enumerate(self.tasks) if t.task_id == task_id), -1)
            if 0 <= idx < len(self.tasks) - 1 and self.tasks[idx].status in ["queued", "paused"]:
                if self.tasks[idx + 1].status in ["queued", "paused"]:
                    self.tasks[idx], self.tasks[idx + 1] = self.tasks[idx + 1], self.tasks[idx]

    def clear_finished(self):
        with self.lock:
            self.tasks = [t for t in self.tasks if t.status not in ["completed", "cancelled", "failed"]]

    def _queue_worker_loop(self):
        while self.running:
            next_task = None
            with self.lock:
                for task in self.tasks:
                    if task.status == "queued":
                        next_task = task
                        break

            if next_task:
                self._run_task(next_task)
            else:
                time.sleep(0.5)

    def _run_task(self, task):
        with self.lock:
            task.status = "downloading"
            task.pause_requested = False
            task.cancel_requested = False
            self.active_task = task

        safe_folder = task.repo_id.replace("/", "--")
        project_dir = os.path.join(task.target_dir, f"{task.repo_type}s--{safe_folder}")
        os.makedirs(project_dir, exist_ok=True)

        task.start_time = time.time()
        initial_elapsed = task.elapsed_seconds

        try:
            if task.mode == "snapshot":
                snapshot_download(
                    repo_id=task.repo_id,
                    repo_type=task.repo_type,
                    local_dir=project_dir,
                    local_dir_use_symlinks=False,
                    token=task.token if task.token else None
                )
                task.downloaded_bytes = task.total_bytes
                task.status = "completed"
            else:
                files_to_dl = task.selected_files
                for f_info in files_to_dl:
                    if task.cancel_requested or task.pause_requested:
                        break

                    f_path = f_info["path"]
                    f_size = f_info.get("size", 0)

                    local_fp = os.path.join(project_dir, f_path)
                    if os.path.exists(local_fp) and os.path.getsize(local_fp) == f_size:
                        task.downloaded_bytes += f_size
                        continue

                    f_start_time = time.time()
                    hf_hub_download(
                        repo_id=task.repo_id,
                        filename=f_path,
                        repo_type=task.repo_type,
                        local_dir=project_dir,
                        local_dir_use_symlinks=False,
                        token=task.token if task.token else None
                    )
                    task.downloaded_bytes += f_size

                    current_active_time = (time.time() - task.start_time)
                    task.elapsed_seconds = initial_elapsed + current_active_time
                    
                    if current_active_time > 0:
                        task.current_speed_bps = task.downloaded_bytes / current_active_time
                        task.current_speed_mbps = (task.current_speed_bps * 8) / 1000000.0
                    else:
                        task.current_speed_mbps = 0.0

                    if task.current_speed_mbps > task.peak_speed_mbps:
                        task.peak_speed_mbps = task.current_speed_mbps

                    task.speed_history.append(task.current_speed_mbps)

                    rem_bytes = max(task.total_bytes - task.downloaded_bytes, 0)
                    if task.current_speed_bps > 1024:
                        task.eta_seconds = rem_bytes / task.current_speed_bps
                    else:
                        task.eta_seconds = 0.0

            if task.cancel_requested:
                task.status = "cancelled"
            elif task.pause_requested:
                task.status = "paused"
            else:
                task.status = "completed"
                task.downloaded_bytes = task.total_bytes
                task.current_speed_mbps = 0.0
                task.speed_history.append(0.0)

        except Exception as e:
            if task.cancel_requested:
                task.status = "cancelled"
            elif task.pause_requested:
                task.status = "paused"
            else:
                task.status = "failed"
                task.error_message = str(e)
        finally:
            with self.lock:
                self.active_task = None
