# src/utils/progress.py
import threading

# Global progress store
_progress_store = {}
_lock = threading.Lock()

def update_progress(task_id: str, percentage: float, message: str = "", status: str = None):
    """Updates the progress for a specific task."""
    with _lock:
        if status is None:
            status = "in_progress" if percentage < 100 else "completed"
        _progress_store[task_id] = {
            "percentage": round(percentage, 2),
            "message": message,
            "status": status
        }

def get_progress(task_id: str):
    """Retrieves the progress for a specific task."""
    with _lock:
        return _progress_store.get(task_id, {"percentage": 0, "message": "Initialized", "status": "pending"})

def reset_task(task_id: str):
    """Resets the task progress."""
    with _lock:
        if task_id in _progress_store:
            del _progress_store[task_id]
