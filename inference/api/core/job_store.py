import asyncio
import json
import os
import threading

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "job_results.json")
_file_lock = threading.Lock()


def _read_all():
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _write_all(data: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f)


class JobStore:
    def __init__(self):
        self.queues = {}

    def get_queue(self, job_id: str):
        if job_id not in self.queues:
            self.queues[job_id] = asyncio.Queue()
        with _file_lock:
            data = _read_all()
            if job_id not in data:
                data[job_id] = []
                _write_all(data)
        return self.queues[job_id]

    def save_action(self, job_id: str, action_data: dict):
        with _file_lock:
            data = _read_all()
            if job_id not in data:
                data[job_id] = []
            data[job_id].append(action_data)
            _write_all(data)

    def get_results(self, job_id: str):
        with _file_lock:
            data = _read_all()
        results = data.get(job_id, [])
        return results

    def delete_job(self, job_id: str):
        if job_id in self.queues:
            del self.queues[job_id]
        with _file_lock:
            data = _read_all()
            if job_id in data:
                del data[job_id]
                _write_all(data)


job_store = JobStore()
