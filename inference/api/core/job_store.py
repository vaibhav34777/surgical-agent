import asyncio
import json
import os

class JobStore:
    def __init__(self, storage_path="job_results.json"):
        self.queues = {}
        self.results = {}
        self.storage_path = storage_path
        self._load_results()

    def _load_results(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    self.results = json.load(f)
        except:
            self.results = {}

    def _save_results(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.results, f)
        except:
            pass

    def get_queue(self, job_id: str):
        if job_id not in self.queues:
            self.queues[job_id] = asyncio.Queue()
        if job_id not in self.results:
            self.results[job_id] = []
            self._save_results()
        return self.queues[job_id]

    def save_action(self, job_id: str, data: dict):
        if job_id in self.results:
            self.results[job_id].append(data)
            self._save_results()
            
    def get_results(self, job_id: str):
        if job_id not in self.results:
            self._load_results()
        return self.results.get(job_id, [])

    def delete_job(self, job_id: str):
        if job_id in self.queues:
            del self.queues[job_id]
        if job_id in self.results:
            del self.results[job_id]
            self._save_results()

job_store = JobStore()
