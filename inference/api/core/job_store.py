import asyncio

class JobStore:
    def __init__(self):
        self.queues = {}

    def get_queue(self, job_id: str):
        if job_id not in self.queues:
            self.queues[job_id] = asyncio.Queue()
        return self.queues[job_id]

    def delete_job(self, job_id: str):
        if job_id in self.queues:
            del self.queues[job_id]

job_store = JobStore()
