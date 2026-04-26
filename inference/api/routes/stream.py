from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from core.job_store import job_store
import json

router = APIRouter()

@router.get("/stream/{job_id}")
async def stream_job(job_id: str):
    async def event_generator():
        queue = job_store.get_queue(job_id)
        while True:
            data = await queue.get()
            yield {
                "event": data["event"],
                "data": json.dumps(data["data"])
            }
            if data["event"] in ["end", "error"]:
                job_store.delete_job(job_id)
                break
    
    return EventSourceResponse(event_generator())
