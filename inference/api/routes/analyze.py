from fastapi import APIRouter, UploadFile, File, BackgroundTasks
import uuid
import os
from core.config import settings
from models.schemas import JobResponse
from services.orchestrator import run_pipeline

router = APIRouter()

@router.post("/analyze", response_model=JobResponse)
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    background_tasks.add_task(run_pipeline, job_id, file_path)
    return JobResponse(job_id=job_id, status="started")
