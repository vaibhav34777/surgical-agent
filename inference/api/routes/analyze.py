from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import sys
from core.config import settings
from models.schemas import JobResponse
from services.orchestrator import run_pipeline
from core.job_store import job_store

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from agent.pdf_generator import generate_surgical_report

router = APIRouter()

@router.post("/analyze", response_model=JobResponse)
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    background_tasks.add_task(run_pipeline, job_id, file_path)
    return JobResponse(job_id=job_id, status="started")

@router.get("/report/{job_id}")
async def get_report(job_id: str):
    results = job_store.get_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this job ID")
    
    report_path = os.path.join(settings.MEDIA_DIR, f"{job_id}_report.pdf")
    global_summary = f"Automated Surgical Analysis for Job: {job_id}"
    
    generate_surgical_report(report_path, global_summary, results)
    
    return FileResponse(report_path, media_type="application/pdf", filename=f"Surgical_Report_{job_id}.pdf")
