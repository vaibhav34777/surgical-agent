from pydantic import BaseModel
from typing import List, Optional

class JobResponse(BaseModel):
    job_id: str
    status: str

class HealthResponse(BaseModel):
    status: str
    llm_active: bool
