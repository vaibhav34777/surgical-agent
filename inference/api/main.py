from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import analyze, stream, media
from core.config import settings
from models.schemas import HealthResponse
import os
import google.generativeai as genai

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix=settings.API_V1_STR, tags=["analyze"])
app.include_router(stream.router, prefix=settings.API_V1_STR, tags=["stream"])
app.include_router(media.router, prefix=settings.API_V1_STR, tags=["media"])

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        llm_status = True
    except:
        llm_status = False
    return HealthResponse(status="ok", llm_active=llm_status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
