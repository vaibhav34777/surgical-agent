from fastapi import APIRouter
from fastapi.responses import FileResponse
import os
from core.config import settings

router = APIRouter()

@router.get("/media/{filename}")
async def get_media(filename: str):
    path = os.path.join(settings.MEDIA_DIR, filename)
    return FileResponse(path)
