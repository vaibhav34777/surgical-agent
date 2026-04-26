import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

HF_REPO_ID = "imvaibhavrana/surgical-agent-models"

class Settings:
    PROJECT_NAME: str = "Surgical Agent API"
    API_V1_STR: str = "/api/v1"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "../uploads")
    MEDIA_DIR: str = os.path.join(os.path.dirname(__file__), "../media")

    SEG_MODEL_PATH: str = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="segmentation_model/segformer_b2_quantized.onnx",
    )
    RENDEZVOUS_MODEL_PATH: str = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="video_transformer/rendezvous.pt",
    )

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MEDIA_DIR, exist_ok=True)
