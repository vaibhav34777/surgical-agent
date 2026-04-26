from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "imvaibhavrana/surgical-agent-models"
REPO_TYPE = "model"

api = HfApi()

create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

files_to_upload = [
    {
        "local_path": os.path.join(base_dir, "segmentation_model", "segformer_b2_quantized.onnx"),
        "repo_path": "segmentation_model/segformer_b2_quantized.onnx",
    },
    {
        "local_path": os.path.join(base_dir, "video_transformer", "rendezvous.pt"),
        "repo_path": "video_transformer/rendezvous.pt",
    },
]

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file["local_path"],
        path_in_repo=file["repo_path"],
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"Uploaded: {file['repo_path']}")

print(f"All models uploaded to https://huggingface.co/{REPO_ID}")
