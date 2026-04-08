from huggingface_hub import HfApi
import os

api = HfApi()

print("Uploading to tarun8477/truth_seeker_env using native HfApi...")
api.upload_folder(
    folder_path=".",
    repo_id="tarun8477/truth_seeker_env",
    repo_type="space",
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=[
        "env",
        "env/*",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "uv.lock",
        ".git",
        "*.zip",
        "*.docx",
        "Requirements/*"
    ]
)
print("Push complete! Check https://huggingface.co/spaces/tarun8477/truth_seeker_env")
