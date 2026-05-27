from huggingface_hub import snapshot_download

MODEL_ID = "dicta-il/dictalm2.0"
LOCAL_DIR = "/mnt/ssd2/cyttic/models/dictalm2"

print(f"Downloading {MODEL_ID}...")
snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
print(f"Done. Model saved to {LOCAL_DIR}")
