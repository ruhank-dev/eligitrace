"""`task download-weights`: base + draft models and the EligiTrace LoRA adapter from Hugging Face."""

from huggingface_hub import snapshot_download

from settings.config import get_settings


def main() -> None:
    s = get_settings()
    for repo in (s.target_model_id, s.draft_model_id):
        print(f"base model  : {repo}")
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
    print(f"LoRA adapter: {s.adapter_repo_id} -> {s.adapter_local_dir}")
    snapshot_download(s.adapter_repo_id, local_dir=s.adapter_local_dir)
    print("done. Start the app with: task app")


if __name__ == "__main__":
    main()
