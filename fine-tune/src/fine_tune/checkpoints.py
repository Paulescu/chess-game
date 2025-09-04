from pathlib import Path


def check_for_existing_checkpoint(paths: dict):
    """
    Check if there's an existing checkpoint to resume training from.

    This enables resumable training, which is crucial for long-running experiments
    that might be interrupted by infrastructure issues or resource limits.
    """
    checkpoint_dir = paths["checkpoints"]
    if not checkpoint_dir.exists():
        return None

    # Look for the most recent checkpoint directory
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    return None


def _get_or_create_path_to_model_checkpoints(
    wandb_experiment_name: str,
) -> Path:
    """
    Returns path to the cached dataset in a Modal volume.
    """
    path = Path("/model_checkpoints") / wandb_experiment_name.replace("/", "--")

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path
