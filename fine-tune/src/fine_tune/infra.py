"""
Here we define the infrastructure for our training jobs.

- Modal app object
- Docker image with all Python dependencies baked in
- Persistent volumes for pretrained models and datasets downloaded from HF
- Persistent volume for model checkpoints generated during training
- Secrets for API keys and credentials
- Retry policies for failed tasks
"""
import modal

from .config import TrainingJobConfig

config = TrainingJobConfig()

def get_modal_app() -> modal.App:
    """
    Returns the Modal application object.
    """
    return modal.App(config.modal_app_name)

def get_docker_image() -> modal.Image:
    """
    Returns a Modal Docker image with all the required Python dependencies installed.
    """
    docker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "accelerate==1.9.0",
            "datasets==3.6.0",
            "hf-transfer==0.1.9",
            "huggingface_hub==0.34.2",
            "peft==0.16.0",
            "transformers==4.54.0",
            "trl==0.19.1",
            "unsloth[cu128-torch270]==2025.7.8",
            "unsloth_zoo==2025.7.10",
            "wandb==0.21.0",
            "torch==2.7.0",
            "pydantic-settings==2.10.1",

            # TODO: not sure if I need this or not
            # "causal-conv1d==1.5.0.post8"
        )
        # .add_local_python_source(".")
        .env({"HF_HOME": "/model_cache"})
    )

    with docker_image.imports():
        # unsloth must be first!
        import unsloth  # noqa: F401,I001
        import datasets
        import torch
        import wandb
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import standardize_sharegpt

    return docker_image

def get_docker_image_for_evaluation() -> modal.Image:
    """
    Returns a Modal Docker image with all the required Python dependencies installed.
    """
    docker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "datasets==3.6.0",
            "hf-transfer==0.1.9",
            "huggingface_hub==0.34.2",
            "peft==0.16.0",
            "transformers==4.54.0",
            "wandb==0.21.0",
            "torch==2.7.0",
            "pydantic-settings==2.10.1",
            "chess==1.11.2",
        )
        # .add_local_python_source(".")
        .env({"HF_HOME": "/model_cache"})
    )

    with docker_image.imports():
        # unsloth must be first!
        import datasets
        import torch
        import wandb
        from transformers import TrainingArguments

    return docker_image


def get_volume(name: str) -> modal.Volume:
    """
    Returns a Modal volume object for the given name.
    """
    return modal.Volume.from_name(name, create_if_missing=True)

def get_secrets() -> list[modal.Secret]:
    """
    Returns the Weights & Biases secret.
    """
    wandb_secret = modal.Secret.from_name("wandb-secret")
    return [wandb_secret]

def get_retries() -> modal.Retries:
    """
    Returns the retry policy for failed tasks.
    """
    return modal.Retries(initial_delay=0.0, max_retries=config.modal_max_retries)