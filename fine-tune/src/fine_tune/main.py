import argparse
import pathlib
import wandb
# from transformers import AutoTokenizer

from .config import TrainingJobConfig
from .data import prepare_datasets
from .model import prepare_model
from .trainer import prepare_trainer

from .infra import (
    get_modal_app,
    get_docker_image,
    get_volume,
    get_retries,
    get_secrets,
)

config = TrainingJobConfig()

modal_app = get_modal_app()

# Modal volumes used for caching processed datasets and base LLMs and to store model checkpoints
pretrained_models_volume = get_volume(config.modal_volume_pretrained_models)
datasets_volume = get_volume(config.modal_volume_datasets)
model_checkpoints_volume = get_volume(config.modal_volume_model_checkpoints)

@modal_app.function(
    image=get_docker_image(),
    gpu=config.modal_gpu_type,
    volumes={
        "/pretrained_models": pretrained_models_volume,
        "/datasets": datasets_volume,
        "/model_checkpoints": model_checkpoints_volume,
    },
    secrets=get_secrets(),
    timeout=config.modal_timeout_hours * 60 * 60,
    retries=get_retries(),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def finetune(config: TrainingJobConfig):
    
    # Initialize Weights & Biases for experiment tracking if enabled
    if config.wandb_enabled:
        print(f"Initializing WandB experiment {config.wandb_experiment_name}")
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )

    print(f"Preparing {config.model_name} for fine-tuning")
    model, tokenizer = prepare_model(config)

    print(f"Loading and processing train/eval datasets for {config.dataset_name}")
    train_dataset, eval_dataset = prepare_datasets(config, datasets_volume, tokenizer)

    # Prepare checkpoint directory and check for existing checkpoints
    # from .checkpoints import check_for_existing_checkpoint
    # checkpoint_path = paths["checkpoints"]
    # checkpoint_path.mkdir(parents=True, exist_ok=True)
    # resume_from_checkpoint = check_for_existing_checkpoint(paths)
    # resume_from_checkpoint = False

    from .checkpoints import _get_or_create_path_to_model_checkpoints
    checkpoint_path = _get_or_create_path_to_model_checkpoints(
        config.wandb_experiment_name)

    print("Preparing trainer...")
    trainer = prepare_trainer(
        model, tokenizer, train_dataset, eval_dataset, config, checkpoint_path
    )

    # # Start training or resume from checkpoint
    # if resume_from_checkpoint:
    #     print(f"Resuming training from {resume_from_checkpoint}")
    #     trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # else:
    #     print("Starting training from scratch...")
    #     trainer.train()

    print("Start trainining...")
    trainer.train()

    # Save the final trained model and tokenizer
    print("Saving final model...")
    final_model_path = checkpoint_path / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Clean up experiment tracking
    if config.enable_wandb:
        wandb.finish()

    print(f"Training completed! Model saved to: {final_model_path}")
    return config.experiment_name


@modal_app.local_entrypoint()
def main(
    model_name: str = 'unsloth/LFM2-350M',
    learning_rate: float = 2e-4,
    max_steps: int = 10000,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.05,
    experiment_name: str = None,
    invalidate_dataset_cache: bool = False
):
    # print(f'Invalidate dataset cache: {invalidate_dataset_cache}')

    print('Creating the TrainingJobConfig for this run...')
    config = TrainingJobConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        wandb_experiment_name=experiment_name,
        invalidate_dataset_cache=invalidate_dataset_cache,
    )

    print(f"Starting finetuning experiment {config.wandb_experiment_name}...")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"LoRA configuration: rank={config.lora_r}, alpha={config.lora_alpha}")
    print(
        f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    print(f"Training steps: {config.max_steps}")

    # Launch the training job on Modal infrastructure
    finetune.remote(config)
    
    print(f"Training completed successfully: {config.wandb_experiment_name}")