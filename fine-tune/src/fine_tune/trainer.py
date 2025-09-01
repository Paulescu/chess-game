from pathlib import Path

from .config import TrainingJobConfig

def prepare_trainer(
    model: "Model",
    tokenizer: "Tokenizer",
    train_dataset: "Dataset",
    eval_dataset: "Dataset",
    config: TrainingJobConfig,
    checkpoint_path: Path
) -> "Trainer":
    """
    """
    from trl import SFTTrainer

    print("Extracting trainer parameters from TrainingJobConfig")
    training_args = get_training_arguments(config, checkpoint_path)

    # Initialize the supervised finetuning trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=config.dataset_text_field,
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.preprocessing_workers,
        packing=config.packing,  # Sequence packing for efficiency
        args=training_args,
    )

    # Display training information for transparency
    print(f"Training dataset size: {len(train_dataset):,}")
    print(f"Evaluation dataset size: {len(eval_dataset):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Experiment: {config.wandb_experiment_name}")

    return trainer

def get_training_arguments(config: TrainingJobConfig, output_path: Path) -> "TrainingArguments":
    """
    Create training arguments for the SFTTrainer.

    These arguments control the training process, including optimization settings,
    evaluation frequency, and checkpointing behavior.
    """
    import torch
    from transformers import TrainingArguments

    return TrainingArguments(
        # Core training configuration
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        
        # Evaluation and checkpointing
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="no" if config.skip_eval else "steps",
        save_strategy="steps",
        do_eval=not config.skip_eval,
        
        # Optimization settings based on hardware capabilities
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 not available
        bf16=torch.cuda.is_bf16_supported(),  # Prefer bf16 when available
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        
        # Logging and output configuration
        logging_steps=config.logging_steps,
        output_dir=str(output_path),
        report_to="wandb" if config.wandb_enabled else None,
        seed=config.seed,
    )