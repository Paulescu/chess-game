from datetime import datetime
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import model_validator

class TrainingJobConfig(BaseSettings):

    # Model configuration
    model_name: str = "LiquidAI/LFM2-350M"
    max_seq_length: int = 2048 #32768
    load_in_4bit: bool = False  # unsloth: use 4bit quant for frozen model weights
    load_in_8bit: bool = False  # unsloth: use 8bit quant for frozen model weights

    # Dataset configuration
    dataset_name: str = "Paulescu/ChessInstruct"
    dataset_input_column: str = "input"
    dataset_output_colum: str = "next_best_move"
    train_split_ratio: float = 0.9
    preprocessing_workers: int = 2
    dataset_conversations_field: str = "conversations"
    dataset_text_field: str = "text"
    invalidate_dataset_cache: bool = False
    
    # LoRA-specific hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"  # unsloth: optimized lora kernel
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    # General training hyperparameters
    optim: str = "adamw_8bit"  # unsloth: 8bit optimizer
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    packing: bool = False
    use_gradient_checkpointing: str = "unsloth"  # unsloth: optimized gradient offloading
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_steps: int = 10000  # increase!
    save_steps: int = 1000  # increase!
    eval_steps: int = 1000  # increase!
    logging_steps: int = 10  # increase!
    eval_sample_callback_enabled: bool = False

    # Modal configuration
    modal_app_name: str = "finetune-chess-llm"
    modal_volume_pretrained_models: str = "pretrained_models"
    modal_volume_datasets: str = "datasets"
    modal_volume_model_checkpoints: str = "model_checkpoints"
    modal_gpu_type: str = "L40S"
    modal_timeout_hours: int = 6
    modal_max_retries: int = 3

    # Experiment configuration
    seed: int = 105
    wandb_project_name: str = "finetuning-things"
    wandb_experiment_name: Optional[str] = None
    wandb_enabled: bool = True
    skip_eval: bool = False
    output_dir: str = "outputs"
    
    @model_validator(mode='after')
    def set_experiment_name(self):
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"

        return self
    
    
class ModalConfig(BaseSettings):
    """
    Configuration for the Modal app
    """
    app_name: str = "finetune-chess-llm"
    pretrained_models_volume: str = "pretrained_models"
    datasets_volume: str = "datasets"
    model_checkpoints_volume: str = "model_checkpoints"
    gpu_type: str = "L40S"
    timeout_hours: int = 6
    max_retries: int = 3

# training_job_config = TrainingJobConfig()
# modal_app_config = ModalConfig()
# print(training_job_config.__dict__)