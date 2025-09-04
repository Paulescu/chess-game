from .config import TrainingJobConfig


def prepare_model(config: TrainingJobConfig) -> tuple["Model", "Tokenizer"]:
    """
    Loads the model from HF using smart caching with Modal volumes
    and
    Adds the LoRA adapters to the model for fine-tuning
    """
    model, tokenizer = load_pretrained_model(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )
    model = add_lora_adapters(
        model,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_bias=config.lora_bias,
        lora_target_modules=config.lora_target_modules,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        seed=config.seed,
        use_rslora=config.use_rslora,
    )
    return model, tokenizer


def load_pretrained_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> "FastLanguageModel":
    """
    Downloads and loads into memory a pretrained language model.
    """
    from unsloth import FastLanguageModel

    print(f"Downloading pretrained model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    return model, tokenizer


def add_lora_adapters(
    model,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_target_modules: list[str],
    use_gradient_checkpointing: str,
    seed: int,
    use_rslora: bool,  # Rank-stabilized LoRA
) -> "FastLanguageModel":
    """
    Adds LoRA adapters to the model for efficient finetuning.
    """
    from unsloth import FastLanguageModel

    print("Configuring LoRA for training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        # What is this?
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=seed,
        use_rslora=use_rslora,
        loftq_config=None,  # LoFTQ quantization config
    )
    return model
