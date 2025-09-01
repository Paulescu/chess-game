from pathlib import Path

import modal

from .config import TrainingJobConfig

def prepare_datasets(
    config: TrainingJobConfig,
    datasets_volume: modal.Volume,
    tokenizer: "Tokenizer",
):
    import datasets

    # Get path of cache datasets generate in a previous run using the same
    # `dataset_name`, `train_split_ratio` and `seed`
    dataset_cache_path = _get_path_to_cached_datasets(
        dataset_name=config.dataset_name,
        train_split_ratio=config.train_split_ratio,
        seed=config.seed
    )

    if dataset_cache_path.exists() and not config.invalidate_dataset_cache:
        print(f"Loading train/eval cached datasets from {dataset_cache_path}")
        train_dataset = datasets.load_from_disk(dataset_cache_path / "train")
        eval_dataset = datasets.load_from_disk(dataset_cache_path / "eval")
    else:
        print(f"Downloading and processing dataset: {config.dataset_name}")

        # Load and standardize the dataset format
        dataset = datasets.load_dataset(config.dataset_name, split="train")
        print(f"Dataset {config.dataset_name} has {len(dataset)} examples.")

        dataset = dataset.map(_convert_to_openai_conversation_format)

        # Split into training and evaluation sets with fixed seed for reproducibility
        dataset = dataset.train_test_split(
            test_size=1.0 - config.train_split_ratio, seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Apply chat template formatting to convert conversations to text
        print("Formatting datasets with chat template...")
        train_dataset = train_dataset.map(
            lambda examples: _format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=config.preprocessing_workers,
            remove_columns=train_dataset.column_names,
        )

        eval_dataset = eval_dataset.map(
            lambda examples: _format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=config.preprocessing_workers,
            remove_columns=eval_dataset.column_names,
        )

        # Cache the processed datasets for future runs
        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")
            
        # Commit the dataset cache to the volume
        datasets_volume.commit()
        print(f"Commited write operation to {datasets_volume}")

    print("Printing 5 first samples from the training dataset...")
    for i in range(5):
        print('Sample ', i)
        print(train_dataset[i]['text'])
        print('--------')

    return train_dataset, eval_dataset

def _get_path_to_cached_datasets(
    dataset_name: str,
    train_split_ratio: float,
    seed: int,
) -> Path:
    """
    Returns path to the cached dataset in a Modal volume.
    """
    return (
        Path("/datasets") /
        dataset_name.replace("/", "--") /
        f"train-{train_split_ratio}-seed-{seed}"
    )

def _convert_to_openai_conversation_format(example) -> dict:
    """
    Convert a single example to the OpenAI conversation format.
    """
    return {
        "conversations": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["next_best_move"]},
        ]
    }

def _format_chat_template(examples, tokenizer):
    texts = []
    for conversation in examples["conversations"]:
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {"text": texts}