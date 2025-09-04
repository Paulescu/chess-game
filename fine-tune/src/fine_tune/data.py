from pathlib import Path

import datasets
import modal

from .config import TrainingJobConfig

# from transformers import AutoTokenizer
from .prompt_template import get_prompt


def prepare_datasets(
    config: TrainingJobConfig,
    datasets_volume: modal.Volume,
    tokenizer: "AutoTokenizer",
):
    # import datasets

    # Get path of cache datasets generate in a previous run using the same
    # `dataset_name`, `train_split_ratio` and `seed`
    dataset_cache_path = _get_path_to_cached_datasets(
        dataset_name=config.dataset_name,
        train_split_ratio=config.train_split_ratio,
        seed=config.seed,
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

        if config.dataset_samples is not None:
            dataset = dataset.select(range(config.dataset_samples))
            print(f"Selected {config.dataset_samples} samples for training.")

        print("Converting instructions to conversation format...")
        # dataset = dataset.map(_convert_to_openai_conversation_format)
        # dataset = convert_dataset_to_conversation_format(
        #   dataset, config.prompt_template_file)
        dataset = dataset.map(convert_to_conversation_format)

        print("Sample conversations before applying chat templates:")
        for i in range(5):
            print(f"Sample {i}: {dataset[i]['conversations']}")
            print("--------")

        print("Applying chat templates to conversations...")
        dataset = dataset.map(
            lambda examples: apply_chat_template(examples, tokenizer),
            batched=True,
            num_proc=config.preprocessing_workers,
            remove_columns=dataset.column_names,
        )

        print("Sample conversations after applying chat templates:")
        for i in range(5):
            print(f"Sample {i}: {dataset[i]['text']}")
            print("--------")

        print("Splitting dataset into train and eval sets...")
        dataset = dataset.train_test_split(
            test_size=1.0 - config.train_split_ratio, seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")

        print(f"Commiting write operation to {datasets_volume}")
        datasets_volume.commit()
        print(f"Commited write operation to {datasets_volume}")

    print("Printing 5 first samples from the training dataset...")
    for i in range(5):
        print("Sample ", i)
        print(train_dataset[i]["text"])
        print("--------")

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
        Path("/datasets")
        / dataset_name.replace("/", "--")
        / f"train-{train_split_ratio}-seed-{seed}"
    )


def convert_to_conversation_format(example: dict) -> dict:
    # data = {
    #     'player_to_move': example["player_to_move"],
    #     'game_state': example["game_state"],
    #     'last_5_moves_uci': example["last_5_moves_uci"],
    #     'valid_moves': example["valid_moves"],
    # }
    prompt = get_prompt(
        # player_to_move=example["player_to_move"],
        game_state=example["game_state"],
        last_5_moves_uci=example["last_5_moves_uci"],
        valid_moves=example["valid_moves"],
    )

    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["next_move"]},
        ]
    }


def apply_chat_template(examples, tokenizer):
    texts = []
    for conversation in examples["conversations"]:
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {"text": texts}
