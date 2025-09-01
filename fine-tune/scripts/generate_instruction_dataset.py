"""
Filters the Thytu/ChessInstruct dataset to keep only samples with a "missing_moves" list
as expected output
"""

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
)
import json

def validate_dataset(dataset: Dataset):
    """
    Checks that all rows in the dataset have a valid next_best_move
    Otherwise, it throws an exception.
    """
    print(f"Validating dataset {dataset}...")

    for example in dataset:
        if len(example['next_best_move']) not in [4, 5]:
            raise Exception(f"Invalid move found: {example}")
            
    print('Validation completed!')


def generate_instruction_dataset(
    from_dataset: str,
    to_dataset: str,
):
    print(f"Loading dataset {from_dataset}...")
    dataset: DatasetDict = load_dataset(from_dataset)
    train = dataset['train']
    test = dataset['test']

    print(f"Original dataset {from_dataset}")
    print(f"{len(train)} train samples")
    print(f"{len(test)} test samples")

    print("Processing train split")
    train = process_one_dataset(train)
    print("Processing test split")
    test = process_one_dataset(test)

    print('Final dataset:')
    print(f"{len(train)} train samples")
    print(f"{len(test)} test samples")
    
    print('Validating datasets have valid `next_best_move` targets')
    validate_dataset(train)
    validate_dataset(test)
    
    dataset = DatasetDict({
        "train": train,
        "test": test,
    })

    print(f"Pushing dataset to the hub: {to_dataset}...")
    dataset.push_to_hub(to_dataset)

    print('Done!')

def is_valid_move(move: str) -> bool:
    """
    'f8c8' -> Valid!
    'y3i9' -> Invalid!
    "h7h8q" -> Valid (pawn promotion to queen)
    """
    return move[0] in 'abcdefgh' and move[1] in '12345678' and move[2] in 'abcdefgh' and move[3] in '12345678'

def extract_next_best_move(example):
    example['next_best_move'] = json.loads(example['expected_output']).get('next best move')
    return example if is_valid_move(example['next_best_move']) else None

def process_one_dataset(dataset: Dataset) -> Dataset:

    return (
        dataset
        .filter(lambda x: x['KIND'] == 'FIND_NEXT_BEST_MOVE')
        .map(extract_next_best_move)
        .select_columns(['input', 'next_best_move'])
    )

if __name__ == "__main__":
    generate_instruction_dataset(
        from_dataset='Thytu/ChessInstruct',
        to_dataset='Paulescu/ChessInstruct',
    )