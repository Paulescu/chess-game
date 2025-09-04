import glob
import json
import os
from pathlib import Path
from typing import Any

import chess
import chess.pgn
from datasets import Dataset
from tqdm import tqdm


def count_games(pgn_file_path: str) -> int:
    """Count total number of games in PGN file."""
    game_count = 0
    with open(pgn_file_path) as pgn_file:
        while chess.pgn.read_game(pgn_file) is not None:
            game_count += 1
    return game_count


def extract_game_data(pgn_file_path: str) -> list[dict[str, Any]]:
    """
    Parse PGN file and extract move sequences, game states, and valid moves for each
    position.

    Returns:
        List of dictionaries containing:
        - moves_uci: List of moves in UCI notation up to this position
        - game_state: FEN string representing board state after moves
        - valid_moves: List of valid next moves in UCI notation
        - move_number: Position in the game (0-indexed)
        - game_id: Unique identifier for the game
    """
    extracted_data = []
    game_id = 0

    total_games = count_games(pgn_file_path)
    print(f"Total games in PGN: {total_games}")

    with open(pgn_file_path) as pgn_file:
        with tqdm(total=total_games, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = chess.Board()
                moves_uci = []
                white_player = game.headers.get("White", "Unknown")
                black_player = game.headers.get("Black", "Unknown")

                # Extract each move and corresponding game state
                for move_number, move in enumerate(game.mainline_moves()):
                    # Get current state before applying move
                    game_state = board.fen()
                    valid_moves = [str(legal_move) for legal_move in board.legal_moves]

                    # Determine which player is making the move
                    # White moves on even numbers, Black on odd
                    current_player = (
                        white_player if move_number % 2 == 0 else black_player
                    )

                    # Create data point for this position
                    data_point = {
                        "moves_uci": moves_uci.copy(),
                        "last_5_moves_uci": moves_uci[-5:],
                        "game_state": game_state,
                        "valid_moves": valid_moves,
                        "move_number": move_number,
                        "game_id": game_id,
                        "next_move": str(move),  # The actual move played
                        "player_to_move": current_player,
                    }
                    extracted_data.append(data_point)

                    # Apply the move and add to move list
                    board.push(move)
                    moves_uci.append(str(move))

                game_id += 1
                pbar.update(1)

    return extracted_data


def save_dataset(data: list[dict[str, Any]], output_file: str):
    """Save extracted data to JSON file."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} data points to {output_file}")


def process_one_pgn_file(
    pgn_path: str,
    output_path: str,
):
    """
    Extracts games from the given `pgn_path` and saves it into an `output_path` json
    file
    """
    print(f"Extracting game data from {pgn_path}")
    data = extract_game_data(pgn_path)

    print(f"Extracted {len(data)} positions from games")

    # Show sample data point
    if data:
        print("\nSample data point:")
        print(json.dumps(data[0], indent=2))

    save_dataset(data, output_path)


def generate_instruction_dataset(
    raw_data_dir: str | None = None,
    processed_data_dir: str | None = None,
    hugging_face_dataset_name: str | None = None,
):
    """
    Process all PGN files in raw_data_dir, create processed JSON files,
    then combine into a HuggingFace dataset and push to hub.
    """
    if raw_data_dir is None:
        script_path = Path(__file__).resolve()
        raw_data_dir = script_path.parent.parent / "data" / "raw"

    if processed_data_dir is None:
        script_path = Path(__file__).resolve()
        processed_data_dir = script_path.parent.parent / "data" / "processed"

    # Ensure processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Find all PGN files in raw data directory
    pgn_files = glob.glob(os.path.join(raw_data_dir, "*.pgn"))
    print(
        f"Found {len(pgn_files)} PGN files: {[os.path.basename(f) for f in pgn_files]}"
    )

    # Process each PGN file
    processed_files = []
    for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
        filename = os.path.splitext(os.path.basename(pgn_path))[0]
        output_path = os.path.join(processed_data_dir, f"{filename}.json")

        print(f"\nProcessing {os.path.basename(pgn_path)}...")
        process_one_pgn_file(pgn_path, output_path)
        processed_files.append(output_path)

        # TODO: remove this
        # break

    # Load all processed data
    all_data = []
    for json_path in processed_files:
        print(f"Loading {os.path.basename(json_path)}...")
        with open(json_path) as f:
            data = json.load(f)
            all_data.extend(data)

    print(f"\nTotal data points before filtering: {len(all_data)}")

    # Filter to keep only positions where Magnus Carlsen is the player to move
    carlsen_data = [
        point for point in all_data if "Carlsen" in point.get("player_to_move", "")
    ]
    print(f"Data points with Carlsen to move: {len(carlsen_data)}")

    all_data = carlsen_data

    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_data)
    print(f"Created dataset with {len(dataset)} samples")

    # Push to hub
    print(f"Pushing dataset to HuggingFace Hub: {hugging_face_dataset_name}")
    dataset.push_to_hub(hugging_face_dataset_name)
    print("Dataset successfully pushed to hub!")


if __name__ == "__main__":
    from fire import Fire

    # Example usage
    Fire(generate_instruction_dataset)
