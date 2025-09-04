""" "
Bunch of functions to test the output produced by our ChessInstruct model make any sense
"""

from pathlib import Path

from .config import TrainingJobConfig
from .game import ChessGame
from .infra import (
    # get_docker_image,
    get_docker_image_for_evaluation,
    get_modal_app,
    get_retries,
    # get_secrets,
    get_volume,
)
from .players import LLMPlayer, Player, RandomPlayer

config = TrainingJobConfig()

modal_app = get_modal_app()
docker_image = get_docker_image_for_evaluation()
model_checkpoints_volume = get_volume(config.modal_volume_model_checkpoints)


@modal_app.function(
    image=docker_image,
    gpu=config.modal_gpu_type,
    volumes={
        "/model_checkpoints": model_checkpoints_volume,
    },
    timeout=config.modal_timeout_hours * 60 * 60,
    retries=get_retries(),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def evaluate(
    model_checkpoint_path: str,
):
    """ """
    model_checkpoint_path = Path("/model_checkpoints") / model_checkpoint_path

    # Initialize the AI player
    ai_player = LLMPlayer(model_checkpoint_path=model_checkpoint_path)
    sanity_check(ai_player)

    # Initialize the random player
    random_player = RandomPlayer()
    sanity_check(random_player)

    game = ChessGame(
        white_player=random_player,
        black_player=ai_player,
    )
    result = game.play()
    print(result)


@modal_app.local_entrypoint()
def main(
    model_checkpoint: str,
):
    print(f"Running evaluation on the model {model_checkpoint}")

    # Launch the training job on Modal infrastructure
    evaluate.remote(
        model_checkpoint_path=model_checkpoint,
    )

    print("✅ Evaluation completed!")


def sanity_check(player: Player):
    """
    Prints on console the next_move from the player for a set of cases
    """
    print("Sanity checks for ", player.name)

    games = [
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3"],
        [
            "e2e4",
            "c7c5",
            "g1f3",
            "b8c6",
            "d2d4",
            "c5d4",
            "f3d4",
            "d7d6",
            "c1e3",
            "g7g6",
            "f2f3",
            "e7e5",
            "d4b3",
            "g8f6",
            "b1c3",
            "c8e6",
            "d1e2",
            "a7a5",
            "a2a4",
            "f8e7",
            "e1c1",
            "e8g8",
            "b3c5",
            "d6d5",
            "e4d5",
            "f6d5",
            "c3d5",
            "e6d5",
            "c5b7",
            "d8c8",
            "d1d5",
            "c8b7",
            "e2b5",
            "b7c8",
            "b5b3",
            "c6d4",
            "e3d4",
            "e5d4",
            "c1b1",
            "a8b8",
            "d5b5",
            "c8c7",
            "b3c4",
            "c7d6",
            "f1d3",
            "f8c8",
            "c4b3",
            "b8b5",
            "a4b5",
            "d6b4",
            "b3b4",
            "e7b4",
            "b1a2",
            "g8f8",
            "h1d1",
            "c8c7",
            "f3f4",
            "f8e7",
            "a2b3",
            "e7d6",
            "f4f5",
            "g6g5",
            "f5f6",
            "h7h6",
            "d1f1",
            "b4d2",
            "b3a4",
            "d2b4",
            "f1f2",
            "c7c5",
            "f2e2",
            "c5e5",
            "e2e5",
            "d6e5",
            "b5b6",
            "e5f6",
            "g2g4",
            "f6e6",
            "a4b5",
            "e6d7",
            "d3f5",
            "d7d6",
        ],
        [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "g8f6",
            "e1g1",
            "f8e7",
            "f1e1",
            "d7d6",
            "d2d4",
            "e5d4",
            "f3d4",
            "c8d7",
            "b1c3",
            "e8g8",
            "d4f5",
            "d7f5",
            "e4f5",
            "a7a6",
            "b5f1",
            "h7h6",
            "g2g4",
            "d6d5",
            "h2h3",
            "e7b4",
            "f1g2",
            "d8d6",
            "d1f3",
            "c6d4",
            "f3d3",
            "d4b5",
            "c1d2",
            "c7c6",
            "a2a4",
            "b5c3",
            "b2c3",
            "b4a5",
            "c3c4",
            "a5d2",
            "d3d2",
            "f8d8",
            "a1b1",
            "d6c7",
            "d2b4",
            "d8d7",
            "c4d5",
            "c6d5",
            "e1e2",
            "a8c8",
            "a4a5",
            "c7d8",
            "g2f3",
            "c8c4",
            "b4e1",
            "c4a4",
            "b1a1",
            "a4c4",
            "a1d1",
            "d8c7",
            "g1g2",
            "g8h7",
            "d1d3",
        ],
        ["d2d4", "g8f6", "c2c4"],
        ["c2c3"],
        [],
        ["g2g3", "d7d5", "e2e3"],
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "e2e3", "f8e7", "g1f3"],
    ]
    for game in games:
        next_move = player.get_next_move(previous_moves=game)
        print("Previous moves: ", game)
        print("Next move: ", next_move)
        print("-----")
