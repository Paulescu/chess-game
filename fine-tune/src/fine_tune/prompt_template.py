from jinja2 import Template

# Define template as a string
CHESS_PROMPT_TEMPLATE = """
You are the great Magnus Carlsen.
Your task is to make the best move in the given game state.

Game state: {{ game_state }}
Last 5 moves: {{ last_5_moves_uci }}
Valid moves: {{ valid_moves }}

Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
Make sure your next move is one of the valid moves.
"""


def get_prompt(
    # player_to_move: str,
    game_state: str,
    last_5_moves_uci: list[str],
    valid_moves: list[str],
) -> str:
    template = Template(CHESS_PROMPT_TEMPLATE)
    prompt = template.render(
        # player_to_move=player_to_move,
        game_state=game_state,
        last_5_moves_uci=last_5_moves_uci,
        valid_moves=valid_moves,
    )
    return prompt
