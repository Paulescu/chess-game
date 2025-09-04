import chess
from pydantic import BaseModel

from .players import Player


class ChessGameStats(BaseModel):
    result: str
    n_moves: int
    moves: list[str]


class ChessGame:
    def __init__(
        self,
        white_player: Player,
        black_player: Player,
        log_enabled: bool = True,
    ):
        self.white_player = white_player
        self.black_player = black_player
        self.previous_moves = []
        self.white_plays: bool = True
        self.board = chess.Board()

        self.log_enabled = log_enabled

    def play(self) -> ChessGameStats:
        """
        Simulate a game of chess between two players.
        """
        self._log("Starting a new game of chess!")

        # loop until the game is over
        while not self._is_game_over():
            # Get player whose turn to play is NOW
            player = self._get_player()

            # Ask the player for its next move, until we get a valid move
            next_move = ""
            n_attempts = 0
            while not self._is_valid_move(next_move):
                # Ask player for its next move
                next_move = player.get_next_move(self.previous_moves)

                n_attempts += 1
                if (n_attempts > 2) and ("LLMPlayer" in player.name):
                    print(
                        f"Player {player.name} failed to provide a valid move after \
                            {n_attempts} attempts. Last move was: {next_move}"
                    )

                    return ChessGameStats(
                        result="aborted",
                        n_moves=len(self.previous_moves),
                        moves=self.previous_moves,
                    )

            self._log(f"Applying move {next_move}")

            # Apply the move and update the game state
            self._apply_move(next_move)

        # The game is over
        self._log("Game over!")
        result = self._get_result()

        return ChessGameStats(
            result=result,
            n_moves=len(self.previous_moves),
            moves=self.previous_moves,
        )

    def _get_result(self) -> bool:
        return self.board.result()

    def _is_game_over(self) -> bool:
        """
        Checks if the game is over (checkmate, stalemate, draw, etc.)
        """
        return self.board.is_game_over()

    def _apply_move(self, move: str):
        """
        Adds the move to the game state and switches turns.
        """
        # update the state of the self.board
        move_obj = chess.Move.from_uci(move)
        self.board.push(move_obj)

        self.previous_moves.append(move)
        self.white_plays = not self.white_plays

        # print('moves so far: ', self.previous_moves)

    def _get_player(self) -> Player:
        """
        Returns the player whose turn is now
        """
        if self.white_plays:
            return self.white_player
        else:
            return self.black_player

    def _is_valid_move(self, next_move: str) -> bool:
        """
        Validate if a chess move is legal given the current game state.

        Args:
            next_move: The move to validate (e.g., "e2e4")

        Returns:
            True if the move is valid, False otherwise
        """
        try:
            next_move_obj = chess.Move.from_uci(next_move)
            return next_move_obj in self.board.legal_moves
        except ValueError:
            return False

    def _log(self, msg: str):
        if self.log_enabled:
            print(msg)
