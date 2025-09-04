import random
from abc import ABC, abstractmethod
from pathlib import Path

import chess
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt_template import get_prompt


class Player(ABC):
    """
    Abstract base class for chess players.
    """

    @abstractmethod
    def get_next_move(self, previous_moves: list[str]) -> str:
        """
        Get the next move for the player based on previous moves.

        Args:
            previous_moves: List of previous moves in algebraic notation

        Returns:
            The next move in algebraic notation (e.g., 'e2e4', 'f8c8')
        """
        pass

    def _get_board(self, previous_moves: list[str]) -> chess.Board:
        """
        Get the current game state in FEN notation based on previous UCI moves.

        Args:
            previous_moves: List of moves in UCI notation (e.g., ['e2e4', 'e7e5'])

        Returns:
            board
        """
        board = chess.Board()

        for move_uci in previous_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    raise ValueError(f"Illegal move: {move_uci}")
            except (chess.InvalidMoveError, ValueError) as e:
                raise ValueError(f"Invalid move in sequence: {move_uci}") from e

        return board

    def _get_game_state(self, previous_moves: list[str]) -> str:
        board = self._get_board(previous_moves)
        return board.fen()

    def _get_last_5_moves(self, previous_moves: list[str]) -> list[str]:
        return previous_moves[-5:]

    def _get_valid_moves(self, previous_moves: list[str]) -> list[str]:
        board = self._get_board(previous_moves)
        return [str(move) for move in board.legal_moves]


class LLMPlayer(Player):
    """
    A chess player that chooses its next move using a fine-tuned LLM
    for this task.
    """

    def __init__(
        self,
        model_checkpoint_path: Path,
    ):
        print(f"🤖 Initializing LLMPlayer from {model_checkpoint_path}")

        # Load the model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            model_checkpoint_path=model_checkpoint_path
        )
        print("🤖 LLMPlayer was successfully initialized!")

        self.name = f"LLMPlayer-from-{model_checkpoint_path}"
        # self.name = 'LLMPlayer'

    def get_next_move(self, previous_moves: list[str]) -> str:
        """
        Get the next move for the player based on previous moves.
        """
        game_state = self._get_game_state(previous_moves)
        last_5_moves_uci = self._get_last_5_moves(previous_moves)
        valid_moves = self._get_valid_moves(previous_moves)

        prompt = get_prompt(
            game_state=game_state,
            last_5_moves_uci=last_5_moves_uci,
            valid_moves=valid_moves,
        )
        message = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        # print('type(self.model)', type(self.model))
        # print('type(self.tokenizer)', type(self.tokenizer))

        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.80,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
            # eos_token_id=None,
        )

        # Decode only the new tokens, excluding the ones from input_ids
        input_length = input_ids.shape[1]
        output = self.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )
        # output = self.tokenizer.decode(output[0], skip_special_tokens=False)

        return output

    @staticmethod
    def _load_model_and_tokenizer(
        # modal_volume: modal.Volume,
        model_checkpoint_path: Path,
    ):
        # Step 1: Load the adapter configuration from the Modal volume
        # adapter_path = Path('/model_checkpoints') / model_checkpoint_path
        adapter_path = model_checkpoint_path
        print(f"📋 Loading adapter configuration from {adapter_path}...")
        peft_config = PeftConfig.from_pretrained(adapter_path)

        # Step 2. Load the base model
        base_model_name = peft_config.base_model_name_or_path
        print(f"🏗️  Loading base model {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="auto",  # Automatically distribute across GPUs if available
            trust_remote_code=True,  # In case the model requires custom code
        )

        # Step 3: Load the tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        # Step 4: Load and merge the LoRA adapter
        print("🔗 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # print('type(model)', type(model))
        # print('type(tokenizer)', type(tokenizer))

        return model.eval(), tokenizer


class RandomPlayer(Player):
    def __init__(self):
        self.name = "RandomPlayer"

    def get_next_move(self, previous_moves: list[str]) -> str:
        """
        Generate a random algebraic chess move.
        """
        valid_moves = self._get_valid_moves(previous_moves)

        # sample 1 random move
        return random.choice(valid_moves)
