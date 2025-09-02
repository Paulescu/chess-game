from pathlib import Path
import json
import random
from abc import ABC, abstractmethod

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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


class LLMPlayer(Player):
    """
    A chess player that chooses its next move using a fine-tuned LLM
    for this task.
    """
    def __init__(
        self,
        model_checkpoint_path: Path,
        # model: PeftModel,
        # tokenizer: AutoTokenizer,
    ):
        print(f"🤖 Initializing LLMPlayer from {model_checkpoint_path}")

        # self.model = model # = model.eval()
        # self.tokenizer = tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            model_checkpoint_path=model_checkpoint_path
        )
        print("🤖 LLMPlayer was successfully initialized!")

        self.name = f"LLMPlayer-from-{model_checkpoint_path}"

    def get_next_move(self, previous_moves: list[str]) -> str:
        """
        Get the next move for the player based on previous moves.
        """
        # Prepare the `user` message
        moves = json.dumps({'moves': previous_moves})
        message = [{'role': 'user', 'content': moves}]
        
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
            temperature=0.8,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
            # eos_token_id=None,
        )

        # Decode only the new tokens, excluding the ones from input_ids
        input_length = input_ids.shape[1]
        output = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        # output = self.tokenizer.decode(output[0], skip_special_tokens=False)

        return output

    # def name(self) -> str:
    #     return f"LLMPlayer-from-{self.model_checkpoint_path}"

    @staticmethod
    def _load_model_and_tokenizer(
        # modal_volume: modal.Volume,
        model_checkpoint_path: Path
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
            device_map="auto",          # Automatically distribute across GPUs if available
            trust_remote_code=True      # In case the model requires custom code
        )

        # Step 3: Load the tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True)
        
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
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        
        from_square = random.choice(files) + random.choice(ranks)
        to_square = random.choice(files) + random.choice(ranks)
        
        return from_square + to_square