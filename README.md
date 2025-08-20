## Let's build a Chess game using LLMs running privately on your phone/computer/microwave

## TODOs

- [ ] Fine-tune LFM2-350M using the ChessInstruct dataset using Colab notebook
- [ ] Push model to HuggingFace model registry
- [ ] Craete GGUF version of the model
- [ ] Deploy the model to my Mac using llama.cpp


### Chat template used for fine-tuning

Example:

```
<|startoftext|><|im_start|>system
Given an incomplit set of chess moves and the game's final score, write the last missing chess move.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: The missing chess move<|im_end|>
<|im_start|>user
{"moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "d7d5", "a2a3", "b4c3", "c2c3", "f6e4", "c3c2", "e8g8", "g1f3", "b7b6", "c1f4", "c7c5", "e2e3", "c8b7", "d4c5", "b6c5", "e1c1", "b8c6", "f1d3", "d8b6", "d3e4", "d5e4", "f3e5", "c6e5", "f4e5", "f7f6", "e5c3", "a8d8", "d1d8", "f8d8", "c2a4", "b7c6", "a4a5", "g8f7", "c1c2", "e6e5", "b2b3", "d8d7", "a5b6", "a7b6", "a3a4", "d7a7", "h1a1", "f7e7", "a4a5", "b6a5", "c3a5", "e7e6", "c2b2", "h7h5", "a1a3", "a7a6", "a5c7", "c6b7", "a3a5", "a6a5", "c7a5", "e6d6", "b3b4", "c5b4", "a5b4", "d6d7", "b2c3", "b7c8", "b4f8", "g7g6", "c3b4", "h5h4", "b4b5", "c8b7", "f8g7", "f6f5", "g7e5", "b7c6", "b5c5", "h4h3", "g2h3", "c6a8", "e5d6", "a8c6", "d6f8", "c6b7", "c5d4", "d7e6", "h3h4", "b7c6", "f8h6", "c6a8", "h6f8", "a8c6", "d4c5", "e6d7", "c5b6", "c6a8", "f8h6", "a8c6", "h6f4", "c6a8", "f4h6", "d7d6", "b6a7", "a8c6", "h6f4", "d6c5", "a7b8", "c5c4", "b8c8", "c4d5", "f4c7", "d5e6", "c7b6", "c6d7", "c8c7", "d7a4", "b6d4", "a4d7", "c7d8", "d7b5", "d4b6", "e6f7", "b6d4", "b5a4", "d4b2", "f7e6", "d8c7", "a4d7", "c7b8", "e6e7", "b2a3", "e7e6", "a3b2", "d7a4", "b8c7", "a4b5", "c7b6", "b5e8", "b6b7", "e8d7", "b2a3", "d7b5", "b7c8", "b5d7", "c8d8", "d7b5", "d8c7", "b5a4", "a3c5", "a4d7", "c5a7", "d7e8", "a7c5", "e8d7", "c5b6", "d7a4", "c7d8", "a4d7", "b6a7", "d7a4", "d8c7", "a4b5", "c7c8", "b5a4", "a7b6", "a4d7", "c8d8", "d7a4", "d8c7", "a4d7", "b6a7", "d7a4", "a7b6", "a4b5", "b6d4", "b5e8", "d4a7", "e8a4", "c7d8", "a4b5", "a7d4", "b5d7", "d4a7", "d7b5", "d8c8", "b5d7", "c8b8", "d7c6", "a7b6", "e6d7", "b6a7", "d7e6", "b8c7", "c6a4", "a7d4", "a4b5", "d4b2", "e6d5", "b2g7", "b5a4", "g7a1", "a4b5", "a1f6", "b5c6", "c7b6", "d5d6", "f6e5", "d6d7", "e5d4", "c6a8", "b6a7", "d7c7", "a7a8", "c7c6", "a8b8", "c6d6", "b8b7", "d6d7", "d4c5", "d7e6", "b7c6", "e6f7", "c6d6", "f7g7", "h4h5", "g6g5", "d6e5", "f5f4", "e3f4", "g7h6", "c5e3", "g5f4", "e5f4", "h6h5", "f4e4", "h5g6", "e4e5", "g6h5", "f2f4", "h5h4", "f4f5", "h4h3", "f5f6", "h3g2", "e5e4", "g2f1", "f6f7", "f1e1", "f7f8q", "e1d1", "f8f7", "d1e2", "f7f2", "e2d1", "?"], "result": "1-0"}<|im_end|>
<|im_start|>assistant
{"missing move": "f2d2"}<|im_end|>
```