.PHONY: download-magnus-games

download-magnus-games:
	curl -L -o data/raw/Carlsen.zip https://www.pgnmentor.com/players/Carlsen.zip
	cd data/raw && unzip -o Carlsen.zip
	rm data/raw/Carlsen.zip

instruction-dataset:
	uv run python scripts/generate_instruction_dataset.py \
		--hugging-face-dataset-name Paulescu/MagnusInstruct

#############
BASE_MODEL = unsloth/LFM2-700M
#############

fine-tune:
	uv run modal run -m src.fine_tune.main --model-name $(BASE_MODEL) --invalidate-dataset-cache

#############
CHECKPOINT := LFM2-350M-r16-20250902-232247/checkpoint-5000
#############

evaluate:
	uv run modal run -m src.fine_tune.evaluate --model-checkpoint $(CHECKPOINT)

merge-model:
	uv run python scripts/generate_merged_model.py \
		--modal-volume-name model_checkpoints \
		--remote-checkpoint-dir $(CHECKPOINT) \
		--local-merged-models-dir ./merged-models

bundle-model: merge-model
	mkdir -p merged-models && cd merged-models && \
	uv run leap-bundle create $(CHECKPOINT)

list-bundles:
	uv run leap-bundle list

#############
BUNDLE_ID = 6
#############

download-model-bundle:
	mkdir -p model-bundles && cd model-bundles && \
	uv run leap-bundle download $(BUNDLE_ID)

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .