.PHONY: lint fix

instruction-dataset:
	uv run python scripts/instruction_dataset_generation.py

fine-tune:
	uv run modal run -m src.fine_tune.main --model-name unsloth/LFM2-350M --invalidate-dataset-cache

evaluate:
	uv run modal run -m src.fine_tune.evaluate --model-checkpoint LFM2-350M-r16-20250901-135855/checkpoint-9000

merged-model:
	uv run python scripts/generate_merged_model.py \
		--modal-volume-name model_checkpoints \
		--remote-checkpoint-dir $(or $(CHECKPOINT),LFM2-350M-r16-20250901-135855/checkpoint-9000) \
		--local-merged-models-dir ./merged-models

model-bundle:
	mkdir -p merged-models && cd merged-models && \
	uv run leap-bundle create $(or $(CHECKPOINT),LFM2-350M-r16-20250901-135855/checkpoint-9000)

download-model-bundle:
	mkdir -p model-bundles && cd model-bundles && \
	uv run leap-bundle download $(or $(DOWNLOAD_ID),5)

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .