"""
Downloads a LoRA adapter checkpoint from a remote Modal volume,
and
Merges it with a base model, optionally pushing the result to the Hugging Face Hub.
"""

import os
import shutil
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model_checkpoint_from_modal_volume(
    modal_volume_name: str,
    remote_path_to_checkpoint: str,
):
    """
    Downloads LoRA adapters from a remote Modal volume
    """
    # Start a shell command to download the adapters
    import subprocess

    print(f"Downloading LoRA adapters from Modal volume '{modal_volume_name}'...")
    subprocess.run(
        f"modal volume get {modal_volume_name} {remote_path_to_checkpoint} --force",
        shell=True,
    )

    # Return the local path to the downloaded LoRA adapters
    return f"./{remote_path_to_checkpoint.split('/')[-1]}"


def merge_lora_adapter_to_base_model(
    adapter_path: str,
    output_dir: str,
):
    """
    Merge LoRA adapter with base model and optionally push to HF Hub

    Args:
        adapter_path: Path to the LoRA adapter checkpoint directory
        output_dir: Local directory to save the merged model
    """

    print("🔄 Starting LoRA adapter merge process...")

    # Step 1: Load the adapter configuration
    print("📋 Loading adapter configuration...")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"📦 Base model: {base_model_name}")
    print(f"🎯 Adapter path: {adapter_path}")

    # Step 2: Load the base model
    print("🏗️  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",  # Automatically distribute across GPUs if available
        trust_remote_code=True,  # In case the model requires custom code
    )

    # Step 3: Load the tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Step 4: Load and merge the LoRA adapter
    print("🔗 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("⚡ Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    # Step 5: Save the merged model locally
    print(f"💾 Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    merged_model.save_pretrained(
        output_dir,
        save_method="save_pretrained",
        safe_serialization=True,  # Use safetensors format
    )
    tokenizer.save_pretrained(output_dir)

    print("✅ Model saved locally!")


def run(
    modal_volume_name: str,
    remote_checkpoint_dir: str,
    local_merged_models_dir: str,
):
    local_path_to_lora_adapters = download_model_checkpoint_from_modal_volume(
        modal_volume_name=modal_volume_name,
        remote_path_to_checkpoint=remote_checkpoint_dir,
    )
    print(f"LoRA adapters downloaded to: {local_path_to_lora_adapters}")

    # Merge the adapter
    merge_lora_adapter_to_base_model(
        adapter_path=local_path_to_lora_adapters,
        output_dir=Path(local_merged_models_dir) / remote_checkpoint_dir,
    )

    print("Deleting LoRA adapters from local directory...")
    shutil.rmtree(local_path_to_lora_adapters)


if __name__ == "__main__":
    from fire import Fire

    Fire(run)
