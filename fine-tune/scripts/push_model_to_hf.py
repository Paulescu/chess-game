#!/usr/bin/env python3
"""
Script to merge LoRA adapter checkpoints with base model and push to Hugging Face Hub
"""

import os

import torch
from huggingface_hub import login
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_adapter_to_base_model(
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_name: str = None,
    hub_token: str = None,
):
    """
    Merge LoRA adapter with base model and optionally push to HF Hub

    Args:
        adapter_path: Path to the LoRA adapter checkpoint directory
        output_dir: Local directory to save the merged model
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_name: Name for the model on HF Hub (e.g., "username/model-name")
        hub_token: Hugging Face token for authentication
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

    # Step 6: Push to Hugging Face Hub (optional)
    if push_to_hub and hub_model_name:
        print("🚀 Pushing to Hugging Face Hub...")

        # Login to Hugging Face
        if hub_token:
            login(token=hub_token)
        else:
            print("🔑 Please login to Hugging Face (you'll be prompted)...")
            login()

        # Push model
        merged_model.push_to_hub(
            hub_model_name,
            use_temp_dir=False,
            commit_message="Upload merged LoRA model",
        )

        # Push tokenizer
        tokenizer.push_to_hub(
            hub_model_name, use_temp_dir=False, commit_message="Upload tokenizer"
        )

        print(
            f"🎉 Model successfully pushed to https://huggingface.co/{hub_model_name}"
        )

    print("🎯 Merge process completed!")
    return merged_model, tokenizer


# Example usage
if __name__ == "__main__":
    # Configuration - Update these paths and names
    ADAPTER_PATH = (
        "./checkpoints/checkpoint-6000"  # Path to your LoRA adapter checkpoint
    )
    OUTPUT_DIR = (
        "./checkpoints/checkpoint-6000/merged-model"  # Where to save the merged model
    )
    HUB_MODEL_NAME = "Paulescu/LFM2-350M-ChessInstruct"  # HF Hub model name
    PUSH_TO_HUB = False  # Set to False if you only want local merge

    # Optional: Set your HF token (or login interactively)
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set this or use interactive login

    try:
        # Merge the adapter
        merged_model, tokenizer = merge_lora_adapter_to_base_model(
            adapter_path=ADAPTER_PATH,
            output_dir=OUTPUT_DIR,
            push_to_hub=PUSH_TO_HUB,
            hub_model_name=HUB_MODEL_NAME,
            hub_token=HF_TOKEN,
        )

        # Create a model card
        peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        # create_model_card(OUTPUT_DIR, peft_config.base_model_name_or_path,
        # ADAPTER_PATH)
        print("🎊 All done! Your merged model is ready to use.")

    except Exception as e:
        print(f"❌ Error during merge process: {str(e)}")
        raise
