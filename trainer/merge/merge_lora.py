import os
import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def merge_lora(base_path: str, adapter_path: str, output_path: str, dtype_str="auto"):
    """
    Efficiently merge LoRA weights into a base model and save the result.

    Args:
    - base_path: str, Base model directory.
    - adapter_path: str, LoRA/PEFT adapter directory.
    - output_path: str, Where to save the merged model.
    - dtype_str: str, Model param dtype ("auto", "float16", "bfloat16").
    """
    print(f"Loading base model from: {base_path}")
    if dtype_str == "bfloat16": dtype = torch.bfloat16
    elif dtype_str == "float16": dtype = torch.float16
    else: dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"[ERROR] Couldn't load adapter: {e}")
        return

    print("Merging PEFT weights and unloading adapter layers...")
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="10GB")

    print("Saving processor (tokenizer & image)...")
    processor = AutoProcessor.from_pretrained(base_path)
    processor.save_pretrained(output_path)

    print(f"Merge complete! All files saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base QwenVL model efficiently.")
    parser.add_argument("--base_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Base model directory.")
    parser.add_argument("--adapter_path", type=str, default="output/mineru_style_finetune", help="LoRA adapter directory.")
    parser.add_argument("--output_path", type=str, default="output/mineru_merged_model", help="Final saved model directory.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Model parameter dtype.")
    args = parser.parse_args()
    merge_lora(args.base_path, args.adapter_path, args.output_path, args.dtype)