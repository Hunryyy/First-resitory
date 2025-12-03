import torch
import os
import yaml
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from trainer.dataset.custom_dataset import DocParseDataset  # 引入正确的Dataset
from torch.utils.data import DataLoader

# === CONFIGURATION (从 config.yaml 或默认值加载) ===
CONFIG_PATH = "trainer/config/config.yaml"
DEFAULT_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
    "data_path": "data/train_formatted.jsonl",  # 使用预处理后的数据
    "output_dir": "output/mineru_style_finetune",
    "max_length": 8192,  # 保持和配置一致
    "lora_rank": 64,
    "lora_alpha": 128,
    "batch_size": 1,
    "grad_acc_steps": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG

cfg = load_config()
# =================================================

def train():
    print("Loading base model (bfloat16 for stability)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg['model_name_or_path'],
        torch_dtype=torch.bfloat16,  # 优化：使用 bfloat16 (如果GPU支持)
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # 优化：扩展 LoRA Target 以增强结构和推理能力
    lora_config = LoraConfig(
        r=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(cfg['model_name_or_path'])

    print(f"Building training dataset from {cfg['data_path']}...")
    train_dataset = DocParseDataset(cfg['data_path'], processor) 

    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        num_train_epochs=cfg['num_train_epochs'],
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['grad_acc_steps'],
        bf16=True,  # 优化：使用 BF16
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=float(cfg['learning_rate']),
        weight_decay=0.01,
        report_to=[],
        gradient_checkpointing=True,  # 显存优化
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    print(f"Saving adapter weights and processor to {cfg['output_dir']}...")
    model.save_pretrained(cfg['output_dir'])
    processor.save_pretrained(cfg['output_dir'])
    print("Training complete.")

if __name__ == "__main__":
    train()

