from datasets import load_from_disk
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import torch
from datetime import datetime
import json

torch.manual_seed(0)

# Register the current config to the register.json file
def register_config(config, timestamp):
    # Register the current config to the register.json file
    output_dir = f"training_output_gsm8k/config_{timestamp}"
    with open('training_output_gsm8k/register.json', 'r') as j_file:
        existing_config = json.load(j_file)
    # Create a new config dictionary and update the existing config
    new_config = {
        f"config_{now.strftime('%Y_%m_%d_%H_%M_%S')}": config
    }
    existing_config.update(new_config)
    # Serialize the existing config to string, then insert blank lines between top-level keys
    json_str = json.dumps(existing_config, indent=2)
    lines = json_str.split('\n')
    result = []
    for line in lines:
        # Top-level keys are indented by exactly 2 spaces
        if line.startswith('  "') and result and result[-1] != '{':
            result.append('')   # blank line before each new top-level key
        result.append(line)
    # Write the existing config to the register.json file
    with open('training_output_gsm8k/register.json', 'w') as config_register_file:
        config_register_file.write('\n'.join(result))

# Load the model and tokenizer
def load_model_and_tokenizer(config):
    # Load the model and tokenizer
    model_name = config['model']['name']
    model_dtype = config['model']['torch_dtype']
    print("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=model_dtype, device_map="auto")
    tokenizer.padding_side = "left"                     # Left padding for inference
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, device_map="auto")
    return model, tokenizer


if __name__ == "__main__":
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Check the current timestamp for saving purpose
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # Register the config to the register.json file
    register_config(config, timestamp)
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # LoRA configuration
    lora_config = LoraConfig(**config['lora'])
    # Training configuration
    output_dir = f"training_output_gsm8k/config_{timestamp}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        **config['sft']
    )

    # Define the trainer
    train_set = load_from_disk("dataset/gsm8k/train_converted")
    val_set = load_from_disk("dataset/gsm8k/val_converted")
    print(train_set, val_set)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    # Model training
    saving_dir = f"{output_dir}/final_result"
    trainer.train()
    trainer.model.save_pretrained(saving_dir)