from transformers import AutoTokenizer
from datasets import load_from_disk
import yaml
import re

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the tokenizer
model_name = config['model']['name']
model_dtype = config['model']['torch_dtype']
tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=model_dtype)

def convert_gsm8k_answer(example):
    answer = example["answer"]

    m = re.search(r"^(.*?)(?:\n)?####\s*(.+?)\s*$", answer, flags=re.DOTALL)
    if m is None:
        raise ValueError(f"Could not parse answer:\n{answer}")

    reasoning = m.group(1).rstrip()
    final_answer = m.group(2).strip()

    if reasoning:
        new_answer = reasoning + "\n**Final Answer:**\n" + f"$$\n\\boxed{{{final_answer}}}\n$$"
    else:
        new_answer = rf"\boxed{{{final_answer}}}"

    example["answer"] = new_answer
    return example

# Define prompt format for training and validation
def format_prompt_training(example):
    """For SFT — includes question + full CoT answer."""
    SYSTEM_PROMPT = "You are a careful and rigorous mathematician. Solve problems step by step, and put your final answer within \\boxed{}. Do not output anything after the boxed answer."
    example = convert_gsm8k_answer(example)
    chat = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    )
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

# Load the saved training and validation datasets
save_dir = "dataset/gsm8k"
ds = load_from_disk(save_dir)
train_set = ds["train"]
val_set = ds["val"]

# Transform the dataset with correct prompt format for training and validation before fine-tuning
train_set = train_set.map(format_prompt_training, remove_columns=train_set.column_names)
val_set = val_set.map(format_prompt_training, remove_columns=val_set.column_names)

train_set.save_to_disk("dataset/gsm8k/train_converted")
val_set.save_to_disk("dataset/gsm8k/val_converted")

print(len(train_set), len(val_set))

print("Training set example: ", train_set[0])
print("Validation set example: ", val_set[0])
