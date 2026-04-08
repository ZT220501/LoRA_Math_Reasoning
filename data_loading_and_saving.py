import os
from datasets import load_dataset, load_from_disk, DatasetDict

save_dir = "dataset/gsm8k"  # put this inside your repo

def get_splits(cache_dir: str = save_dir):
    dataset = load_dataset("openai/gsm8k", "main")
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_set = train_val["train"]    # Training data for LoRA fine tuning
    val_set = train_val["test"]     # Validation data for LoRA fine tuning
    test_set  = dataset["test"]     # Test data for evaluation
    ds = DatasetDict({
        "train": train_set,
        "val": val_set,
        "test": test_set,
    })
    ds.save_to_disk(save_dir)

    return train_set, val_set, test_set

train_set, val_set, test_set = get_splits()
print(train_set, val_set, test_set)