import os
from datasets import load_dataset, load_from_disk, DatasetDict

save_dir = "dataset/gsm8k"  # put this inside your repo

def get_splits(cache_dir: str = save_dir):
    dataset = load_dataset("openai/gsm8k", "main")
    train_set = dataset["train"]
    val_set = dataset["test"]
    test_set = dataset["test"]
    ds = DatasetDict({
        "train": train_set,
        "val": val_set,
        "test": test_set,
    })
    ds.save_to_disk(save_dir)

    return train_set, val_set, test_set

train_set, val_set, test_set = get_splits()
print(train_set, val_set, test_set)