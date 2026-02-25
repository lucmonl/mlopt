
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)
import os

DATA_FOLDER = os.environ["DATA_HOME"]
model_id = "meta-llama/Llama-3.1-8B-Instruct"
remote_size = '10B'
seq_len = 1024
# 1. Load Tokenizer & Quantization Config
tokenizer = AutoTokenizer.from_pretrained(model_id)


import hashlib
from datasets import Dataset

def with_text_hash(example):
    h = hashlib.sha1(example["text"].encode("utf-8")).hexdigest()
    return {"_text_hash": h}


def tokenize_with_eos(example):
    ids = tokenizer(
        example["text"],
        truncation=True,
        max_length=seq_len,
        add_special_tokens=False,
    )["input_ids"]

    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
    }

raw_ds = load_dataset("HuggingFaceFW/fineweb", name=f"sample-{remote_size}T", split="train")
print(f"Length of raw dataset: {len(raw_ds)}")
"""
hashed = raw_ds.map(
    with_text_hash,
    num_proc=8,
)

# Deduplicate
hashed = hashed.unique("_text_hash")

# Drop hash column
dedup_ds = hashed.remove_columns("_text_hash")

"""
seen = set()

def keep_first(example):
    h = hashlib.sha1(example["text"].encode("utf-8")).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True

dedup_ds = raw_ds.filter(
    keep_first,
    num_proc=1,   # MUST be 1 for correctness
    desc="Exact dedup (keep one copy)",
)
print(f"Seen samples: {len(seen)}")
print(f"After dedup: {len(dedup_ds)}")

tok_ds = dedup_ds.map(
    tokenize_with_eos,
    remove_columns=dedup_ds.column_names,
    num_proc=8,
    desc="Tokenize + EOS",
)
print("Type after tokenization: ", type(tok_ds))

ex = next(iter(tok_ds))
assert ex["input_ids"][-1] == tokenizer.eos_token_id

SAVE_DIR = DATA_FOLDER + f"fineweb{remote_size}/{model_id}/"

os.makedirs(SAVE_DIR, exist_ok=True)

tok_ds.save_to_disk(SAVE_DIR + "fineweb_dedup_eos.jsonl")
"""
tok_ds.to_json(
    SAVE_DIR + "fineweb_dedup_eos.jsonl",
    orient="records",
    lines=True,
)
"""