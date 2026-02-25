import os
import re
import unicodedata
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def normalize(text: str) -> str:
    """
    Canonical normalization for exact dedup.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def hash_text(text: str) -> str:
    """
    SHA1 hash of normalized text.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def exact_dedup(dataset, text_key="text"):
    """
    Streaming exact deduplication.
    Keeps first occurrence.
    """
    seen = set()

    for example in dataset:
        text = example[text_key]
        norm = normalize(text)
        h = hash_text(norm)

        if h in seen:
            continue

        seen.add(h)
        yield example

DATA_FOLDER = os.environ["DATA_HOME"]

#dataset = load_dataset(DATA_FOLDER + "fineweb10B", streaming=True)

fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

deduped = exact_dedup(fw, text_key="text")

# Example: write to disk as JSONL
import json

with open(DATA_FOLDER + "fineweb_dedup/fineweb_10BT_dedup.jsonl", "w") as f:
    for ex in tqdm(deduped):
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")