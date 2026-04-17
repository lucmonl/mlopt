import os
import sys
import json
import numpy as np
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arch.llama import get_llama_model_and_formats, format_and_mask_instruction

import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EvalPrediction,
    DataCollatorForSeq2Seq
)
import evaluate
from tqdm import tqdm


def load_megascience_federated(model_name, batch_size, client_num, model_params, dtype, init_weights):
    DATA_FOLDER = os.environ["DATA_HOME"]
    SAVE_DIR = DATA_FOLDER + f"tokenized_megascience/{model_name}/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # load the tokenized datasets
    print(SAVE_DIR)
    train_dataset = load_from_disk(SAVE_DIR + "tokenized_megascience_train.jsonl")
    eval_dataset = load_from_disk(SAVE_DIR + "tokenized_megascience_validation.jsonl")

    if 'llama' in model_name:
        model, tokenizer, formatting_prompts_func = get_llama_model_and_formats(model_name, dtype, model_params)
    else:
        raise NotImplementedError

    if init_weights:
        print("Model weights are re-initialized.")
        model_params["init"] = "weights"
        model.apply(model._init_weights)

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    if "train_size" in model_params:
        max_train_samples = min(len(train_dataset), model_params["train_size"])
        train_dataset = train_dataset.select(range(max_train_samples))

    analysis_size = min(max(batch_size, 128), len(train_dataset))
    analysis_dataset = train_dataset.select(range(analysis_size))

    if "train_size" in model_params:
        max_eval_samples = min(len(eval_dataset), model_params["train_size"])
        max_eval_samples = min(max_eval_samples, 128)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    max_eval_samples = min(len(eval_dataset), 128)
    eval_dataset = eval_dataset.select(range(max_eval_samples))
    print("Number of samples in train dataset: {}".format(len(train_dataset)))
    print("Number of samples in eval dataset: {}".format(len(eval_dataset)))
    eval_analysis_size = min(max(batch_size, 128), len(eval_dataset))
    analysis_eval_dataset = eval_dataset.select(range(eval_analysis_size))

    # Get the metric function
    metric = evaluate.load("rouge")

    def compute_metrics(p: EvalPrediction):
        preds, labels = p

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = np.argmax(preds, axis=-1)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        return {k: round(v, 4) for k, v in result.items()}

    if dtype in ["default", "bf16"]:
        use_bf16 = True
    else:
        use_bf16 = False

    training_args = TrainingArguments(
        output_dir="output/",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=100,
        bf16=use_bf16,
        save_strategy="steps",
        save_steps=50,
        optim="adamw_torch_fused",
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    data_params = {"compute_acc": False}
    train_loader = trainer.get_train_dataloader()
    test_loader = trainer.get_eval_dataloader()
    val_loader = test_loader  # val/test already been split

    train_iter = iter(train_loader)

    client_loaders = [train_iter, train_loader]
    training_args = TrainingArguments(output_dir="output/", per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=analysis_dataset,
        eval_dataset=analysis_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    analysis_loader = trainer.get_train_dataloader()
    analysis_test_loader = trainer.get_eval_dataloader()
    transform_to_one_hot = False
    C = None
    data_params["compute_ex_score"] = evaluate_func
    data_params["analysis_dataset"] = analysis_dataset
    data_params["test_dataset"] = analysis_eval_dataset
    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params


def evaluate_func(model, eval_dataset, device, tokenizer=None):
    max_new_tokens= 1024
    import torch

    example = eval_dataset[0]

    instruction = example["instruction"]
    reference_answer = example["answer"]

    # The dataset is already tokenized; extract the prompt portion using labels.
    # Labels are -100 for the user/prompt tokens and real IDs for the assistant tokens.
    labels = example["labels"]
    prompt_len = next(i for i, l in enumerate(labels) if l != -100)
    input_ids = torch.tensor(example["input_ids"][:prompt_len], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("=" * 60)
    print("INSTRUCTION:")
    print(instruction)
    print("-" * 60)
    print("MODEL OUTPUT:")
    print(output_text)
    print("-" * 60)
    print("REFERENCE ANSWER:")
    print(reference_answer)
    print("=" * 60)

    return {"instruction": instruction, "output": output_text, "reference_answer": reference_answer}


def preprocess_megascience(model_id, val_ratio=0.05, test_ratio=0.05, seed=42, max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = load_dataset("MegaScience/MegaScience", split="train")
    ds = ds.select(range(30))

    # Build list of dicts: instruction (question), response (answer for training), reference_answer (true answer)
    sample_list = []
    for item in tqdm(ds):
        sample = {
            "instruction": item["question"],
            "response": item["answer"],                  # model-generated answer is the training target
            "answer": item["reference_answer"],  # ground-truth answer kept for reference
        }
        sample_list.append(sample)

    print(f"Total samples before filtering: {len(sample_list)}")
    sample_list = sample_list
    # Filter out samples whose combined instruction + response token length exceeds max_length
    filtered = []
    for sample in tqdm(sample_list, desc="filtering by length"):
        n_instruction = len(tokenizer.encode(sample["instruction"], add_special_tokens=False))
        n_response = len(tokenizer.encode(sample["response"], add_special_tokens=False))
        if n_instruction + n_response <= max_length:
            filtered.append(sample)
    sample_list = filtered

    print(f"Total samples after filtering (max_length={max_length}): {len(sample_list)}")

    dataset = Dataset.from_list(sample_list)

    # Split into train / val / test
    # First cut off test set, then split remainder into train/val
    tmp = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_ds = tmp["test"]
    val_adjusted = val_ratio / (1.0 - test_ratio)
    tv = tmp["train"].train_test_split(test_size=val_adjusted, seed=seed)
    train_ds = tv["train"]
    val_ds = tv["test"]

    def tokenize(ds):
        return ds.map(
            format_and_mask_instruction,
            fn_kwargs={"tokenizer": tokenizer, "max_length": 2048},
        )

    DATA_FOLDER = os.environ["DATA_HOME"]
    SAVE_DIR = DATA_FOLDER + f"tokenized_megascience/{model_id}/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenized_train = tokenize(train_ds)
    print(f"saving {len(tokenized_train)} train samples to {SAVE_DIR}")
    tokenized_train.save_to_disk(SAVE_DIR + "tokenized_megascience_train.jsonl")

    tokenized_val = tokenize(val_ds)
    print(f"saving {len(tokenized_val)} validation samples to {SAVE_DIR}")
    tokenized_val.save_to_disk(SAVE_DIR + "tokenized_megascience_validation.jsonl")

    tokenized_test = tokenize(test_ds)
    print(f"saving {len(tokenized_test)} test samples to {SAVE_DIR}")
    tokenized_test.save_to_disk(SAVE_DIR + "tokenized_megascience_test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MegaScience/MegaScience dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of data to use as validation.")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Fraction of data to use as test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max combined token length for instruction + response.")

    args = parser.parse_args()
    preprocess_megascience(
        'meta-llama/Llama-3.2-3B',
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_length=args.max_length,
    )