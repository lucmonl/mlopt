import os
import sys
import json
import numpy as np
import argparse
from datasets import load_dataset, Dataset, load_from_disk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arch.llama import get_llama_model_and_formats

import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
)
import evaluate
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

# Mimics what Llama-3.1/3.2-Instruct's apply_chat_template(tools=...) builds:
# a system turn that lists available tools, then a user turn, then the
# assistant's <tool_call> response.
_SYSTEM_TEMPLATE = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions.\n\n"
    "Here are the available tools:\n"
    "<tools>\n"
    "{tools}\n"
    "</tools>\n\n"
    "For each function call, return a JSON object with the function name and "
    "arguments within <tool_call></tool_call> XML tags as follows:\n"
    "<tool_call>\n"
    '{{"name": "function_name", "arguments": {{...}}}}\n'
    "</tool_call>"
)


def _build_prompt_and_response(example):
    """
    Build the prompt string (system + user turns, open assistant header) and
    the response string (assistant tool-call content + eot).

    Mirrors the token sequence that apply_chat_template(messages, tools=tools,
    add_generation_prompt=True, enable_thinking=False) would emit for a
    Llama-3.1/3.2-Instruct tokenizer.
    """
    tools = json.loads(example["tools"])
    tools_str = json.dumps(tools, indent=2)
    system_content = _SYSTEM_TEMPLATE.format(tools=tools_str)

    query = example["query"]

    answers = json.loads(example["answers"])  # list of {"name": ..., "arguments": ...}
    response_parts = [
        f"<tool_call>\n{json.dumps(call)}\n</tool_call>" for call in answers
    ]
    response_body = "\n".join(response_parts)

    # --- prompt (everything up to and including the open assistant header) ---
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # --- response (what the model should generate) ---
    response = f"{response_body}<|eot_id|>"

    return prompt, response


# ---------------------------------------------------------------------------
# Tokenisation / masking
# ---------------------------------------------------------------------------

def format_and_mask_func_call(example, tokenizer, max_length=2048):
    """
    Tokenise one sample, masking the prompt tokens with -100 so the model
    only learns from the assistant's tool-call response.
    """
    prompt, response = _build_prompt_and_response(example)
    #print(prompt)
    #print(response)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": list(map(int, input_ids)),
        "labels": list(map(int, labels)),
        "attention_mask": list(map(int, attention_mask)),
    }


# ---------------------------------------------------------------------------
# Preprocessing (download → filter → split → tokenise → save)
# ---------------------------------------------------------------------------

def preprocess_func_call(model_id, val_ratio=0.05, test_ratio=0.05, seed=42, max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    ds = ds.select(np.arange(100))
    # Collect raw samples
    sample_list = []
    for item in tqdm(ds, desc="loading dataset"):
        sample_list.append({
            "query":   item["query"],
            "answers": item["answers"],
            "tools":   item["tools"],
        })

    print(f"Total samples before filtering: {len(sample_list)}")

    # Filter samples whose combined prompt + response exceeds max_length tokens
    filtered = []
    for sample in tqdm(sample_list, desc="filtering by length"):
        #try:
        prompt, response = _build_prompt_and_response(sample)
        n_prompt   = len(tokenizer.encode(prompt,   add_special_tokens=False))
        n_response = len(tokenizer.encode(response, add_special_tokens=False))
        if n_prompt + n_response <= max_length:
            filtered.append(sample)
        #print(n_prompt + n_response)
        """
        except Exception as e:
            # skip malformed JSON entries
            print(e)
            continue
        """

    print(f"Total samples after filtering (max_length={max_length}): {len(filtered)}")

    dataset = Dataset.from_list(filtered)
    # Train / val / test split (same strategy as megascience.py)
    tmp = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_ds = tmp["test"]
    val_adjusted = val_ratio / (1.0 - test_ratio)
    tv = tmp["train"].train_test_split(test_size=val_adjusted, seed=seed)
    train_ds = tv["train"]
    val_ds   = tv["test"]

    def tokenize(split_ds):
        return split_ds.map(
            format_and_mask_func_call,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )

    DATA_FOLDER = os.environ["DATA_HOME"]
    SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_func_call/{model_id}/length_{max_length}/")
    os.makedirs(SAVE_DIR, exist_ok=True)

    for name, split_ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
        tokenized = tokenize(split_ds)
        path = os.path.join(SAVE_DIR, f"tokenized_func_call_{name}.jsonl")
        print(f"saving {len(tokenized)} {name} samples to {path}")
        #tokenized.save_to_disk(path)
    """
    import torch
    from transformers import AutoModel
    model = AutoModelForCausalLM.from_pretrained(model_id)
    example = tokenized[0]
    print(example)
    evaluate_func(model, tokenized, torch.device("cpu"), tokenizer)
    """

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_func(model, eval_dataset, device, tokenizer=None):
    import torch
    for i in range(min(len(eval_dataset), 5)):
        example = eval_dataset[i]

        # Recover the prompt portion (labels == -100 for prompt tokens)
        labels = example["labels"]
        prompt_len = next(i for i, l in enumerate(labels) if l != -100)
        input_ids = torch.tensor(
            example["input_ids"][:prompt_len], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"Example {i}")
        print("-" * 60)
        #print("MODEL INPUT:")
        #print(input_text)
        #print("-" * 60)
        print("MODEL OUTPUT:")
        print(output_text)
        print("-" * 60)

    return 0


# ---------------------------------------------------------------------------
# Federated loader (mirrors load_megascience_federated)
# ---------------------------------------------------------------------------

def load_func_call_federated(model_name, batch_size, client_num, model_params, dtype, init_weights, max_length=None):
    DATA_FOLDER = os.environ["DATA_HOME"]
    if max_length is not None:
        SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_func_call/{model_name}/length_{max_length}/")
    else:
        SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_func_call/{model_name}/")

    print(SAVE_DIR)
    train_dataset = load_from_disk(SAVE_DIR + "tokenized_func_call_train.jsonl")
    eval_dataset  = load_from_disk(SAVE_DIR + "tokenized_func_call_validation.jsonl")

    if 'llama' in model_name:
        model, tokenizer, _ = get_llama_model_and_formats(model_name, dtype, model_params)
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
    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in eval dataset:  {len(eval_dataset)}")
    eval_analysis_size = min(max(batch_size, 128), len(eval_dataset))
    analysis_eval_dataset = eval_dataset.select(range(eval_analysis_size))

    metric = evaluate.load("rouge")

    def compute_metrics(p: EvalPrediction):
        preds, labels = p
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(np.argmax(preds, axis=-1), skip_special_tokens=True)
        labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    use_bf16 = dtype in ["default", "bf16"]

    training_args = TrainingArguments(
        output_dir="output/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=100,
        bf16=use_bf16,
        save_strategy="steps",
        save_steps=50,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
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
    test_loader  = trainer.get_eval_dataloader()
    val_loader   = test_loader

    train_iter = iter(train_loader)
    client_loaders = [train_iter, train_loader]

    analysis_args = TrainingArguments(
        output_dir="output/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    analysis_trainer = Trainer(
        model=model,
        args=analysis_args,
        train_dataset=analysis_dataset,
        eval_dataset=analysis_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    analysis_loader      = analysis_trainer.get_train_dataloader()
    analysis_test_loader = analysis_trainer.get_eval_dataloader()

    transform_to_one_hot = False
    C = None
    data_params["compute_ex_score"] = evaluate_func
    data_params["analysis_dataset"] = analysis_dataset
    data_params["test_dataset"]     = analysis_eval_dataset

    return (
        model, tokenizer,
        train_loader, client_loaders,
        val_loader, test_loader,
        analysis_loader, analysis_test_loader,
        C, transform_to_one_hot, data_params,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Salesforce/xlam-function-calling-60k dataset."
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B",
        help="Model name or path for tokenizer.",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Max token length (prompt + response combined).",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Fraction of data to use as validation.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.05,
        help="Fraction of data to use as test.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split.",
    )

    args = parser.parse_args()
    preprocess_func_call(
        args.model_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_length=args.max_length,
    )
