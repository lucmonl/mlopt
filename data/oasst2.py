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

def get_local_datasets(file_path):
    dataset = load_dataset("json", data_files=file_path)['train']
    return [dataset]

def get_fed_datasets(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    local_datasets = []
    for value in data.values():
        dataset = Dataset.from_list(value)
        local_datasets.append(dataset)
    
    return local_datasets

def recombine_datasets_evenly(datasets_list, M):
    N = len(datasets_list)
    assert M < N, "M must be smaller than N"

    group_size = N // M
    remainder = N % M

    new_datasets = []
    idx = 0

    for i in range(M):
        size = group_size + (1 if i < remainder else 0)
        group = datasets_list[idx : idx + size]
        new_datasets.append(concatenate_datasets(group))
        idx += size

    return new_datasets

def load_oasst2_federated(model_name, task_name, batch_size, client_num, model_params, dtype, init_weights, max_length=None):
    DATA_FOLDER = os.environ["DATA_HOME"]
    if max_length is not None:
        SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_oasst2/{model_name}/length_{max_length}/")
    else:
        SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_oasst2/{model_name}/")

    # load the tokenized datasets
    print(SAVE_DIR)
    train_dataset = load_from_disk(SAVE_DIR + f"tokenized_oasst2_train_{task_name}.jsonl")
    eval_dataset = load_from_disk(SAVE_DIR + f"tokenized_oasst2_validation_{task_name}.jsonl")

    if 'llama' in model_name:
        model, tokenizer, formatting_prompts_func = get_llama_model_and_formats(model_name, dtype, model_params)
    else:
        raise NotImplementedError
    
    if init_weights:
        print("Model weights are re-initialized.")
        #model_params = model_params | {"init": "weights"}
        model_params["init"] = "weights"
        #model.init_weights()
        model.apply(model._init_weights)


    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    #train_dataset = tokenized_datasets[0] # dummy dataset
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

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds, labels = p
        
        # 1. Take the argmax to get the predicted token IDs
        # preds shape: (batch, seq_len, vocab_size) -> (batch, seq_len)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = np.argmax(preds, axis=-1)

        # 2. Replace -100 in labels (we don't want to decode prompt/padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # 3. Decode back to text
        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 4. Clean up strings (ROUGE expects newline separation for summaries/instructions)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # 5. Compute ROUGE
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
        per_device_train_batch_size=batch_size, # Kept low to avoid OOM in full precision
        gradient_accumulation_steps=1, # Increase this to simulate larger batch size
        learning_rate=2e-5,            # Lower LR for full fine-tuning, dummy
        weight_decay=0.01,
        logging_steps=1,
        max_steps=100,
        bf16=use_bf16,                     # Required for Llama 3.1 stability
        save_strategy="steps",
        save_steps=50,
        optim="adamw_torch_fused",     # Fused optimizer is faster
        gradient_checkpointing=True    # CRITICAL: Saves VRAM by recomputing activations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # if training_args.do_train else None,
        eval_dataset=eval_dataset, # if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    data_params = {"compute_acc": False}
    train_loader = trainer.get_train_dataloader()
    test_loader = trainer.get_eval_dataloader()
    val_loader = test_loader # val/test already been split

    train_iter = iter(train_loader)

    #client_loaders = []
    #for i in range(client_num):
    #    client_loaders.append(train_iter)
    client_loaders = [train_iter, train_loader]
    training_args=TrainingArguments(output_dir="output/", per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=analysis_dataset, # if training_args.do_train else None,
        eval_dataset=analysis_eval_dataset, # if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    analysis_loader = trainer.get_train_dataloader()
    analysis_test_loader = trainer.get_eval_dataloader()
    transform_to_one_hot = False
    C = None
    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params



def preprocess_oasst2(model_id, split="train", lang_filter="en", max_length=2048):
    from datasets import load_dataset
    from collections import defaultdict
    import numpy as np
    import os
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = load_dataset("OpenAssistant/oasst2")

    # dict: creativity → list of (prompt, response)
    train_ds = ds[split]
    train_ds = train_ds.filter(lambda m: m["lang"] == lang_filter)

    # Build a lookup table
    #multi_attribute_dict = {}
    chat_list = []
    id2msg = {m["message_id"]: m for m in train_ds}


    for msg in tqdm(train_ds):
        # Only consider assistant responses
        if msg["role"] != "assistant":
            continue

        if 'labels' not in msg or msg.get("labels") is None:
            continue
        
        # must have a parent user message
        parent_id = msg["parent_id"]
        if parent_id is None:
            continue
        
        parent = id2msg.get(parent_id)
        if parent is None or parent["role"] != "prompter":
            continue
        """
        chat = [
            {"role": "user", "content": parent.get('text')},
            {"role": "assistant", "content": msg.get('text')}
        ]
        """
        chat = {"instruction": parent.get('text'), "response": msg.get('text')}
        chat_list.append(chat)
        #creativity_list.append(creativity)

    #dataset = Dataset.from_dict({"chat": chat_list})
    #dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=True)})
    #dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=True)})
    dataset = Dataset.from_list(chat_list)
    tokenized_dataset = dataset.map(format_and_mask_instruction,
                                    fn_kwargs={
                                        "tokenizer": tokenizer,
                                        "max_length": max_length,
                                    },)

    DATA_FOLDER = os.environ["DATA_HOME"]
    SAVE_DIR = os.path.join(DATA_FOLDER, f"tokenized_oasst2/{model_id}/length_{max_length}/")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"saving {len(tokenized_dataset)} samples to {SAVE_DIR}")
    tokenized_dataset.save_to_disk(SAVE_DIR + f"tokenized_oasst2_{split}_{lang_filter}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OpenAssistant/oasst2 dataset.")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B",
        help="Model name or path for tokenizer.",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Max token length (prompt + response combined).",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to preprocess.",
    )
    parser.add_argument(
        "--lang_filter", type=str, default="en",
        help="Language filter for the dataset.",
    )

    args = parser.parse_args()
    preprocess_oasst2(args.model_name, args.split, args.lang_filter, args.max_length)
