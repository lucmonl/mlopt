import os
import json
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

from arch.llama import get_llama_model_and_formats

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

def load_fineweb_federated(model_name, task_name, batch_size, client_num, model_params, do_eval, init_weights):
    DATASETS_FOLDER = os.environ["DATA_HOME"]

    if task_name == "10B":
        local_dir = "fineweb10B"
        remote_name = "sample-10BT"
    elif task_name == "100B":
        local_dir = "fineweb100B"
        remote_name = "sample-100BT"

    dataset = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    print("Number of Samples in the dataset: ", len(dataset))


    if 'llama' in model_name:
        model, tokenizer, formatting_prompts_func = get_llama_model_and_formats(model_name)
    else:
        raise NotImplementedError
    
    if init_weights:
        print("Model weights are re-initialized.")
        #model_params = model_params | {"init": "weights"}
        model_params["init"] = "weights"
        #model.init_weights()
        model.apply(model._init_weights)

    """
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    tokenized_datasets = []
    for client_dataset in client_datasets:  #the last dataset is reserved for evaluation, for now TODO: replace it with standard benchmarks like Vicuna
        tokenized_dataset = client_dataset.map(formatting_prompts_func,
                                               fn_kwargs={
                                                    "tokenizer": tokenizer, 
                                                    "max_length": 1024
                                                },)
        tokenized_datasets.append(tokenized_dataset)

    train_dataset = tokenized_datasets[0] # dummy dataset
    if "train_size" in model_params:
        max_train_samples = min(len(train_dataset), model_params["train_size"])
        train_dataset = train_dataset.select(range(max_train_samples))

    train_dataset = train_dataset.map(formatting_prompts_func,
                                fn_kwargs={
                                    "tokenizer": tokenizer, 
                                    "max_length": 1024
                                },)

    analysis_size = min(max(batch_size, 128), len(train_dataset))
    analysis_dataset = train_dataset.select(range(analysis_size))

    tokenized_datasets = recombine_datasets_evenly(tokenized_datasets, client_num+1) #last one for eval_dataset
    if do_eval:
        eval_dataset = tokenized_datasets[-1]
    else:
        eval_dataset = tokenized_datasets[-1]
    tokenized_datasets = tokenized_datasets[:-1]
    #if data_args.max_eval_samples is not None:

    if "train_size" in model_params:
        max_eval_samples = min(len(eval_dataset), model_params["train_size"])
        max_eval_samples = min(max_eval_samples, 128)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    max_eval_samples = min(len(eval_dataset), 128)
    eval_dataset = eval_dataset.select(range(max_eval_samples))
    print("Number of samples in eval dataset: {}".format(len(eval_dataset)))
    eval_analysis_size = min(max(batch_size, 128), len(eval_dataset))
    analysis_eval_dataset = eval_dataset.select(range(eval_analysis_size))
    """

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    """
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    """
    
    # Initialize our Trainer
    #training_args=TrainingArguments(output_dir="output/", per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size)
    """
    training_args = TrainingArguments(
        output_dir="output/",
        per_device_train_batch_size=batch_size, # Kept low to avoid OOM in full precision
        gradient_accumulation_steps=1, # Increase this to simulate larger batch size
        learning_rate=2e-5,            # Lower LR for full fine-tuning, dummy
        weight_decay=0.01,
        logging_steps=1,
        max_steps=100,
        bf16=True,                     # Required for Llama 3.1 stability
        save_strategy="steps",
        save_steps=50,
        optim="adamw_torch_fused",     # Fused optimizer is faster
        gradient_checkpointing=True    # CRITICAL: Saves VRAM by recomputing activations
    )
    """

    # train/val split
    dataset = dataset.shuffle(seed=42)
    eval_dataset = dataset.select(range(256))
    train_dataset = dataset.select(range(256, len(dataset)))
    analysis_dataset = train_dataset.select(range(256))
    analysis_eval_dataset = eval_dataset


    sft_config = SFTConfig(
        output_dir="output/",
        max_seq_length=1024,      # Context window size
        packing=True,             # THE MAGIC SWITCH
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        learning_rate=2e-4,
        max_steps=100,
        save_steps=50,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        # No need for a custom collator; SFTTrainer handles it when packing=True
    )

    data_params = {"compute_acc": False}
    train_loader = trainer.get_train_dataloader()
    test_loader = trainer.get_eval_dataloader()
    val_loader = test_loader # val/test already been split

    client_loaders = []
    for i in range(client_num):
        client_loaders.append(train_loader)

    trainer = SFTTrainer(
        model=model,
        train_dataset=analysis_dataset,
        eval_dataset=analysis_eval_dataset,
        args=sft_config,
        # No need for a custom collator; SFTTrainer handles it when packing=True
    )

    """
    print("first time sample")
    for batch_idx, item in enumerate(client_loaders[0]):
        if batch_idx > 0:
            continue
        else:
            print(item["labels"])
        
    
    print("second time sample")
    for batch_idx, item in enumerate(client_loaders[0]):
        if batch_idx > 0:
            continue
        else:
            print(item["labels"])
    """
    analysis_loader = trainer.get_train_dataloader()
    analysis_test_loader = trainer.get_eval_dataloader()
    transform_to_one_hot = False
    C = None

    print("Number of samples in test_dataset: ", len(test_loader.dataset))

    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params

