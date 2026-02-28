import os
import json
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
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

def load_fineweb_federated(model_name, task_name, batch_size, client_num, model_params, dtype, init_weights):
    DATASETS_FOLDER = os.environ["DATA_HOME"]

    if task_name == "10B":
        local_dir = "fineweb10B"
        remote_name = "sample-10BT"
    elif task_name == "100B":
        local_dir = "fineweb100B"
        remote_name = "sample-100BT"

    #dataset = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    DATA_FOLDER = os.environ["DATA_HOME"]
    PROCESSED_DATA_DIR = DATA_FOLDER + f"fineweb{task_name}/{model_name}/"
    dataset = load_from_disk(PROCESSED_DATA_DIR + "fineweb_dedup_eos.jsonl")
    print("Number of Samples in the dataset: ", len(dataset))

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

    # train/val split
    dataset = dataset.shuffle(seed=42)
    eval_dataset = dataset.select(range(256))
    train_dataset = dataset.select(range(256, len(dataset)))
    analysis_dataset = train_dataset.select(range(256))
    analysis_eval_dataset = eval_dataset

    """
    train_dataset = train_dataset.select(range(20))
    eval_dataset = eval_dataset.select(range(20))
    analysis_dataset= analysis_dataset.select(range(20))
    analysis_eval_dataset = analysis_eval_dataset.select(range(20))
    """

    if dtype in ["default", "bf16"]:
        use_bf16 = True
    else:
        use_bf16 = False

    sft_config = SFTConfig(
        output_dir="output/",
        max_length=None,      # Context window size
        packing=False,
        #packing=True,             # THE MAGIC SWITCH
        dataset_text_field=None,  # IMPORTANT
        per_device_train_batch_size=batch_size,
        learning_rate=2e-4,
        max_steps=100,
        save_steps=50,
        bf16=use_bf16,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        #add_eos_token=False,      # IMPORTANT
        # No need for a custom collator; SFTTrainer handles it when packing=True
    )

    data_params = {"compute_acc": False}
    train_loader = trainer.get_train_dataloader()
    test_loader = trainer.get_eval_dataloader()
    val_loader = test_loader # val/test already been split

    #train_iter init
    train_iter = iter(train_loader)

    #client_loaders = []
    #for i in range(client_num):
    #    client_loaders.append(train_iter)
    client_loaders = [train_iter, train_loader]

    trainer = SFTTrainer(
        model=model,
        train_dataset=analysis_dataset,
        eval_dataset=analysis_eval_dataset,
        args=sft_config,
        # No need for a custom collator; SFTTrainer handles it when packing=True
    )
    """
    print("first time sample")
    data_iter = client_loaders[0]
    item = next(data_iter)
    print(item)

    print("second time sample")
    data_iter = client_loaders[1]
    item = next(data_iter)
    print(item)
    """
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

