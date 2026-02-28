import os
import json
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets

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

def load_fedllm_bench_federated(model_name, task_name, batch_size, client_num, model_params, do_eval, dtype, init_weights):
    DATASETS_FOLDER = os.environ["DATA_HOME"]

    data_dir = DATASETS_FOLDER + f"FedLLM-Bench-Data/Fed-{task_name}/"
    
    if task_name == "ChatbotIT":
        data_dir += "chatbotIT_237c_6k.json.json"
    else:
        raise NotImplementedError
    print("loading cached dataset: {data_dir}")
    client_datasets = get_fed_datasets(data_dir)
    print("Client Num", len(client_datasets))
    assert client_num+1 <= len(client_datasets)
    print("Sample: ")
    print(client_datasets[0][0])

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
    """
    eval_dataset = eval_dataset.map(formatting_prompts_func,
                                    fn_kwargs={
                                        "tokenizer": tokenizer, 
                                        "max_length": 1024
                                    },)
    """
    if "train_size" in model_params:
        max_eval_samples = min(len(eval_dataset), model_params["train_size"])
        max_eval_samples = min(max_eval_samples, 128)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    max_eval_samples = min(len(eval_dataset), 128)
    eval_dataset = eval_dataset.select(range(max_eval_samples))
    print("Number of samples in eval dataset: {}".format(len(eval_dataset)))
    eval_analysis_size = min(max(batch_size, 128), len(eval_dataset))
    analysis_eval_dataset = eval_dataset.select(range(eval_analysis_size))

    # Get the metric function
    """
    if model_params["task_name"] is not None:
        metric = evaluate.load("glue", model_params["task_name"], cache_dir=DATASETS_FOLDER)
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=DATASETS_FOLDER)
    else:
        metric = evaluate.load("accuracy", cache_dir=DATASETS_FOLDER)
    """
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

    client_loaders = []
    assert len(tokenized_datasets) == client_num
    for i in range(client_num):
        client_train = tokenized_datasets[i]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=client_train, # if training_args.do_train else None,
            eval_dataset=eval_dataset, # if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        client_loaders.append(trainer.get_train_dataloader())
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
    training_args=TrainingArguments(output_dir="output/", per_device_train_batch_size=analysis_size, per_device_eval_batch_size=analysis_size)
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

    print("Number of samples in test_dataset: ", len(test_loader.dataset))

    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params

