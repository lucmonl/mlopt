import os
import json
from datasets import load_dataset, Dataset

from arch.llama import get_llama_model_and_formats

import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

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



def load_fedllm_bench_federated(model_name, task_name, batch_size, client_num, model_params, do_eval):
    DATASETS_FOLDER = os.environ["DATA_HOME"]

    data_dir = DATASETS_FOLDER + f"FedLLM-Bench-Data/Fed-{task_name}/"
    
    if task_name == "ChatbotIT":
        data_dir += "chatbotIT_237c_6k.json.json"
    else:
        raise NotImplementedError
    print("loading cached dataset: {data_dir}")
    client_datasets = get_fed_datasets(data_dir)
    print("Client Num", len(client_datasets))
    assert client_num == len(client_datasets)
    print("Sample: ")
    print(client_datasets[0][0])

    if 'llama' in model_name:
        model, tokenizer, formatting_prompts_func = get_llama_model_and_formats(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    tokenized_datasets = []
    for client_dataset in client_datasets:
        tokenized_dataset = client_dataset.map(formatting_prompts_func)
        tokenized_datasets.append(tokenized_dataset)

    train_dataset = tokenized_datasets[0] # dummy dataset
    if "train_size" in model_params:
        max_train_samples = min(len(train_dataset), model_params["train_size"])
        train_dataset = train_dataset.select(range(max_train_samples))

    analysis_size = max(batch_size, 128)
    analysis_dataset = train_dataset.select(range(analysis_size))

    if do_eval:
        eval_dataset = analysis_dataset
    else:
        eval_dataset = analysis_dataset
    #if data_args.max_eval_samples is not None:
    if "train_size" in model_params:
        max_eval_samples = min(len(eval_dataset), model_params["train_size"])
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    analysis_eval_dataset = eval_dataset.select(range(analysis_size))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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
        preds, labels = eval_preds
        
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
        result = rouge_metric.compute(
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
    training_args=TrainingArguments(output_dir="output/", per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size)

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

    return model, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params

