"""
adapted from https://github.com/premAI-io/premsql/blob/main/premsql/tuner/full.py
"""

from datetime import datetime
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset


from premsql.datasets import (
    BirdDataset,
    SpiderUnifiedDataset,
    DomainsDataset,
    GretelAIDataset
)

from premsql.executors.from_sqlite import SQLiteExecutor
from premsql.datasets import Text2SQLDataset
from premsql.datasets.error_dataset import ErrorDatasetGenerator
from premsql.datasets.collator import DataCollatorForSupervisedDataset
from premsql.tuner.config import DefaultTrainingArguments
from premsql.datasets.base import Text2SQLBaseDataset
from premsql.evaluator.base import BaseExecutor, Text2SQLEvaluator
from premsql.generators.huggingface import Text2SQLGeneratorHF

DATA_HOME = os.environ["DATA_HOME"]
HF_HOME = os.environ["HF_HOME"]

def evaluate_spider(model, eval_dataset, device):
    model = Text2SQLGeneratorHF(
        model_or_name_or_path=model,
        type="test",
        device=device,
        experiment_name="evaluate_spider"
    )
    responses = model.generate_and_save_results(
        dataset=eval_dataset, temperature=0.1, max_new_tokens=256, force=True
    )
    executor = SQLiteExecutor()
    evaluator = Text2SQLEvaluator(
        executor=executor, experiment_path=DATA_HOME
    )
    ex_score = evaluator.execute(
        metric_name="accuracy",
        model_responses=responses,
        filter_by=None,
    )
    return ex_score.get("overall")

def get_dataloader(train_dataset, eval_dataset, data_collator, model, tokenizer, training_arguments):
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module,
    )

    train_loader = trainer.get_train_dataloader()
    if eval_dataset is None:
        return train_loader
    else:
        val_loader = trainer.get_eval_dataloader()
        return train_loader, val_loader


def load_spider(model_name, batch_size):
    base_model = model_name #"deepseek-ai/deepseek-coder-1.3b-instruct"
    hf_token = None
    analysis_size = max(batch_size, 128)
    analysis_ind = range(analysis_size)

    spider_train = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=True)
    spider_analysis = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, rows=analysis_ind, tokenize=False)
    spider_analysis_token = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, rows=analysis_ind, tokenize=True)
    #spider_analysis = spider_train.select(torch.arange(analysis_size))
    #spider_train = spider_train.setup_dataset(model_name_or_path=base_model, num_rows=100)
    #spider_analysis = spider_analysis.setup_dataset(model_name_or_path=base_model)

    spider_val = SpiderUnifiedDataset(split="validation", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=False)
    spider_val_token = SpiderUnifiedDataset(split="validation", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=True)
   
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_size="right", token=hf_token)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    training_arguments = DefaultTrainingArguments(
        output_dir=HF_HOME,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        load_best_model_at_end = False
    )
    train_loader, val_loader = get_dataloader(spider_train, spider_val_token, data_collator, model, tokenizer, training_arguments)
    """
    data_module = dict(
        train_dataset=spider_train,
        eval_dataset=spider_val_token,
        data_collator=data_collator,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module,
    )

    train_loader = trainer.get_train_dataloader()
    val_loader = trainer.get_eval_dataloader()
    """
    test_loader= val_loader
    analysis_loader = get_dataloader(spider_analysis_token, None, data_collator, model, tokenizer, training_arguments)

    C = None
    transform_to_one_hot = True
    analysis_test_loader = test_loader
    data_params = {"compute_acc": False, "compute_ex_score": evaluate_spider, "analysis_dataset": spider_analysis, "test_dataset": spider_val}
    return model, tokenizer, train_loader, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params
    


def load_spider_federated(model_name, batch_size, client_num):
    base_model = model_name #"deepseek-ai/deepseek-coder-1.3b-instruct"
    hf_token = None
    analysis_size = max(batch_size, 128)
    analysis_ind = range(analysis_size)

    spider_train = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=True)
    spider_analysis = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, rows=analysis_ind, tokenize=False)
    spider_analysis_token = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, rows=analysis_ind, tokenize=True)
    #spider_analysis = spider_train.select(torch.arange(analysis_size))
    #spider_train = spider_train.setup_dataset(model_name_or_path=base_model, num_rows=100)
    #spider_analysis = spider_analysis.setup_dataset(model_name_or_path=base_model)

    spider_val = SpiderUnifiedDataset(split="validation", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=False)
    spider_val_token = SpiderUnifiedDataset(split="validation", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, tokenize=True)
   
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_size="right", token=hf_token)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    training_arguments = DefaultTrainingArguments(
        output_dir=HF_HOME,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        load_best_model_at_end = False
    )
    train_loader, val_loader = get_dataloader(spider_train, spider_val_token, data_collator, model, tokenizer, training_arguments)
    """
    data_module = dict(
        train_dataset=spider_train,
        eval_dataset=spider_val_token,
        data_collator=data_collator,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module,
    )

    train_loader = trainer.get_train_dataloader()
    val_loader = trainer.get_eval_dataloader()
    """
    test_loader= val_loader
    analysis_loader = get_dataloader(spider_analysis_token, None, data_collator, model, tokenizer, training_arguments)

    client_loaders = []
    randperm = np.random.permutation(len(spider_train))
    for i in range(client_num):
        data_index = randperm[i:-1:client_num]
        spider_client = SpiderUnifiedDataset(split="train", dataset_folder=DATA_HOME, hf_token=hf_token).setup_dataset(model_name_or_path=base_model, rows=data_index, tokenize=True)
        client_loader = get_dataloader(spider_client, None, data_collator, model, tokenizer, training_arguments)
        client_loaders.append(client_loader)

    C = None
    transform_to_one_hot = True
    analysis_test_loader = test_loader
    data_params = {"compute_acc": False, "compute_ex_score": evaluate_spider, "analysis_dataset": spider_analysis, "test_dataset": spider_val}
    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params

if __name__ == "__main__":
    load_spider()

