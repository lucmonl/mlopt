"""
Adapted from https://github.com/microsoft/LoRA/tree/main/examples/NLU/examples/multiple-choice
"""

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from datasets import load_dataset
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

logger = logging.getLogger(__name__)

pad_to_max_length = False

HF_HOME = os.environ["HF_HOME"]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        """
        import sys
        for feature in features:
            for k,v in feature.items():
                print(k, len(v), v)
        sys.exit()

        for feature in features:
            for i in range(num_choices):
                print(i)
                for k,v in feature.items():
                    print(len(v))
                    print(k, v[i])
                    pass
        """
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    

def load_swag(model_name, batch_size):
    datasets = load_dataset("swag", "regular")

    config = AutoConfig.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        #cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None,
    )

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    max_seq_length = 80
    # Preprocessing the datasets.
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding=False,# if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    train_dataset = datasets["train"]
    #if data_args.max_train_samples is not None:
    #    train_dataset = train_dataset.select(range(data_args.max_train_samples))
    
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )
    
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )

    #store the current random state
    st0 = np.random.get_state()
    #use a fixed seed to ensure same split on the dataset
    np.random.seed(42) 

    eval_len = len(eval_dataset)
    eval_perm = np.random.permutation(eval_len)
    val_index = eval_perm[:eval_len//2]
    test_index = eval_perm[eval_len//2:]
    val_dataset = eval_dataset.select(val_index)
    test_dataset = eval_dataset.select(test_index)


    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=None)

    """
    val_loader = torch.utils.data.DataLoader(
        val_dataset,  # type: ignore
        shuffle=False,
        collate_fn=data_collator, # Default data collator
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,  # type: ignore
        shuffle=False,
        collate_fn=data_collator, # Default data collator
        batch_size=batch_size,
    )
    """
    training_args = TrainingArguments(output_dir=HF_HOME, per_device_train_batch_size=batch_size)
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    train_loader = trainer.get_train_dataloader()

    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    val_loader = trainer.get_eval_dataloader()
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    test_loader = trainer.get_eval_dataloader()

    #reload the initial random state
    np.random.set_state(st0)
    analysis_size = max(batch_size, 128)
    
    #analysis_dataset = torch.utils.data.Subset(train_loader.dataset, torch.arange(analysis_size))
    analysis_dataset = train_dataset.select(torch.arange(analysis_size))
    analysis_dataset = analysis_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )
    """
    analysis_loader = torch.utils.data.DataLoader(
            analysis_dataset,  # type: ignore
            shuffle=True,
            collate_fn=data_collator, # Default data collator
            batch_size=batch_size,
    )
    """
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    analysis_loader = trainer.get_eval_dataloader()

    C = None
    transform_to_one_hot = True
    analysis_test_loader = test_loader
    data_params = {"compute_acc": True}
    return model, tokenizer, train_loader, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params


def load_swag_federated(model_name, batch_size, client_num):
    datasets = load_dataset("swag", "regular")

    config = AutoConfig.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        #cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None,
    )

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    max_seq_length = 80
    """
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    """
    # Preprocessing the datasets.
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding=False,# if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    train_dataset = datasets["train"]
    #if data_args.max_train_samples is not None:
    #    train_dataset = train_dataset.select(range(data_args.max_train_samples))
    """
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )
    """
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )

    #store the current random state
    st0 = np.random.get_state()
    #use a fixed seed to ensure same split on the dataset
    np.random.seed(42) 

    eval_len = len(eval_dataset)
    eval_perm = np.random.permutation(eval_len)
    val_index = eval_perm[:eval_len//2]
    test_index = eval_perm[eval_len//2:]
    val_dataset = eval_dataset.select(val_index)
    test_dataset = eval_dataset.select(test_index)

    """
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )
    """
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=None)
    """
    training_args = TrainingArguments(per_device_train_batch_size=batch_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    """
    train_loader = torch.utils.data.DataLoader(
            train_dataset,  # type: ignore
            shuffle=True,
            collate_fn=data_collator, # Default data collator
            batch_size=batch_size,
        )
    """
    val_loader = torch.utils.data.DataLoader(
        val_dataset,  # type: ignore
        shuffle=False,
        collate_fn=data_collator, # Default data collator
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,  # type: ignore
        shuffle=False,
        collate_fn=data_collator, # Default data collator
        batch_size=batch_size,
    )
    """
    training_args = TrainingArguments(output_dir=HF_HOME, per_device_train_batch_size=batch_size)
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    val_loader = trainer.get_eval_dataloader()
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    test_loader = trainer.get_eval_dataloader()

    client_loaders = []
    randperm = np.random.permutation(len(train_dataset))
    for i in range(client_num):
        data_index = randperm[i:-1:client_num]
        #client_train = torch.utils.data.Subset(train_dataset, data_index)
        client_dataset = train_dataset.select(data_index)
        client_train = client_dataset.map(
            preprocess_function,
            batched=True,
            #num_proc=data_args.preprocessing_num_workers,
            #load_from_cache_file=not data_args.overwrite_cache,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=client_train,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        client_loader = trainer.get_train_dataloader()
        """
        client_loader = torch.utils.data.DataLoader(
            client_train,  # type: ignore
            shuffle=True,
            collate_fn=data_collator, # Default data collator
            batch_size=batch_size,
        )
        """
        client_loaders.append(client_loader)
    #reload the initial random state
    np.random.set_state(st0)
    analysis_size = max(batch_size, 128)
    
    #analysis_dataset = torch.utils.data.Subset(train_loader.dataset, torch.arange(analysis_size))
    analysis_dataset = train_dataset.select(torch.arange(analysis_size))
    analysis_dataset = analysis_dataset.map(
        preprocess_function,
        batched=True,
        #num_proc=data_args.preprocessing_num_workers,
        #load_from_cache_file=not data_args.overwrite_cache,
    )
    """
    analysis_loader = torch.utils.data.DataLoader(
            analysis_dataset,  # type: ignore
            shuffle=True,
            collate_fn=data_collator, # Default data collator
            batch_size=batch_size,
    )
    """
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=None,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    analysis_loader = trainer.get_eval_dataloader()

    C = None
    transform_to_one_hot = True
    analysis_test_loader = test_loader
    data_params = {"compute_acc": True}
    return model, tokenizer, train_loader, client_loaders, val_loader, test_loader, analysis_loader, analysis_test_loader, C, transform_to_one_hot, data_params
