
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

prompt_format = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

                {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                {}<|eot_id|>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt_format.format(instruction, output)
        texts.append(text)
    return { "text" : texts, }

def get_llama_model_and_formats(model_name):
    max_seq_length = 2048 
    dtype = None # None for auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path = "your_dataset.json" # Your (instruction, response) file

    # 1. Load Tokenizer & Quantization Config
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right" 
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    return model, tokenizer, formatting_prompts_func

