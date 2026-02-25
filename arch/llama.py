
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import os
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

def format_and_mask_instruction(example, tokenizer, max_length=2048):
    # 1. Define the components
    user_part = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_part = f"{example['response']}<|eot_id|>"
    
    # 2. Tokenize separately to find lengths
    user_ids = tokenizer.encode(user_part, add_special_tokens=False)
    assistant_ids = tokenizer.encode(assistant_part, add_special_tokens=False)
    
    # 3. Combine them
    input_ids = user_ids + assistant_ids
    
    # 4. Create labels: -100 for the user part, actual IDs for the assistant part
    # This ensures the model only "learns" from the assistant's response.
    labels = ([-100] * len(user_ids)) + assistant_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    # 5. Create the attention mask (all 1s because this is pre-padding)
    attention_mask = [1] * len(input_ids)
    
    # Ensure all values are flat lists of integers (not nested)
    # This prevents the "excessive nesting" error
    
    input_ids = list(map(int, input_ids))
    labels = list(map(int, labels))
    attention_mask = list(map(int, attention_mask))
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def get_llama_model_and_formats(model_name):
    max_seq_length = 2048 
    dtype = None # None for auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
        #torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
        #attn_implementation="flash_attention_3"
    )

    return model, tokenizer, format_and_mask_instruction

