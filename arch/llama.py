
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

# Every target built here ends with <|eot_id|>, so that is the token a fine-tuned
# model learns to stop on. It is not tokenizer.eos_token: the base Llama-3.x
# checkpoints set eos_token to <|end_of_text|> (128001) while <|eot_id|> is
# 128009, and only the Instruct variants make the two agree.
RESPONSE_END_TOKEN = "<|eot_id|>"


def generation_stop_token_ids(tokenizer):
    """Token ids generate() should treat as end-of-response.

    Passing only tokenizer.eos_token_id lets decoding run past the <|eot_id|>
    the model actually emits, all the way to max_new_tokens, which surfaces as
    degenerate repetition after an otherwise correct answer.
    """
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    response_end_id = tokenizer.convert_tokens_to_ids(RESPONSE_END_TOKEN)
    if (response_end_id is not None
            and response_end_id != tokenizer.unk_token_id
            and response_end_id not in stop_ids):
        stop_ids.append(response_end_id)
    return stop_ids

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


def get_llama_model_and_formats(model_name, dtype, model_params):
    max_seq_length = 2048 
    #dtype = None # None for auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    model_id = model_name
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
    if dtype in ["default", "bf16"]:
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
        model_params["dtype"] = dtype
    elif dtype == "fp32":
        torch_dtype = torch.float32
        model_params["dtype"] = dtype
    else:
        raise NotImplementedError

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        #torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
        #attn_implementation="flash_attention_3"
    )

    return model, tokenizer, format_and_mask_instruction

