# Following test script in https://github.com/meta-math/MetaMath
import argparse
import json
import re
import jsonlines
from fractions import Fraction
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import sys
import os
MAX_INT = sys.maxsize

abs_path = "/home/lucmon/lucmon/mlopt/"
cache_dir = os.environ["HF_HOME"] + "/vllm_cache"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model_name, model_path, tokenizer, device, data_path, is_val, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    verbose = not is_val #if valiadation, do not output logs
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    if verbose:
        print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)
    #end=20
    import numpy as np
    dataset_size = len(gsm8k_ins)
    #store the current random state
    st0 = np.random.get_state()
    #use a fixed seed to ensure same split on the dataset
    np.random.seed(42)
    val_size = int(0*dataset_size)
    randperm = np.random.permutation(dataset_size)
    if is_val:
        val_ind = randperm[:val_size]
    else:
        val_ind = randperm[val_size:]
    #reload the initial random state
    np.random.set_state(st0)
    gsm8k_ins = [gsm8k_ins[ind] for ind in val_ind]
    gsm8k_answers = [gsm8k_answers[ind] for ind in val_ind]
    #gsm8k_ins = gsm8k_ins[start:end]
    #gsm8k_answers = gsm8k_answers[start:end]
    if verbose:
        print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    if verbose:
        print('sampleing =====', sampling_params)
    from pathlib import Path
    lora_request = LoRARequest("gsm8k_adapter", 1, abs_path+model_path)
    model_name = "mistralai/Mistral-7B-v0.1"
    llm = LLM(model=model_name,tensor_parallel_size=tensor_parallel_size, enable_lora=True, download_dir= cache_dir)
    
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params, lora_request=lora_request, use_tqdm=verbose)
        #model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        #completions = llm.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        #tokenizer.batch_decode(generated_ids)[0]

        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)
            #res_completions.append(tokenizer.batch_decode(output)[0])

    valid_outputs = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            if verbose:
                print(float(y_pred), float(prompt_answer))
            result.append(float(y_pred) == float(prompt_answer))
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            valid_outputs.append(temp)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('len invalid outputs ====', len(invalid_outputs))
    #print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    """
    print(valid_outputs[0]['question'])
    print(valid_outputs[0]['output'])
    print(valid_outputs[0]['answer'])

    print(invalid_outputs[0]['question'])
    print(invalid_outputs[0]['output'])
    print(invalid_outputs[0]['answer'])
    """
    return acc