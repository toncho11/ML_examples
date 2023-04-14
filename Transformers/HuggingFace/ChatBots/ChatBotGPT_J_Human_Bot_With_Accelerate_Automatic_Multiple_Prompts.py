# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:55:58 2023

@author: antona

Link: https://huggingface.co/docs/accelerate/package_reference/big_modeling

This version is automatic and it does not require separate manual download of the weights.

This version of the code uses Accelerate library and can be loaded in both CPU and GPU RAM.

You might need to close all your programs and restart python to free up enough memory.

"""

import torch
torch.cuda.empty_cache()

import gc
gc.collect()

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer

checkpoint = "EleutherAI/gpt-j-6B"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()

print("Loading the sharded version of GPT-J ...")

# Download the Weights
checkpoint2 = "sgugger/sharded-gpt-j-6B"
weights_location = hf_hub_download(checkpoint2,"pytorch_model-00001-of-00002.bin")

print("Weights location of the sharded version:", weights_location)

# Load the checkpoint and dispatch it to the right devices
model = load_checkpoint_and_dispatch(
    model, 
    weights_location, 
    device_map="auto", 
    no_split_module_classes=["GPTJBlock"],
    offload_folder="offload"
)

def AnswerPrompt(prompt):
    
    print("Processing prompt ...")
    
    import gc
    gc.collect()
    
    start_pos = prompt.rfind("###")

    print("Tokenizing ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(0)

    print("Generating response ...")
    
    #parameters: https://huggingface.co/docs/transformers/main_classes/text_generation
    output = model.generate(inputs["input_ids"], temperature = 0.9, min_new_tokens = 3, max_new_tokens = 20  )
    answer = tokenizer.decode(output[0].tolist())
    
    pos_start = prompt.rfind("###") + 4
    #pos_end = min(answer.rfind("###")-1, answer.find(".", pos_start+11)+1, answer.find("[human]", pos_start+11)-1)
    pos_end = min(answer.rfind("###")-1, answer.find("[human]", pos_start+11)-1) #does not work well with prompt3
    short_answer = answer
    if (pos_end > pos_start):
        short_answer = answer[pos_start : pos_end]
        
    print(short_answer + "\n=================================")
    return short_answer, answer
    
prompt = """This is a discussion between a [human] and a [robot]. 
The [robot] is very nice and empathetic. The name of the [robot] is John. The [robot] is male.
The [humans]'s name is Peter. The age of the [robot] is 31.

[human]: Hello nice to meet you.
[robot]: Nice to meet you too.
###
[human]: How is it going today?
[robot]: Not so bad, thank you! How about you?
###
[human]: How old are you?
[robot]:"""

prompt2 = """This is a discussion between a [human] and a [robot]. 
The [robot] is very nice and empathetic. The name of the [robot] is John. The [robot] is male.
The [humans]'s name is Peter. The age of the [robot] is 31.

[human]: Hello nice to meet you.
[robot]: Nice to meet you too.
###
[human]: How is it going today?
[robot]: Not so bad, thank you! How about you?
###
[human]: What is your name?
[robot]:"""

prompt3 = """This is a discussion between a [human] and a [robot]. 
The [robot] is very nice and empathetic. The name of the [robot] is John. The [robot] is male.
The [humans]'s name is Peter. The age of the [robot] is 31.

[human]: Hello nice to meet you.
[robot]: Nice to meet you too.
###
[robot]:"""

short_answer1, answer1 = AnswerPrompt(prompt)
short_answer2, answer2 = AnswerPrompt(prompt2)
#short_answer3, answer3 = AnswerPrompt(prompt3)