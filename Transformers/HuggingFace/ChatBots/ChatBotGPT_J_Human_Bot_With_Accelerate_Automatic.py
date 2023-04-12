# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:55:58 2023

@author: antona

Link: https://huggingface.co/docs/accelerate/package_reference/big_modeling

This version is automatic and it does not require separate manual download of the weights.

This version of the code uses Accelerate library and can be loaded in both CPU and GPU RAM.

You might need to close all your programs and restart python to free up enough memory.

"""

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

print("Processing prompt ...")

prompt = """This is a discussion between a [human] and a [robot]. 
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

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")
#inputs = inputs.to(0)
output = model.generate(inputs["input_ids"], temperature = 0.1)
print(tokenizer.decode(output[0].tolist()))