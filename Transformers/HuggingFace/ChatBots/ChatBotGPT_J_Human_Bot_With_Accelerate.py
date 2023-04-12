# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:42:20 2023

@author: antona

source: https://huggingface.co/docs/accelerate/usage_guides/big_modeling

This version of the code uses Accelerate library and can be loaded in both CPU and GPU RAM.

You need to clone the sharded version of this model for this script to run:
    
git clone https://huggingface.co/sgugger/sharded-gpt-j-6B
cd sharded-gpt-j-6B
git-lfs install
git lfs pull

Here is more information on downloading a model: https://huggingface.co/docs/hub/models-downloading

The sharded version sharded-gpt-j-6B folder must be in the same folder as this script.

Then the model will be loaded from disk using the Accelerate library that
can use both CPU RAM and GPU RAM in a process called offloading.

You might need to close all your programs and restart python to free up enough memory.

"""

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

checkpoint = "EleutherAI/gpt-j-6B"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()

from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, "sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"],
    offload_folder="offload"
)

from transformers import AutoTokenizer

prompt = """This is a discussion between a [human] and a [robot]. 
The [robot] is very nice and empathetic. The name of the [robot] is John. The [robot] is male.
The [humans]'s name is Peter. The age of the [robot] is 31.

[human]: Hello nice to meet you.
[robot]: Nice to meet you too.
###
[human]: How is it going today?
[robot]: Not so bad, thank you! How about you?
###
[human]: What is your age?
[robot]:"""

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")
#inputs = inputs.to(0) #helps with memory
output = model.generate(inputs["input_ids"])
print(tokenizer.decode(output[0].tolist()))