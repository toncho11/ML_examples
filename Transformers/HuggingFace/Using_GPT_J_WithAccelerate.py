# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:15:57 2023

@author: antona

source: https://huggingface.co/docs/accelerate/usage_guides/big_modeling

This version of the code uses Accelerate library and can be loaded in both CPU and GPU RAM.

You clone the sharded version of this model with
    
git clone https://huggingface.co/sgugger/sharded-gpt-j-6B
cd sharded-gpt-j-6B
git-lfs install
git lfs pull

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

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("Hello, my name is", return_tensors="pt")
#inputs = inputs.to(0) #helps with memory
output = model.generate(inputs["input_ids"])
tokenizer.decode(output[0].tolist())