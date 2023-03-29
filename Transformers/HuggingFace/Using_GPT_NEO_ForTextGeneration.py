# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:01:44 2023

Source: https://huggingface.co/EleutherAI/gpt-neo-2.7B

GPT-Neo 2.7B - a transformer model designed using EleutherAI's replication of the 
GPT-3 architecture. The model is available on HuggingFace. Although it can be used
for different tasks, the model is best at what it was pretrained for, which is 
generating texts from a prompt.

The task in this script is text generation.

There is also a 1.3B and 6B versions.

"""

import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])

input_ids = inputs["input_ids"]

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)