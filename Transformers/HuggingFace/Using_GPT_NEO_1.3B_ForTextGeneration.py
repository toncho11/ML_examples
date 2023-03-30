# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:51:06 2023

Source: https://huggingface.co/EleutherAI/gpt-neo-1.3B

GPT-NEO with 1.3B parameters - a transformer model designed using EleutherAI's replication of the 
GPT-3 architecture. The model is available on HuggingFace. Although it can be used
for different tasks, the model is best at what it was pretrained for, which is 
generating texts from a prompt.

The task in this script is text generation.

There are also a 2.7B and 6B versions.

"""

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print("=========================================================")
print(gen_text)