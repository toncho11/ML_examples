# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:37:18 2023

Source: https://nlpcloud.com/instruct-version-of-gpt-j-using-stanford-alpaca-dataset.html

This script fine-tunes the GPT-J model to make it an instruct model.
GPT-J is a GPT-2-like causal language model trained on the Pile dataset.
GPT-J description: https://huggingface.co/EleutherAI/gpt-j-6B
GPT-J is best at what it was pretrained for, which is generating text from a prompt.

This GPT-J instuct model (instruct-gpt-j-fp16) has been trained on the same dataset as 
Alpaca Stanford model (which is also an instruct model).

Warning - the model file is big. It requires CUDA enabled PyTorch.
This model is an fp16 version of our fine-tuned model, which works very well on
a GPU with 16GB of VRAM like an NVIDIA Tesla T4.

"""

from transformers import pipeline
import torch

generator = pipeline(model="nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16, device=0)

prompt = "Correct spelling and grammar from the following text.\nI do not wan to go\n"

print(generator(prompt))