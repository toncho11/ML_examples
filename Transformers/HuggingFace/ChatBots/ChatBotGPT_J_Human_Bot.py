# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:26:57 2023

@author: antona

source: https://nlpcloud.com/effectively-using-gpt-j-gpt-neo-gpt-3-alternatives-few-shot-learning.html
online demo: https://huggingface.co/nlpcloud/instruct-gpt-j-fp16

Requires 16 GB GPU. You can try on Google Colab for free.

"""

from transformers import pipeline
import torch

device = -1
force_gpu = False
if force_gpu:
    device = 0

generator = pipeline(model="nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16, device = device)

prompt = """This is a discussion between a [human] and a [robot]. 
The [robot] is very nice and empathetic.

[human]: Hello nice to meet you.
[robot]: Nice to meet you too.
###
[human]: How is it going today?
[robot]: Not so bad, thank you! How about you?
###
[human]: I am ok, but I am a bit sad...
[robot]: Oh? Why that?
###
[human]: I broke up with my girlfriend...
[robot]:"""

print(generator(prompt))