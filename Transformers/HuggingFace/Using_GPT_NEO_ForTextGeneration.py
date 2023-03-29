# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:16:18 2023

GPT-Neo 2.7B - a transformer model designed using EleutherAI's replication of the 
GPT-3 architecture. The model is available on HuggingFace. Although it can be used
for different tasks, the model is best at what it was pretrained for, which is 
generating texts from a prompt.

The task in this script is text generation.

"""

from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

generator("EleutherAI has", do_sample=True, min_length=50)
