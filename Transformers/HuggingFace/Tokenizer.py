# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:47:14 2023

@author: antona
From HuggingFace tutorial

Tokenizing means matching each word from the input sentence against a specified dictionary. 
Once the word is matched an ID is provided. 
This way we can continue working with the IDs instead of the real word.

pip install transformers
pip install transformers[sentencepiece]
"""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

#padding is used because the sentences are from different length
#it returns a pytorch type of tensor
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)