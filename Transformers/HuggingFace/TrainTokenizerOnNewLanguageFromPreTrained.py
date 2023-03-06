# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:53:16 2023

From Hugging Face tutorial

Tokenization is a way of separating a piece of text into smaller units called tokens. 
Tokens can be either words, characters, or subwords.

This example trains a new tokenzier on a specific language.
A tokenizer can be automatically trained (no need for manual annotation) to a specific language.
Here a new tokenizer is trained on Python code using the CodeSearchNet dataset.
It is not a training from scratch, we use a pre-trained tokenizer from Generative Pre-trained Transformer 2 (GPT-2) with English
and we optimize it for the Python language.
We compare results from the old one and the newly specialized one.

pip install datasets evaluate transformers[sentencepiece]

"""

from datasets import load_dataset

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")

#explore data: 11 columns and 412178 rows
#raw_datasets["train"]

#we will use the column "whole_func_string"
#print(raw_datasets["train"][123456]["whole_func_string"])

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
        
training_corpus = get_training_corpus()
        
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
print("Old tokenizer:")
print(tokens)

#train a new tokenizer from pre-trained
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
print("New tokenizer:")
print(tokens)

#different number of tokens/words detected
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))

#tokenizer.save_pretrained("code-search-net-tokenizer")