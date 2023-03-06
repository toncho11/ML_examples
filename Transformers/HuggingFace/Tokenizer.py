# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:47:14 2023

@author: antona
From HuggingFace tutorial

Tokenization is a way of separating a piece of text into smaller units called tokens. 
Tokens can be either words, characters, or subwords.
Also tokenz are assign IDs.
This way we can continue working with the IDs instead of the real tokens (words for example).

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
#all sentences are padded with 'tokenizer.pad_token_id' in order to become the same length
#it returns a pytorch type of tensor

#Usage 1 
#contains both the ids in inputs['input_ids'] and the attention mask in inputs['attention_mask]
#ids are with added 101 ans 102
#altomatic padding is performed
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

#Usage 2 
tokens = tokenizer.tokenize(raw_inputs) #all the words in the sentence
ids = tokenizer.convert_tokens_to_ids(tokens) #the ids corresponding to each word
#almost the same as inputs['input_ids'] except that here 101 (begin) and 102 (end) ids are missing

print(tokenizer.decode(inputs['input_ids'][0])) #when decoding the "inputs" we get the 101 and 102 converted to [CLS] and [SEP]
print(tokenizer.decode(ids)) #here there are no 101 and 102 ids, so they are not converted to [CLS] and [SEP], here it contains both sentences

