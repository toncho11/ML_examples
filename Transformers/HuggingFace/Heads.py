# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:58:35 2023

@author: antona
From HuggingFace tutorial

This code shows what a "head" is. Heads are used to transform the output
of a model for a specific task (e.g. sentiment-analysis).


pip install transformers
pip install transformers[sentencepiece]
"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel

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

#here we start using the result of the tokenizer as an input for model ()

#1 It is a model without a specfied "head".
#This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll 
#call hidden states, also known as features. For each model input, we’ll retrieve a high-dimensional 
#vector representing the contextual understanding of that input by the Transformer model.

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

#2 Here we instantiate a model with a specified "head"
#AutoModelForSequenceClassification has a prespecified head for sequence classification, which
#is what we need for sentiment analysis later
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) 
outputs = model(**inputs)
print(outputs.logits.shape) #in this case logits are available

#3 Perform the sentiment analysis based on 2 
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) #use the logits to produce binary brobability
#the output tensor for each sentence is [negative prob, positive prob]
print("Positive score first sentence:", predictions[0][1], "Positive score second sentence:", predictions[1][1])
#print(predictions)