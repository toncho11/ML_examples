# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:41:34 2023

@author: antona
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("mrsinghania/asr-question-detection")

model = AutoModelForSequenceClassification.from_pretrained("mrsinghania/asr-question-detection")

pipe = pipeline("text-classification", model = model, tokenizer = tokenizer)

#label 0 is statement
#label 1 is question

print(pipe("Who are you?"))
print(pipe("Who are you"))
print(pipe("I am from Boston"))

result = pipe("Who are you?")
if (result[0]["label"] == 'LABEL_1' ):
    print("This is a question")
elif result[0]["label"] == 'LABEL_0':
    print("This is a statement")
    

