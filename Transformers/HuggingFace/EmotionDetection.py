# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:09:09 2023

@author: antona
"""

def get_emotion(text):

    from transformers import AutoTokenizer, AutoModelWithLMHead
    
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion") #bertweet-base-emotion-analysis

    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids,
               max_length=2)
  
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    
    return label

result = get_emotion("I am sad.")
print(result)