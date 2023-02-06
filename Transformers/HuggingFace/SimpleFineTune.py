# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:08 2023


Fine-tuning trains a pretrained model on a new dataset without training from scratch. 
This process, also known as "transfer learning".
We access the pre-trained model using the checkpoint "bert-base-uncased.
Tthen we train (fine-tune) using a only two samples.
"""

import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1]) #sets 2 lables with the value 1 (two positive evaluations corresponding to the two sentences above)

optimizer = AdamW(model.parameters())

#a step in training a model in PyTorch: loss, backaward(), step()
loss = model(**batch).loss
loss.backward()
optimizer.step()