# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:25:26 2023

We are using the pre-trained model "bert-base-uncased". This model is trained with two objectives:
    - masked language modeling (MLM) objective
    - next sentence prediction (NSP) objective 
    
The goal of the NSP task is to model the relationship between pairs of sentences. It is a classification of a pair of sentences. It can be for example in 3 categories: "contradiction", "neutral", "entailment" 

The MRPC dataset is a dataset of pairs of sentences that says if two sentences are "equivalent".

The objective here is to fine-tune the "bert-base-uncased" model for classification of pair of sentences using 
the MRPC dataset in order to be able to say if two sentences are "equivalent".

Both the fine-tuning and evaluation are performed on the MRPC dataset (by separating them into Train and Validate datasets first)

Here the Trainer class is used to make the training process easier.

The processing of a pair of sentences is achieved by using 'token_type_ids' which allows us to select the first and the second sentence.
Note that if you select a different checkpoint, you won’t necessarily have the token_type_ids in your tokenized inputs.

It uses "dynamic padding" with a DataCollator which does the padding according to the batch instead of a fixed max number for all sentences.

pip install datasets evaluate transformers[sentencepiece]
pip install accelerate
"""

from accelerate import Accelerator
# from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
# from transformers import pipeline
import numpy as np

accelerator = Accelerator()

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #It pads all items in a batch so they have the same length.

from transformers import TrainingArguments

training_args = TrainingArguments("fune-tuned-trainer") #sets the name of the folder where the new model will be saved

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    evaluation_strategy="epoch",
)

#the new model (the weights) and its associated configurations will be saved in "fune-tuned-trainer"
trainer.train()

#Evaluate
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("glue", "mrpc")
print("Evaluation on Validation dataset: ",metric.compute(predictions=preds, references=predictions.label_ids))
#accuracy should be around 85%
