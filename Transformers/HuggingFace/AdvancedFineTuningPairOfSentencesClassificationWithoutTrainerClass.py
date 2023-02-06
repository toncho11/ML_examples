# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:46:52 2023

We are using the pre-trained model "bert-base-uncased". This model is trained with two objectives:
    - masked language modeling (MLM) objective
    - next sentence prediction (NSP) objective 
    
The goal of the NSP task is to model the relationship between pairs of sentences. It is a classification of a pair of sentences. It can be for example in 3 categories: "contradiction", "neutral", "entailment" 

The MRPC dataset is a dataset of pairs of sentences that says if two sentences are "equivalent".

The objective here is to fine-tune the "bert-base-uncased" model for classification of pair of sentences using 
the MRPC dataset in order to be able to say if two sentences are "equivalent".

Both the fine-tuning and evaluation are performed on the MRPC dataset (by separating them into Train and Validate datasets first)

Here we do not use the Trainer class and so more code is required to perform the training task.

The processing of a pair of sentences is achieved by using 'token_type_ids' which allows us to select the first and the second sentence.
Note that if you select a different checkpoint, you won’t necessarily have the token_type_ids in your tokenized inputs.

It uses "dynamic padding" with a DataCollator which does the padding according to the batch instead of a fixed max number for all sentences.

pip install datasets evaluate transformers[sentencepiece]
pip install accelerate
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #it pads all items in a batch so they have the same length.

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

#define data loaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

#get our pre-trained model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

#=====================================================================

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

#=====================================================================
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
#lr - learning rate scheduler - it will progressively set the lr to 0
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

#=====================================================================
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#Our own train loop===================================================

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train() #set the model into training mode (Dropout and BatchNorm are designed to behave differently during training and evaluation)
for epoch in range(num_epochs):
    for batch in train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()} #put on GPU (CPU)
        
        outputs = model(**batch)
        
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step() #always after the optimizer step
        optimizer.zero_grad()
        
        progress_bar.update(1)

print("Training Done")

#Evaluate model=======================================================

import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval() #put the model into evaluation mode

for batch in eval_dataloader:
    
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())