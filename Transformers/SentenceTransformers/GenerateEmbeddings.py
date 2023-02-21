# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:59:01 2023

https://www.sbert.net/#
https://arxiv.org/abs/1908.10084

This is a simple example on how to generate an embedding for each sentence.
Each embedding is of length 384.

pip install -U sentence-transformers

"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")