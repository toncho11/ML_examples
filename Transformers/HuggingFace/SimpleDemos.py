# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:15:28 2023

@author: antona
From HuggingFace tutorial

Demonstrates how easy is to use HuggingFace models for various tasks.
This is from high level perspective.

pip install transformers
pip install transformers[sentencepiece]

"""

from transformers import pipeline

#Sentiment analysis demo
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print("\n#######Sentiment analysis============================================")
print(result)
print("=======================================================================")

#Zero-shot classification
#It allows you to specify which labels to use for the classification, so you don’t have to rely on the labels of the pretrained model. 
#We can select the category (the label) of a sentence choosing from a predefined list. 
#This pipeline is called zero-shot because you don’t need to fine-tune the model on your data to use it. 
#It can directly return probability scores for any list of labels you want!
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print("\n#######Zero-shot classification======================================")
print(result)
print("=======================================================================")

#Text generation
#short - using default model
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print("\n#######Text generation SHORT=========================================")
print(result)
print("=======================================================================")
#long using a particular model
generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2, #how many responses
)
print("\n#######Text generation LONG==========================================")
print(result)
print("=======================================================================")

#Mask filling
#The next pipeline is fill-mask. The idea of this task is to fill in the blanks (<mask>) in a given text.
#The special <mask> word, which is often referred to as a mask token. Not all models use the <mask> word.
#For the bert-base-cased model it is [MASK].
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print("\n#######Mask filling==================================================")
print(result)
print("=======================================================================")

#Named entity recognition
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print("\n#######Named entity recognition======================================")
print(result)
print("=======================================================================")

#Question answering 1

#Using the default pipeline and checkpoint
#Note that this pipeline works by extracting information from the provided context; it does not generate the answer.
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print("\n#######Question Answering 1==========================================")
print(result)
print("=======================================================================")


#Question answering 2

# Using a specific checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back Transformers?"
result = question_answerer(question=question, context=context)
print("\n#######Question Answering 2==========================================")
print(result)
print("=======================================================================")

#Summarization
summarizer = pipeline("summarization")
result = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print("\n#######Text summarization============================================")
print(result)
print("=======================================================================")

#Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print("\n#######Transalation form French to English===========================")
print(result)
print("=======================================================================")