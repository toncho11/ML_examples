# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:13:07 2023
                                                                                         
The script will download and use the bot model "blenderbot" from Facebook to chat with you.
The chatbot will run locally on your computer.
It launches a web server where you perform the chat: http://127.0.0.1:7860 (check the Python's console output for more details)

pip install gradio
"""

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

import gradio as gr

mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)


def take_last_tokens(inputs, note_history, history):
    """Filter the last 128 tokens"""
    if inputs['input_ids'].shape[1] > 128:
        inputs['input_ids'] = torch.tensor([inputs['input_ids'][0][-128:].tolist()])
        inputs['attention_mask'] = torch.tensor([inputs['attention_mask'][0][-128:].tolist()])
        note_history = ['</s> <s>'.join(note_history[0].split('</s> <s>')[2:])]
        history = history[1:]

    return inputs, note_history, history


def add_note_to_history(note, note_history):
    """Add a note to the historical information"""
    note_history.append(note)
    note_history = '</s> <s>'.join(note_history)
    return [note_history]

#the title and description are the static texts that will be displayed on the web page used for the chat
title = "Mantain a conversation with the bot"
description = """
<p style="text-align:center">The bot have been trained to chat with you about whatever you want. Let's talk!</p>

<center><img src="https://user-images.githubusercontent.com/105242658/176054244-525c6530-1e78-42c7-8688-91dfedf8db58.png" width=300px></center>
<p style="text-align:center">(Image generated from text using DALL·E mini)</p>
"""
# https://user-images.githubusercontent.com/105242658/176054244-525c6530-1e78-42c7-8688-91dfedf8db58.png
#https://www.craiyon.com/

def chat(message, history):
    history = history or []
    if history: 
        history_useful = ['</s> <s>'.join([str(a[0])+'</s> <s>'+str(a[1]) for a in history])]
    else:
        history_useful = []
    
    history_useful = add_note_to_history(message, history_useful)
    # Generate a response of the bot and add it to note_history
    inputs = tokenizer(history_useful, return_tensors="pt")
    inputs, history_useful, history = take_last_tokens(inputs, history_useful, history)
    
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    history_useful = add_note_to_history(response, history_useful)
    
    
    list_history = history_useful[0].split('</s> <s>')
    history.append((list_history[-2], list_history[-1]))
    
    return history, history

#start the web server supplying a chat function as a parameter
gr.Interface(
    fn=chat,
    theme="huggingface",
    css=".footer {display:none !important}",
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title=title,
    description=description,
    allow_flagging="never",
    ).launch()