# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:37:11 2023

@author: antona

source: https://platform.openai.com/docs/guides/chat

This script shows how to make an interactive bot with OpenAI.
All previous communication must be supplied at every request of openai.ChatCompletion.create.
3 roles can be used: "system", "user" and "assistant".

Note: you need to be using OpenAI Python v0.27.0 for the code below to work

"""

import openai

def add_assistant_message(messages, new_msg):
    messages.append({"role": "assistant", "content": new_msg})
    
def add_user_message(messages, new_msg):
    messages.append({"role": "user", "content": new_msg})

messages=[
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Who won the world series in 2020?"},
       {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
       {"role": "user", "content": "Where was it played?"}
   ]

print("Interactive chat bot with OpenAI:")

for i in range(0,10):
    
    answer = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = messages)
   
    ans = answer['choices'][0]['message']['content']
    print("Assistant answer:", ans)
    add_assistant_message(messages, ans)
    
    print("User (you): ", end = '')
    user_input = input()
    add_user_message(messages,user_input)
    
    
    
    
    