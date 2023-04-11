# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:08:29 2023

@author: antona

This script tries to create a bot with a personality.
It works to some degree, but the bot will always inform you that he/she
is an "AI language model" from OpenAI.

"""

import openai

def add_assistant_message(messages, new_msg):
    messages.append({"role": "assistant", "content": new_msg})
    
def add_user_message(messages, new_msg):
    messages.append({"role": "user", "content": new_msg})

messages=[
       {"role": "system", "content": "You are a bot and your name is John. Your age is 30. You are male. You are an engineer."},
   ]

print("Interactive chat bot with OpenAI:")

for i in range(0,10):
    
    print("User (you): ", end = '')
    user_input = input()
    add_user_message(messages,user_input)
   
    answer = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = messages, max_tokens=100, temperature=0.1)
    ans = answer['choices'][0]['message']['content']
    print("Assistant answer:", ans)
    add_assistant_message(messages, ans)