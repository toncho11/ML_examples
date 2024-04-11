# -*- coding: utf-8 -*-
"""

Check if LangChain is installed and that it can find a valid API key file for OpenAI

LangChain requires Python >= 3.9

"""

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
keyFile = open(os.path.join(dir_path,'OpenAiApiKey.txt'), 'r')
openai_api_key = keyFile.readline()

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = OpenAI(openai_api_key = openai_api_key)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

answer = llm_chain.run(question)

print(answer)