# -*- coding: utf-8 -*-
"""

An example of RAG(Retrieval Augmented Generation)
It loads PDF docs and provides answers to your questions.
It uses OpenAI as LLM and Chroma as vector database.

pip install pypdf
pip install chromadb

@author: antona
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
#from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
file = open(os.path.join(dir_path,'OpenAiApiKey.txt'), 'r')
openai_api_key = file.readline()

def load_chunk_pdf():
    pdf_folder_path = "C:\\Work\\PythonCode\\ML_examples\\Transformers\\LangChain\\docs"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        add_start_index = True,)
    
    #chunked_documents = text_splitter.split_documents(documents)
    texts = text_splitter.split_documents(documents)
    return texts
    
texts = load_chunk_pdf()

db = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key = openai_api_key))

llm = ChatOpenAI(openai_api_key = openai_api_key, model_name='gpt-3.5-turbo', temperature=0)

retriever = db.as_retriever()

#add context / memory for the chat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)

print("Started conversation:")

#query = 'Tell me about "Amortisation of eligible liabilities instruments"'
#answer = chain.run({'question': query})

while True:
    print("QUERY: ", end="")
    query = input()
    answer = chain.run({'question': query})
    print("ANSWER: ", answer)