# -*- coding: utf-8 -*-
"""

An example of RAG(Retrieval Augmented Generation)

RAG is way to focus the LLM on a domain and/or provide new data for the LLM without pre-training the LLM.

RAG is a kind of automatic prompt engineering to get some context for you question before 
submitting it to the LLM, which is done by creating an embedding of your question and searching
a vector database for sources that can be used to produce this (extra) context.

The implemenation of RAG is in the class ConversationalRetrievalChain which
uses 3 steps that also include the chat history. Please check the link below:
https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html 

Here we provide:
    - the documents that are used as a grounding context for the LLM
    - LLM that combined with augmented data will produce the final answer
    - how are the embeddings produced (ex. OpenAI)
    - a vector database for the similarity search
    - how is the conversation history managed

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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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

retriever = db.as_retriever() #wrapper around the db towards the retriever interface

#add context / memory for the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)

#Chain for having a conversation based on retrieved documents.
#This chain takes in chat history (a list of messages) and new questions, and then returns an answer to that question. 
#The algorithm for this chain consists of three parts.
chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)

print("Started conversation:")

#query = 'Tell me about "Amortisation of eligible liabilities instruments"'
#answer = chain.run({'question': query})

while True:
    print("QUERY: ", end="")
    query = input()
    answer = chain.run({'question': query})
    print("ANSWER: ", answer)