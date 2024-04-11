'''
Documentation: https://python.langchain.com/docs/modules/memory/agent_with_memory/

This is an example that shows how to use:
 - the concept of "tools" supported by OpenAI. Here a tool based on Google search API is used.
 - automatic handling of the chat history (or having a memory, preserving the context)
 
pip install langchain
pip install langchain-openai
pip install google-api-python-client>=2.100.0

2 API keys and one ID:
You need to enable Google "Custom Search API" here: https://console.cloud.google.com/apis/dashboard
You need to also enable API Keys credentials and get your API key: https://console.cloud.google.com/apis/credentials
You need to add a search engine and get its CSE id: https://programmablesearchengine.google.com/controlpanel/all

'''

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_openai import OpenAI

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
file = open(os.path.join(dir_path,'OpenAiApiKey.txt'), 'r')
openai_api_key = file.readline()
file = open(os.path.join(dir_path,'GoogleApiKey.txt'), 'r')
google_api_key = file.readline()
file = open(os.path.join(dir_path,'GoogleCseId.txt'), 'r')
google_cse_id = file.readline()

search = GoogleSearchAPIWrapper(google_api_key = google_api_key, google_cse_id = google_cse_id)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
] #only 1 tool added, but there can be many

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(openai_api_key= openai_api_key, temperature=0), prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

answer = agent_chain.run(input="How many people live in canada?")
print(answer)

answer = agent_chain.run(input="what is their national anthem called?")
print(answer)