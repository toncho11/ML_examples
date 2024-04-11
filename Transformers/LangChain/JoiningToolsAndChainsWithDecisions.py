# -*- coding: utf-8 -*-
"""

Title: Joining Tools and Chains with Decisions.

It uses the type of agent "zero-shot-react-description" to
automatically chain several tools until the answer is found.

pip -q install langchain huggingface_hub openai google-search-results tiktoken wikipedia

Youtube: https://www.youtube.com/watch?v=ziu87EXZVUE

Original file is located at https://colab.research.google.com/drive/1QpvUHQzpHPvlMgBElwd5NJQ6qpryVeWE

SerpApi - An API call away to scrape Google search result. In another word, it is the JSON representative of Google search result.
"""

"""### Creating an Agent"""

from langchain.agents import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
file = open(os.path.join(dir_path,'OpenAiApiKey.txt'), 'r')
openai_api_key = file.readline()
file = open(os.path.join(dir_path,'GoogleApiKey.txt'), 'r')
google_api_key = file.readline()
file = open(os.path.join(dir_path,'GoogleCseId.txt'), 'r')
google_cse_id = file.readline()

search = GoogleSearchAPIWrapper(google_api_key = google_api_key, google_cse_id = google_cse_id)

llm=OpenAI(openai_api_key= openai_api_key, temperature=0)

tools = load_tools(["llm-math","wikipedia","terminal"], llm=llm, allow_dangerous_tools=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),

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

answers = []

answer = agent_chain.run(input="How many people live in canada?")
print(answer)
answers.append(answer)

answer = agent_chain.run(input="what is their national anthem called?")
print(answer)
answers.append(answer)

answer = agent_chain.run("Hi How are you today?")
print(answer)
answers.append(answer)

answer = agent_chain.run("Who is the United States President? What is his current age raised divided by 2?")
print(answer)
answers.append(answer)

answer = agent_chain.run("What is the average age in the United States? How much less is that then the age of the current US President?")
print(answer)
answers.append(answer)

answer = agent_chain.run("Who is the head of DeepMind")
print(answer)
answers.append(answer)

answer = agent_chain.run("What is DeepMind")
print(answer)
answers.append(answer)

answer = agent_chain.run("Take the year DeepMind was founded and add 50 years. Will this year have AGI?")
print(answer)
answers.append(answer)

answer = agent_chain.run("Where is DeepMind's office?")
print(answer)
answers.append(answer)

answer = agent_chain.run("If I square the number for the street address of DeepMind what answer do I get?")
print(answer)
answers.append(answer)

#Windows version
#It is unable to access the current folder on Windows
answer = agent_chain.run("What files are in my current directory on Windows?")
print(answer)
answers.append(answer)