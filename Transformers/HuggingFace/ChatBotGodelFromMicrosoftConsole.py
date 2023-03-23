# -*- coding: utf-8 -*-
'''
GODEL: https://www.microsoft.com/en-us/research/project/godel/
Paper: https://www.microsoft.com/en-us/research/uploads/prod/2022/05/2206.11309.pdf
Git page: https://github.com/microsoft/GODEL

This a chat bot that:
    - can take instructions on how to respond 
    - can take on the fly supplied knowledge before responding

The script will download and use the bot model "GODEL" from Microsoft to chat with you.
The chatbot will run locally on your computer.

This is the console version (as opposed to the web version)

The syntax of the queries is:
1) A sentence starting with "Instruction:" explaining how the LLM should answer
2) Followed by "[CONTEXT]" + the question from the user
3) Followed by "[KNOWLEDGE]" + the information you want to provide (this argument is optional)
 
It is unclear where the dialog from the previous interactions is added. It should be in the Knowledge part? 
'''

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

top_p      = 0.9
min_length = 8
max_length = 64

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

preset_examples = [
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'Does money buy happiness?', 'Chitchat'),
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'What is the goal of life?', 'Chitchat'),
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'What is the most interesing thing about our universe?', 'Chitchat'),
     ('Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.', 
     '''Scooby-Doo is the eponymous character and protagonist of the animated television franchise of the same name, created in 1969 by the American animation company Hanna-Barbera.[1] He is a male Great Dane and lifelong companion of amateur detective Shaggy Rogers, with whom he shares many personality traits. He features a mix of both canine and human behaviors (reminiscent of other talking animals in Hanna-Barbera's series), and is treated by his friends more or less as an equal. Scooby often speaks in a rhotacized way, substituting the first letters of many words with the letter 'r'. His catchphrase is "Scooby-Dooby-Doo!"
     ''',
     'What kind of animal is scooby from scooby doo?', 'Conversational Question Answering'
     ),
     ('Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.', 
     '''Subject: faa demos 
    Dan: PM Team, Attached are some general ideas and issues around developing new demos for our new target markets. Please review and provide feedback. Also, please provide links where we can learn more about various FAA applications. Thanx, Dan. 
    Alex: Dan, Thanks for putting the high level descriptions together. My questions are: *Is it practical to do an EAI demo given the inherent complexity of application integration? ... * Should we delay looking at Outlook for now?... *What do you think that timelines are developing these demos? ... Alex 
    Dan: Alex, Thanks for the feedback, please see my comments below:
     ''',
     'what does Dan ask PM team to do?', 'Conversational Question Answering'
     ),
     ('Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.', 
     '''Carlos Alcaraz, at just 19, completed an improbable journey on Sunday in Flushing Meadows as he defeated No. 5 Casper Ruud to win the 2022 US Open. Alcaraz came away with a 6-4, 2-6, 7-6, 6-2 win over Ruud to win his first career Grand Slam title.
     
     In doing so, Alcaraz became the second-youngest player to win a men's US Open title at 19 years, 129 days old, only trailing Pete Sampras. In addition, Alcaraz is the seventh youngest male or female to ever win a Grand Slam tournament. With the Grand Slam victory, Alcaraz becomes the No. 1 ranked player in the world. Additionally, the 19-year-old budding star is also the youngest player to ever be ranked as the world's No. 1 player.
     ''',
     'who won the 2022 US Open? EOS Carlos Alcaraz EOS how old?', 'Conversational Question Answering'
     ),
     (
        'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.',
        '''Over-the-counter medications such as ibuprofen (Advil, Motrin IB, others), acetaminophen (Tylenol, others) and aspirin.
        ''',
        'I have a headache, what should I do?', "Grounded Response Generation"
     ),
     (
        'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.',
        '''The best Stardew Valley mods PCGamesN_0 / About SMAPI
        ''',
        'My favorite game is stardew valley. stardew valley is very fun.', "Grounded Response Generation"
     ),
     (
        'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.',
        '''Wong Kar-wai BBS (born 17 July 1958) is a Hong Kong film director, screenwriter, and producer. His films are characterised by nonlinear narratives, atmospheric music, and vivid cinematography involving bold, saturated colours. A pivotal figure of Hong Kong cinema, Wong is considered a contemporary auteur, and ranks third on Sight & Sound's 2002 poll of the greatest filmmakers of modern times.[note 1] His films frequently appear on best-of lists domestically and internationally.
        ''',
        'My favorite director is wrong kar wai. i think in modern cinema there is no other director is is making the medium as cool', "Grounded Response Generation"
     )
]

def generate(instruction, knowledge, dialog, top_p, min_length, max_length):
    
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    
    dialog = ' EOS '.join(dialog)
    
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"

    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, min_length = int(min_length), max_length = int(max_length), top_p = top_p, do_sample = True)
    
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #print(query)
    #print(output)
    return output

def api_call_generation(instruction, knowledge, query, top_p, min_length, max_length):

    dialog = [ query ]
    
    response = generate(instruction, knowledge, dialog,
                        top_p, min_length, max_length)
    return response

if __name__ == "__main__":
    
    for i in range(0,len(preset_examples)):
        #print(preset_examples[i])
        instruction_to_bot = preset_examples[i][0]
        knowledge_for_bot  = preset_examples[i][1]
        question_for_bot   = preset_examples[i][2]
        category           = preset_examples[i][3]
        
        answer_from_bot = api_call_generation(instruction_to_bot, knowledge_for_bot, question_for_bot, top_p, min_length, max_length)
        
        print(answer_from_bot)
    