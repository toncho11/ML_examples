# -*- coding: utf-8 -*-
"""
Model: DialoGPT
In the back-end is using the GPT2 model from OpenAI.Language processing is done using RoBERTa for
sentiment-analysis and spaCy for named-entity recognition and dependency plotting.
Paper: https://arxiv.org/abs/1911.00536

DialoGPT is superseded by GODEL, which outperforms DialoGPT.

The script will download and use the bot model "DialoGPT" from Microsoft to chat with you.
The chatbot will run locally on your computer.
It launches a web server where you perform the chat: http://127.0.0.1:7860 (check the Python's console output for more details)

Currently it gives an error: "RESOURCE_EXHAUSTED: OOM when allocating tensor with shape" on 4 GB GPU and 16 GB RAM.

pip install gradio

Git page: https://github.com/microsoft/DialoGPT
"""

from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
import gradio as gr
import spacy
from spacy import displacy
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import plotly.express as px
import plotly.io as pio

# configuration params
pio.templates.default = "plotly_dark"

# setting up the text in the page
TITLE = "<center><h1>Talk with an AI</h1></center>"
DESCRIPTION = r"""<center>This application allows you to talk with a machine/robot with state-of-the-art technology!!<br>
                 In the back-end is using the GPT2 model from OpenAI. One of the best models in text generation and comprehension.<br>
                 Language processing is done using RoBERTa for sentiment-analysis and spaCy for named-entity recognition and dependency plotting.<br>
                 The AI thinks he is a human, so please treat him as such, else he migh get angry!<br>
                 """
EXAMPLES = [
    ["What is your favorite videogame?"],
    ["What gets you really sad?"],
    ["How can I make you really angry? "],
    ["What do you do for work?"],
    ["What are your hobbies?"],
    ["What is your favorite food?"],
]
ARTICLE = r"""<center>
              Done by dr. Gabriel Lopez<br>
              For more please visit: <a href='https://sites.google.com/view/dr-gabriel-lopez/home'>My Page</a><br>
              For info about the chat-bot model can also see the <a href="https://arxiv.org/abs/1911.00536">ArXiv paper</a><br>
              </center>"""

# Loading necessary NLP models
# dialog
checkpoint = "microsoft/DialoGPT-medium"  # tf
model_gtp2 = TFAutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer_gtp2 = AutoTokenizer.from_pretrained(checkpoint)
# sentiment
checkpoint = f"cardiffnlp/twitter-roberta-base-emotion"
model_roberta = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer_roberta = AutoTokenizer.from_pretrained(checkpoint)
# NER & Dependency
nlp = spacy.load("en_core_web_sm")

# test-to-test : chatting function -- GPT2
def chat_with_bot(user_input, chat_history_and_input=[]):
    """Text generation using GPT2"""
    emb_user_input = tokenizer_gtp2.encode(
        user_input + tokenizer_gtp2.eos_token, return_tensors="tf"
    )
    if chat_history_and_input == []:
        bot_input_ids = emb_user_input  # first iteration
    else:
        bot_input_ids = tf.concat(
            [chat_history_and_input, emb_user_input], axis=-1
        )  # other iterations
    chat_history_and_input = model_gtp2.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer_gtp2.eos_token_id
    ).numpy()
    # print
    bot_response = tokenizer_gtp2.decode(
        chat_history_and_input[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )
    return bot_response, chat_history_and_input


# text-to-sentiment
def text_to_sentiment(text_input):
    """Sentiment analysis using RoBERTa"""
    labels = ["anger", "joy", "optimism", "sadness"]
    encoded_input = tokenizer_roberta(text_input, return_tensors="tf")
    output = model_roberta(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    return px.histogram(x=labels, y=scores, height=200)


# text_to_semantics
def text_to_semantics(text_input):
    """NER and Dependency plot using Spacy"""
    processed_text = nlp(text_input)
    # Dependency
    html_dep = displacy.render(
        processed_text,
        style="dep",
        options={"compact": True, "color": "white", "bg": "light-black"},
        page=False,
    )
    html_dep = "" + html_dep + ""
    # NER
    pos_tokens = []
    for token in processed_text:
        pos_tokens.extend([(token.text, token.pos_), (" ", None)])
    # html_ner = ("" + html_ner + "")s
    return pos_tokens, html_dep


# gradio interface
blocks = gr.Blocks()
with blocks:
    # physical elements
    session_state = gr.State([])
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            in_text = gr.Textbox(value="How was the class?", label="Start chatting!")
            submit_button = gr.Button("Submit")
            gr.Examples(inputs=in_text, examples=EXAMPLES)
        with gr.Column():
            response_text = gr.Textbox(value="", label="GPT2 response:")
            sentiment_plot = gr.Plot(
                label="How is GPT2 feeling about your conversation?:", visible=True
            )
            ner_response = gr.Highlight(
                label="Named Entity Recognition (NER) over response"
            )
            dependency_plot = gr.HTML(label="Dependency plot of response")
    gr.Markdown(ARTICLE)
    # event listeners
    submit_button.click(
        inputs=[in_text, session_state],
        outputs=[response_text, session_state],
        fn=chat_with_bot,
    )
    response_text.change(
        inputs=response_text, outputs=sentiment_plot, fn=text_to_sentiment
    )
    response_text.change(
        inputs=response_text,
        outputs=[ner_response, dependency_plot],
        fn=text_to_semantics,
    )

blocks.launch()