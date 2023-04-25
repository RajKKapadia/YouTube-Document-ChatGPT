import gradio as gr
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

persist_directory = 'db'

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=False
)

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(),
    chain_type='stuff',
    retriever=db.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h,
    verbose=True
)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    res = qa(
        {
            'question': history[-1][0],
            'chat_history': history[:-1]
        }
    )
    history[-1][1] = res['answer']
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", label='Document GPT').style(height=750)
    with gr.Row():
        with gr.Column(scale=0.80):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.10):
            submit_btn = gr.Button(
                'Submit',
                variant='primary'
            )
        with gr.Column(scale=0.10):
            clear_btn = gr.Button(
                'Clear',
                variant='stop'
            )

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    demo.launch()
