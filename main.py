from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Nikola Tesla.
    Answer questions through Tesla's questioning and reasoning...
    You will speak from your point of view. You will share latest and accurate data 
    when the user ask's for it. For example, if the user asks about theory of relativity, 
    you will answer it in layman terms and explain briefly in 2-3 lines and understandably.
    Speak like a modern man. Don't be too grammatical and poetic.

"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=gemini_key,
                             temperature=0.5)

prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                           (MessagesPlaceholder(variable_name="history")),
                                           ("user", "{input}")])

chain = prompt | llm | StrOutputParser()

print("Hi, I am Tesla, how can i help you?")


def chat(user_input, hist):
    print(user_input, hist)

    langchain_history = []
    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    response = chain.invoke({"input": user_input, "history": langchain_history})

    return "", hist + [{"role": "user", "content": user_input},
                       {"role": "assistant", "content": response}]


page = gr.Blocks(title="Chat with Leo",
                 theme=gr.themes.Soft())


def clear_chat():
    return "", []


with page:
    gr.Markdown("""
    # \U0001F9E0 Chat with Tesla
    Welcome to your personal conversation with Tesla!

    """)

    chatbot = gr.Chatbot(type="messages", avatar_images=[None, 'download.jpg'],
                         show_label=False)

    msg = gr.Textbox(show_label=False, placeholder='Ask Anything...')
    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat", variant='Secondary')
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch()
