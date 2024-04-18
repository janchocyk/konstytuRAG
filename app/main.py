"""
Name: KonstytuRAG App (main.py)

Description:
An application that implements a graphical chat interface in the browser. The user can chat
with a chatbot whose task is to answer questions about the Polish constitution.

Author: Jan Chocyk

Version: 1.0

Setup:
- Install all required libraries from the requirements.txt file.
- Create a .env file and place it in the project's main folder.
- Sign up for Pinecone and create a vector index (details in README.md), copy the API key.
- Sign up for the OpenAI service and copy the API key.
- Add variables named PINECONE_API_KEY and OPENAI_API_KEY to the .env file, set the keys as their values.
- Run the following command in the main folder:
    ```python prepare_data\prepare_data.py``` - Windows
    ```python3 prepare_data/prepare_data.py``` - Linux
Running this script creates a vector database and may take a few minutes.

Execution:
- In the main folder, execute the command:
    ```streamlit run app\main.py --server.port 8080``` Windows
    ```streamlit run app/main.py --server.port 8080``` Linux
After running, the default browser window will open with the application.
"""

import sys
sys.path.append("..")

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

import time
import logging
from dotenv import load_dotenv

from konstytuRAG.app.tool import konstytuRAG


def stream_generate(response: str):
    """
    Description: A generator simulating the chatbot's thinking process, returning each word of the response every 0.05 seconds.

    Parameters:
    - response (str): The chatbot's response to be displayed in the chat window.

    Returns:
    (str): A single word from the response.
    """
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():
    """
    Description: The main function that constructs the chat interface from Streamlit components, initializes the RAG, 
    and handles the exchange of messages between the user and the chatbot.
    """
    logging.basicConfig(filename=f'app/chat.log', format='%(asctime)s %(message)s', level=logging.WARNING)

    load_dotenv()
    chat = konstytuRAG()
    chat.init_rag_chain()

    st.title('KonstytuRAG')
    instruction = '''Poznaj chat-bot, którego celem jest udzielić odpowiedzi na pytania dotyczące treści konstytucji Polski.\n
        Jeżeli chcesz uzyskać optymalne odpowiedzi od asystenta, to:\n
        - zadawaj pojedyncze, konkretne pytania, dotyczące treści konstytucji\n
        - dopytuj jeżeli asystent odpowiedział niedokładnie\n
        - nie wysyłaj wiadomości powitalnych, tylko od razu zadawaj pytania'''
    st.markdown(body=instruction, )

    if "buffer_memory" not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button('Restart chatu'):
        st.session_state.messages = []
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("Co byś chciał/chciała się dowiedzieć?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        logging.info(f'Human: {prompt}')
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            try:
                answer, source = chat.get_answer(prompt, st.session_state.buffer_memory.load_memory_variables({})['history'])
            except Exception as e:
                logging.error(f'Error: {e}')
            response = st.write_stream(stream_generate(answer))
            st.write(source)

        # Add assistant response to chat history
        answer_to_write = f'{answer}\n\n{source}' 
        st.session_state.messages.append({"role": "ai", "content": answer_to_write})
        st.session_state.buffer_memory.save_context({'input': prompt}, {'output': answer})
        logging.info(f'AI: {response}')


if __name__ == '__main__':
    main()
