import sys
sys.path.append("..")

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

import time
import logging
from dotenv import load_dotenv

from konstytuRAG.app.tool import konstytuRAG, customize_chat_history, test_prompt


def stream_generate(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():
    logging.basicConfig(filename=f'app/logs/chat_msgs.log', format='%(asctime)s %(message)s', level=logging.INFO)

    load_dotenv()
    chat = konstytuRAG()
    chat.init_rag_chain()

    st.title('KonstytuRAG')
    st.markdown(body='Poznaj chat-bot, którego celem jest udzielić odpowiedzi na pytania dotyczące treści konstytucji Polski.')
    st.markdown(
        body=
        """Jeżeli chcesz uzyskać optymalne odpowiedzi od asystenta, to:
        - zadawaj pojedyncze, konkretne pytania, dotyczące treści konstytucji
        - dopytuj jeżeli asystent odpowiedział niedokładnie
        - nie wysyłaj wiadomości powitalnych, tylko od razu zadawaj pytania
        """
    )
    if "buffer_memory" not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Initialize chat history
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

    # Accept user input
    if prompt := st.chat_input("Co byś chciał/chciała się dowiedzieć?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "content": prompt})
        logging.info(f'Human: {prompt}')
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):            
            answer, source = chat.get_answer(prompt, st.session_state.buffer_memory.load_memory_variables({})['history'])
            response = st.write_stream(stream_generate(answer))
            st.write(source)

        # Add assistant response to chat history
        answer_to_write = answer + '\n' + source 
        st.session_state.messages.append({"role": "ai", "content": answer_to_write})
        # st.session_state.messages = customize_chat_history(st.session_state.messages)
        st.session_state.buffer_memory.save_context({'input': prompt}, {'output': answer})
        logging.info(f'AI: {response}')


if __name__ == '__main__':
    main()
