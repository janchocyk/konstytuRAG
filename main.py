import streamlit as st
import random
import time
from dotenv import load_dotenv

from tool import konstytuRAG

load_dotenv()
chat = konstytuRAG()
chat.init_rag_chain()


#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(chat.get_answer(prompt, st.session_state.messages))
        print('udało się')
        print(st.session_state.messages)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "ai", "content": response})