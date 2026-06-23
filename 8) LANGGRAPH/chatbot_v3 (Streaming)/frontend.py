import streamlit as st
from backend import chatBot # Change this
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="ChatBuddy",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("ChatBuddy")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'messageHistory' not in st.session_state:
    st.session_state['messageHistory'] = []
# loading the conversation history
for message in st.session_state['messageHistory']:
    with st.chat_message(message['role']):
        st.text(message['content'])



user_input = st.chat_input("Type Your Message Here ......")
if user_input:
    # To Maintain the history
    st.session_state['messageHistory'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        chatBotReply = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatBot.stream(
                {'messages': [HumanMessage(content= user_input) ]},
                config = {'configurable': {'thread_id': 'thread-1'}},
                stream_mode = 'messages'
            )
        )
    st.session_state['messageHistory'].append({'role': 'assistant', 'content': chatBotReply})
