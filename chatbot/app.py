import streamlit as st
import os
import sys

chatbot_dir = os.path.dirname(os.path.abspath(__file__))

if chatbot_dir not in sys.path:
    sys.path.insert(0, chatbot_dir)

from rag_agent import app, memory

st.set_page_config(
    page_title="BANK-NBFC-CHATBOT",
    layout="centered"
)

st.title("BANK-NBFC-CHATBOT")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({"question": prompt})
                response = result["answer"]

                memory.save_turn(
                    user_message=prompt,
                    assistant_reply=response
                )

            except Exception as e:
                response = f"Something went wrong: {str(e)}"

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )