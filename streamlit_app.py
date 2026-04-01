from __future__ import annotations

import streamlit as st

from agentic_memory_support_demo.chatbot import SupportChatbot


st.set_page_config(page_title="Agentic Memory Support Demo", page_icon="🧠", layout="wide")

st.title("Agentic Memory Support Demo")
st.caption("A support chatbot that gets measurably better when memory is enabled.")

memory_enabled = st.sidebar.toggle("Enable memory", value=True)
st.sidebar.markdown(
    "This app remembers user facts, troubleshooting history, and support procedures "
    "through the `agentic-memory` SDK."
)

if "bot" not in st.session_state or st.session_state.get("memory_enabled") != memory_enabled:
    st.session_state.bot = SupportChatbot(enable_memory=memory_enabled)
    st.session_state.memory_enabled = memory_enabled
    st.session_state.messages = []
    st.session_state.debug = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Try: My name is Priya. / I'm on the enterprise plan. / What plan am I on?")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = st.session_state.bot.reply(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    st.session_state.debug.append(
        {
            "message": user_prompt,
            "recalled_memories": response.recalled_memories,
            "recalled_procedures": response.recalled_procedures,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response.text)

st.subheader("Last Retrieval")
if st.session_state.debug:
    latest = st.session_state.debug[-1]
    left, right = st.columns(2)
    with left:
        st.markdown("**Recalled memories**")
        for item in latest["recalled_memories"] or ["none"]:
            st.code(item)
    with right:
        st.markdown("**Recalled procedures**")
        for item in latest["recalled_procedures"] or ["none"]:
            st.code(item)
else:
    st.info("Send a message to see which memories and procedures were recalled.")
