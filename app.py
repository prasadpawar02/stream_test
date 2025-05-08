# app.py

# app.py

import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from agent_module import agent_with_chat_history, get_session_history

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 LangSmith Chatbot")

# Generate a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{uuid.uuid4().hex[:8]}"

chat_history = get_session_history(st.session_state.session_id)

# Show history
for msg in chat_history.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Input field
user_input = st.chat_input("Ask anything about LangSmith...")

if user_input:
    st.chat_message("user").write(user_input)

    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )

    st.chat_message("assistant").write(response.content)