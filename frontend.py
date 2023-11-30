import streamlit as st
import requests
import json

st.set_page_config(page_title="🤗💬 Chatbot")
st.title("HuggingChat")

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/inference"

def call_inference_endpoint(prompt):
    payload = {
        "prompt": prompt,
        "pinecone_index_name": index_name
    }
    response = requests.post(url = FASTAPI_URL, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        return None


# Streamlit app
with st.sidebar:
    index_name = st.text_input("Pinecone index name", "retrieval-augmentation")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_inference_endpoint(prompt)
            st.write(response[0]['generated_text'])

    message = {"role": "assistant", "content": response[0]['generated_text']}
    if len(st.session_state.messages)>3:
        st.session_state.messages = st.session_state.messages[1:]
    st.session_state.messages.append(message)
