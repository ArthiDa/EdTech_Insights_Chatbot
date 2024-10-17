import streamlit as st
from streamlit_chat import message
import base64
from chatbot import Chatbot

st.set_page_config(page_title="EdTech Insights Chatbot", page_icon="🎓", layout="wide")

# Initialize session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = Chatbot(
        save_dir="OrgX_Embeddings"
    )  # Initialize the chatbot


def get_chat_download_link(chat_history):
    """Generate a download link for the chat history"""
    chat_text = ""
    for user, bot in zip(st.session_state["past"], st.session_state["generated"]):
        chat_text += f"User: {user}\nChatbot: {bot}\n\n"

    b64 = base64.b64encode(chat_text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download chat history</a>'


st.title("EdTech Insights Chatbot")

st.subheader("Chat with OrgX data")

# Chat interface
response_container = st.container()
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output, source_documents = st.session_state["chatbot"].chat(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

        # Display source documents
        with st.expander("View Source Documents"):
            for i, doc in enumerate(source_documents, 1):
                st.write(f"Document {i}:")
                st.write(f"Content: {doc.page_content}")
                st.write(f"Metadata: {doc.metadata}")
                st.write("---")

    # Download chat history
    st.markdown(
        get_chat_download_link(st.session_state["generated"]),
        unsafe_allow_html=True,
    )
