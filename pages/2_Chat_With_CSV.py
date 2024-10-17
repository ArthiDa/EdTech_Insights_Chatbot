import streamlit as st
from streamlit_chat import message
import pandas as pd
from embed import DataProcessor
from chatbot import Chatbot
import os
import base64

st.set_page_config(page_title="EdTech Insights Chatbot", page_icon="🎓", layout="wide")

# Initialize session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "files_uploaded" not in st.session_state:
    st.session_state["files_uploaded"] = False
if "data_processor" not in st.session_state:
    st.session_state["data_processor"] = DataProcessor()
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = None


def get_chat_download_link(chat_history):
    """Generate a download link for the chat history"""
    chat_text = ""
    for user, bot in zip(st.session_state["past"], st.session_state["generated"]):
        chat_text += f"User: {user}\nChatbot: {bot}\n\n"

    b64 = base64.b64encode(chat_text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download chat history</a>'


st.title("EdTech Insights Chatbot")

# Only show file uploader if files haven't been uploaded yet
if not st.session_state["files_uploaded"]:
    uploaded_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True, key="file_uploader"
    )
    if uploaded_files:
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            # Save the uploaded file temporarily
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())

            # Process and analyze the file
            st.session_state["data_processor"].process_and_analyze_file(file.name)

            # Remove the temporary file
            os.remove(file.name)

            # Update progress bar
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.session_state["files_uploaded"] = True
        st.session_state["data_processor"].save_vector_store()
        st.session_state["chatbot"] = Chatbot(
            save_dir="Embeddings"
        )  # Initialize the chatbot
        st.success("All files processed and analyzed. Chat interface is ready!")
        st.rerun()

# If files have been uploaded, show the chat interface
if st.session_state["files_uploaded"]:
    st.subheader("Chat with your data")

    # Clickable option to show CSV previews
    show_previews = st.checkbox("Show CSV file previews")

    if show_previews:
        for file_name, df in st.session_state["data_processor"].dataframes.items():
            with st.expander(f"Preview of {file_name}"):
                st.write(df.head())

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
    if st.session_state["generated"]:
        st.markdown(
            get_chat_download_link(st.session_state["generated"]),
            unsafe_allow_html=True,
        )
else:
    st.info("Please upload at least one CSV file to start the chat.")
