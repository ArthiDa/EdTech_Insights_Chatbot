import streamlit as st

st.set_page_config(page_title="EdTech Insights Chatbot", page_icon="🎓", layout="wide")

st.title("EdTech Insights Chatbot")

st.markdown(
    """
## Problem Statement

Organization X is an ed-tech platform for K-12 education that enables school leaders to conduct walkthrough sessions to observe teachers and provide constructive feedback. The platform aims to improve teachers' instructional practices and students' test scores.

District Admins and Principals want to understand what's happening in their district/school, but the vast amount of data from classroom walkthroughs makes it challenging to extract high-level insights and specific highlights/lowlights.

## My Solution

I've developed an AI-powered chatbot interface that allows OrgsX Admin and Principles to engage with the data more effectively. My solution includes:

## 1. Data Processing
I preprocess multiple CSV files containing customer success metrics, next steps reports, session details, and customer journey information. This includes fixing inconsistencies in school names, district names, and school years, as well as aggregating the data for better performance.

## 2. Data Embedding and Retrieval
I applied OpenAI embeddings using Langchain to convert the processed datasets into vectors for fast, efficient information retrieval. This ensures that even large datasets can be queried quickly.

## 3. Conversational AI (RAG Model)
My chatbot utilizes a Retrieval-Augmented Generation (RAG) model powered by OpenAI's GPT-4 mini chat model. The chatbot retrieves relevant information from the embedded datasets based on user queries and generates insightful responses.

## 4. Insightful Observations
The system generates valuable insights by correlating data across different reports, identifying trends, and providing specific highlights or lowlights based on user queries.

## 5. User-Friendly Interface
I've created an intuitive chat interface using Streamlit, where users can ask questions and receive detailed responses about their educational data.

---

To get started, navigate to the Chat Interface using the sidebar and begin asking questions about your educational data!

"""
)


st.sidebar.success("Select a page above.")
