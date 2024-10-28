# EdTech Insights Chatbot

This project is an AI-powered chatbot interface designed to help district admins and principals gain insights from educational data collected by Organization X, an ed-tech platform for K-12 education.

## Features

- Multi-page Streamlit application with Home and Chat Interface pages
- Natural language processing to understand complex queries about educational data
- Generation of insightful observations from multiple CSV data sources
- User-friendly chat interface for interacting with the data

## Installation

1. Clone this repository and navigate to the project directory
2. Create a virtual python environment
   - For Windows `python -m venv venv`
   - For Linux/MAC `python3 -m venv venv`
3. Activate the virtual environment
   - For Window `./venv/Source/activate`
   - For Linux/MAC `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## [Langfuse configuration](https://github.com/ArthiDa/EdTech_Insights_Chatbot/blob/main/LANGFUSE_README.md)

## Usage

Run the Streamlit app:

```
streamlit run Home.py
```

Navigate to the Chat Interface page to start interacting with the chatbot.

## Get the embeddings

- Go to [here](https://drive.google.com/file/d/10Hu07HXBHgrAeZd9i-MwvGncd_apMFbm/view?usp=sharing) and download the zip.
- Extract the zip and put index.faiss and index.pkl file to OrgX_Embeddings directory.
