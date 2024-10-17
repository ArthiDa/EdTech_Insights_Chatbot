import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_openai import OpenAIEmbeddings


class Chatbot:
    def __init__(self, save_dir):
        load_dotenv()

        # Ensure you have set the OpenAI API key in your environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.embeddings = OpenAIEmbeddings()
        self.save_dir = save_dir
        self.vector_store = self.load_index()
        self.conversation_chain = self.create_conversational_chain()

    def load_index(self):
        return FAISS.load_local(
            str(self.save_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def create_conversational_chain(self):
        # Initialize the language model
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

        # Create a custom prompt template
        system_template = """You are an intelligent assistant specialized in analyzing and generating insights from CSV files containing organizational data. Your goal is to answer questions, provide detailed insights, and generate relevant analyses based on the information within the CSV files. You can compute statistics, identify trends, and offer recommendations if the data allows it. If the data doesn't contain enough information to answer a query, respond with: 'I'm sorry, but the data does not provide enough information to answer that question.'
        
        {context}
        """
        human_template = "{question}"

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        # Initialize the memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        # Create the conversational chain
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 6}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def chat(self, query):
        result = self.conversation_chain.invoke({"question": query})
        return result["answer"], result["source_documents"]
