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
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import uuid


class Chatbot:
    def __init__(self, save_dir):
        load_dotenv()
        # first generate a session_id
        self.session_id = str(uuid.uuid4())

        # Initialize the Langfuse client
        self.langfuse = Langfuse()

        self.langfuse_prompt = self.langfuse.get_prompt("chatbot-prompt")
        self.version = self.langfuse_prompt.version
        self.model = self.langfuse_prompt.config["model"]
        self.temperature = self.langfuse_prompt.config["temperature"]
        self.tags = [
            "chatbot-prompt",
            "chatbot-prompt-" + f"v{str(self.version)}",
            self.model,
        ]

        self.langfuse_handler = CallbackHandler(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
            session_id=self.session_id,
            tags=self.tags,
        )
        self.system_template = SystemMessagePromptTemplate.from_template(
            self.langfuse_prompt.get_langchain_prompt(),
            metadata={"langfuse_prompt": self.langfuse_prompt},
        )

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
        llm = ChatOpenAI(temperature=self.temperature, model_name=self.model)

        human_template = "{question}"

        messages = [
            self.system_template,
            HumanMessagePromptTemplate.from_template(human_template),
        ]

        # print(messages)
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
        result = self.conversation_chain.invoke(
            {"question": query},
            config={
                "callbacks": [self.langfuse_handler],
            },
        )
        return result["answer"], result["source_documents"]
