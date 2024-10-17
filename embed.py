import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import time


class DataProcessor:
    def __init__(self, chunk_size=2000, chunk_overlap=200, embedding_batch_size=100):
        # Load environment variables
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.chunk_size = chunk_size
        self.embedding_batch_size = embedding_batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()

        self.vector_store = None
        self.dataframes = {}

    def process_and_split_chunk(self, chunk, chunk_number, file_name):
        documents = []
        for index, row in chunk.iterrows():
            row_content = row.to_json()
            doc = Document(
                page_content=row_content,
                metadata={
                    "row": index + (chunk_number * self.chunk_size),
                    "chunk": chunk_number,
                    "file": file_name,
                },
            )
            split_docs = self.text_splitter.split_documents([doc])
            documents.extend(split_docs)
        return documents

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_batch(self, texts, metadatas):
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return FAISS.from_embeddings(
                zip(texts, embeddings), self.embeddings, metadatas=metadatas
            )
        except Exception as e:
            print(f"Error occurred during embedding: {e}. Retrying...")
            raise

    def process_file(self, file_path):
        file_name = os.path.basename(file_path)
        total_documents = 0
        batch_texts = []
        batch_metadatas = []

        for chunk_number, chunk in enumerate(
            pd.read_csv(file_path, chunksize=self.chunk_size)
        ):
            chunk_documents = self.process_and_split_chunk(
                chunk, chunk_number, file_name
            )
            total_documents += len(chunk_documents)

            batch_texts.extend([doc.page_content for doc in chunk_documents])
            batch_metadatas.extend([doc.metadata for doc in chunk_documents])

            if len(batch_texts) >= self.embedding_batch_size:
                self.process_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

            print(
                f"Processed chunk {chunk_number + 1} of {file_name}, Total documents: {total_documents}"
            )

        # Process any remaining documents
        if batch_texts:
            self.process_batch(batch_texts, batch_metadatas)

        print(f"Total loaded documents for {file_name}: {total_documents}")
        return total_documents

    def process_batch(self, texts, metadatas):
        chunk_vector_store = self.embed_batch(texts, metadatas)
        if self.vector_store is None:
            self.vector_store = chunk_vector_store
        else:
            self.vector_store.merge_from(chunk_vector_store)
        time.sleep(1)  # Add a small delay to avoid hitting rate limits

    def save_vector_store(self, save_dir="Embeddings"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.vector_store.save_local(save_dir)
        print(f"Vector store saved to {save_dir}")

    def process_and_analyze_file(self, file_path):
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        # Process and embed the file
        total_documents = self.process_file(file_path)

        print(f"Processed {total_documents} documents for {file_name}.")
        return total_documents
