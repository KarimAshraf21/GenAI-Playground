import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":

    model_name = "text-embedding-3-small"
    deployment = "text-embedding-3-small"
    # you can use different document loaders other than text, we have tons of document loaders
    # check langchain documentation fro more loaders
    loader = TextLoader("Section 5: RAG/docs.txt")
    print("loader: ",loader)
    documents = loader.load()
    print("documents:",documents)
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("texts:", texts)
    print("number of texts:", len(texts))
    print("ingesting...")

    embeddings = AzureOpenAIEmbeddings(
    model=model_name,
    deployment=deployment,
    )
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),)
    print("finish")