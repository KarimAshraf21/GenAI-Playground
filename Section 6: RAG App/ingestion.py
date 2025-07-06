from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
import os   

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    deployment="text-embedding-3-small",
)


def ingest_docs():
    loader = ReadTheDocsLoader(
        "Section 6: RAG App/langchain-docs/api.python.langchain.com/en/latest/agents",
    )

    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=0,
    )

    documents = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(documents)} chunks.")

    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name= "langchain-documentation-index",
    )
    print("Ingestion complete.")
if __name__ == "__main__":
    ingest_docs()
