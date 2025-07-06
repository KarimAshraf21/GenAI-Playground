from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    # Load the PDF file
    loader = PyPDFLoader("Section 5: RAG/2210.03629v3.pdf")
    documents = loader.load()

    # Print the number of documents loaded
    print(f"Number of documents loaded: {len(documents)}")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=0,
    )

    docs = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(docs)}")

    model_name = "text-embedding-3-small"
    deployment = "text-embedding-3-small"
    embeddings = AzureOpenAIEmbeddings(
        model=model_name,
        deployment=deployment,
    )

    PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=os.getenv("INDEX2_NAME"),)
