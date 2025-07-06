import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langsmith import Client
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment: str = os.getenv("embedding_deployment")

vector_store_address: str = os.getenv("vector_store_address")
vector_store_password: str = os.getenv("vector_store_password")

if __name__ == "__main__":
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment,
        openai_api_version=azure_openai_api_version,
        azure_endpoint=azure_endpoint,
        api_key=azure_openai_api_key,
    )
    
    # client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    index_name ="langchain-vector-demo"
    vector_store = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    )

    loader = PyPDFLoader("Section 5: RAG/2210.03629v3.pdf")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=0,
    )    
    
    docs = text_splitter.split_documents(documents)

    vector_store.add_documents(documents=docs)