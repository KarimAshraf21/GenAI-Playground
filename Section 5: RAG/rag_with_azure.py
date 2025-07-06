import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langsmith import Client
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


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
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    
    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)

    # llm = AzureChatOpenAI(
    #     azure_deployment=os.getenv("LLM_azure_deployment"),
    #     azure_endpoint=os.getenv("LLM_azure_endpoint"),
    #     api_version= os.getenv("LLM_api_version"),
    #     api_key=os.getenv("LLM_API_KEY"),
    # )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)


    vector_store = AzureSearch(
    azure_search_endpoint = vector_store_address,
    azure_search_key = vector_store_password,
    index_name = "langchain-vector-demo",
    embedding_function = embeddings.embed_query,
    )

    combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever= vector_store.as_retriever(),
        combine_docs_chain=combine_documents_chain)
    

    result = retrieval_chain.invoke(input={"input": "What is ReAct?"})
    print("result:", result['answer'])

    # docs = vector_store.similarity_search(
    # query="what is ReAct?",
    # k=3,
    # search_type="similarity",
    # )
    # print(docs[0].page_content)