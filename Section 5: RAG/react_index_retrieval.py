from langsmith import Client
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI




load_dotenv()   

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":

    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    ai = AzureChatOpenAI(
        azure_deployment=os.getenv("LLM_azure_deployment"),
        azure_endpoint=os.getenv("LLM_azure_endpoint"),
        api_version= os.getenv("LLM_api_version"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        deployment="text-embedding-3-small",
    )   
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX2_NAME"),
        embedding=embeddings,
    )

    rag_chain = ({"context": vectorstore.as_retriever() | format_docs, "input": RunnablePassthrough()}
    | retrieval_qa_chat_prompt | ai
)
    
    res = rag_chain.invoke("Give me a gist in ReAct in 4 sentences")
    print("result:", res.content)


