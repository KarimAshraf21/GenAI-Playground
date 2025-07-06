from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Any, Dict, List
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever


index = 'langchain-documentation-index'

load_dotenv()

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        deployment="text-embedding-3-small",
    )

    vector_store = PineconeVectorStore(
        index_name=index,
        embedding=embeddings,
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    print("Stuff documents chain created:", stuff_documents_chain)

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(), prompt=rephrase_prompt
    )
    print("History aware retriever created:", history_aware_retriever)
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    print("Retrieval chain created:", qa)

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])