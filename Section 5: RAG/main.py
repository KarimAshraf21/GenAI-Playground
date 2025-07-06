import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        deployment="text-embedding-3-small",
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    query = 'What is pinecone in machine learning?'
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print("result:", result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)
    combine_documents_chain = create_stuff_documents_chain( llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_documents_chain)
    
    result = retrieval_chain.invoke(input={"input": query})
    print("result:", result)

# ================================================================================================================================================================================
    template = '''
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
use three sentences maximum and keep the asnwer as concise as possible. Always say "Thanks for asking." at the end of your answer.
context: {context}
Question: {question}
'''

    custom_rag_prompt = PromptTemplate.from_template(template=template)
    # custom_rag_prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=template,
    # )
    rag_chain = ({"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt | llm
)
    
    res = rag_chain.invoke(query)
    print("result:", res)