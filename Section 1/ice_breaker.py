from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

information = """
John is a software engineer who lives in San Francisco and likes to play basketball.
"""

if __name__ == "__main__":
    print("Hello, World!")

    summary_template = """
    Given some {information} about a person, create:
    1. A short summary of the person
    2. two interesting facts about him/her
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

    chain = summary_prompt_template | llm | StrOutputParser()

    response = chain.invoke(input={"information": information})

    print(response)
