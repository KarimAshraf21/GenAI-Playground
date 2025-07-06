from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from third_parties.linkedin import linkedin_data
from agents.linkedin_lookup_agent import linkedin_lookup
from third_parties.twitter import twitter_data
from agents.twitter_lookup_agent import twitter_lookup
from output_parsers.output_parsers import summary_output_parser

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def ice_breaker(name: str) -> str:

    url = linkedin_lookup(name)
    information = linkedin_data(url)

    twitter_url = twitter_lookup(name)
    twitter_information = twitter_data(twitter_url)

    summary_template = """
    Given some {information} and {twitter_information} about a person, create:
    1. A short summary of the person
    2. two interesting facts about him/her

    The summary should be concise and informative, while the interesting facts should be unique and engaging.
    The output should be in the following format:
    {format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information","twitter_information"],
        partial_variables={"format_instructions": summary_output_parser.get_format_instructions()},
        template=summary_template,
    )

    # llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    # llm = ChatGroq(model = "meta-llama/llama-guard-4-12b", temperature=0.0)
    llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.0)


    chain = summary_prompt_template | llm | summary_output_parser

    response = chain.invoke(input={"information": information, "twitter_information": twitter_information})

    print(response)

if __name__ == "__main__":
    ice_breaker("John Doe")
