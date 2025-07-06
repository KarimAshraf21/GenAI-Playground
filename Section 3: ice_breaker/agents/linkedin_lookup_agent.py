import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain import hub
from agent_tools.tools import get_linkedin_url, general_response, get_twitter_url
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
load_dotenv()

def linkedin_lookup(name: str) -> str:
    """
    Lookup the LinkedIn url for a given name.
    """
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        azure_endpoint= "https://19p60-mcm4jmft-eastus2.cognitiveservices.azure.com/",
        api_version= "2024-12-01-preview",
        api_key= "5I83at5HBifOjZAUB2jfwuy0N7cptWFLdJA0r16keBXQcqv0JHXeJQQJ99BGACHYHv6XJ3w3AAAAACOGsxbT",
    )
    # llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    template = '''You will receive an inquiry.
        If the inquiry is a full name of one of our employees, retrieve their LinkedIn and Twitter profile URL from the two tools from our internal database and respond only with the URL.
        If the inquiry is about something else, respond based on your general knowledge. 
        Inquiry: {inquiry}'''
    prompt_template = PromptTemplate(template= template, input_variables=['inquiry'])
    tools_for_agent = [
        Tool(
            name = "search database for linkedin url",
            func= get_linkedin_url, 
            description = "useful when you need to get linkedin page url" 
        ),

        Tool(
            name = 'general response',
            func = general_response,
            description = "useful when you need to answer with a general repsonse"
        ),
        Tool(
            name = "search database for twitter url",
            func= get_twitter_url, 
            description = "useful when you need to get twitter page url" 
        )
    ]
    react_prompt = hub.pull('hwchase17/react')
    agent = create_react_agent(llm = llm, tools = tools_for_agent, prompt = react_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools_for_agent, verbose = True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input = {"input": prompt_template.format_prompt(inquiry = name)}
    )

    return result['output']
    
if __name__ == "__main__":
     print(linkedin_lookup("John Doe"))
    
