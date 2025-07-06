import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain import hub
from agent_tools.tools import get_twitter_url

def twitter_lookup(name: str) -> str:
    """
    Lookup the Twitter url for a given name.
    """
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    template = '''You will receive an inquiry.
        If the inquiry is a full name of one of our employees, retrieve their Twitter profile URL from our internal database and respond only with the URL.
        If the inquiry is about something else, respond based on your general knowledge. 
        Inquiry: {inquiry}'''
    prompt_template = PromptTemplate(template= template, input_variables=['inquiry'])
    tools_for_agent = [
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
    
# if __name__ == "__main__":
#     print(lookup("John Doe"))
    
