from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# tools scrape the web


def get_linkedin_url(name: str) -> str:
    """
    Get the LinkedIn url for a given name.
    """
    print(f"DEBUG: Received name = {name}")
    if name.startswith('name='):
        name = name.split('name=')[1].strip().strip('"')

    print(f"DEBUG: Cleaned name = {name}")

    # instead of this error handling, i can use in the tool arguments an argshema which is  a pydantic input schema.

    if name == "John Doe":
        return "https://www.linkedin.com/in/john-doe-1234567890/"
    elif name == "Jane Smith":
        return "https://www.linkedin.com/in/jane-smith-1234567890/"
    else:
        return "https://www.linkedin.com/in/tomato-doe-1234567890/"
    
def get_twitter_url(name: str) -> str:
    """
    Get the Twitter url for a given name.
    """
    print(f"DEBUG: Received name = {name}")
    if name.startswith('name='):
        name = name.split('name=')[1].strip().strip('"')

    print(f"DEBUG: Cleaned name = {name}")

    if name == "John Doe":
        return "https://twitter.com/johndoe"
    elif name == "Jane Smith":
        return "https://twitter.com/janesmith"
    else:
        return "https://twitter.com/tomatodoe"
    
def general_response(inquiry: str) -> str:

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

    chain =  llm | StrOutputParser()

    response = chain.invoke(input=inquiry)

    return response


