def linkedin_data(url: str) -> str:
    """
    Get the LinkedIn data for a given url.
    """
    if url == "https://www.linkedin.com/in/john-doe-1234567890/":
        return "John Doe is a software engineer who lives in San Francisco and likes to play basketball."
    elif url == "https://www.linkedin.com/in/jane-smith-1234567890/":
        return "Jane Smith is a software engineer who lives in New York and likes to play tennis."
    else:
        return "No LinkedIn data found for this url."
