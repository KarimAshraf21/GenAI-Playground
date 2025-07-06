def twitter_data(url: str) -> str:
    """
    Get the Twitter data for a given url.
    """
    if url == "https://twitter.com/johndoe":
        return "John Doe loves pets and enjoys hiking on weekends."
    elif url == "https://twitter.com/janesmith":
        return "Jane Smith loves cooking and enjoys painting in her free time."
    else:
        return "No Twitter data found for this url."
