"""
Convenience functions for interacting with Wikipedia's API.
"""
from typing import List

import requests
import wikipediaapi
from langchain_core.tools import BaseTool

ARTICLE_NOT_FOUND_MESSAGE = "Article not found or disambiguation error."
NUMBER_OF_SEARCH_RESULTS = 5
USER_AGENT = "WikiGPT (wikigpt777@gmail.com)"


def validate_input(input_text):
    if input_text.startswith("Topic: "):
        topic = input_text[len("Topic: "):].strip()
        if topic:
            return topic, True
        else:
            return "Please provide a topic after 'Topic: '", False
    else:
        return "Please start your input with 'Topic: ' followed by the topic name.", False


# Function to interactively select a Wikipedia article title
def select_wikipedia_title(titles):
    print("Please choose the closest topic from the list:")
    for i, title in enumerate(titles):
        print(f"{i + 1}: {title}")

    choice = int(input("Enter the number of the topic: ")) - 1
    return titles[choice]


def search_wikipedia(query, results_limit=5) -> List[str]:
    """Returns a list of results_limit Wikipedia titles for a given search query."""
    language_code = 'en'
    search_query = query
    number_of_results = results_limit
    headers = {
        'User-Agent': USER_AGENT
    }

    base_url = 'https://api.wikimedia.org/core/v1/wikipedia/'
    endpoint = '/search/page'
    url = base_url + language_code + endpoint
    parameters = {'q': search_query, 'limit': number_of_results}
    response = requests.get(url, headers=headers, params=parameters)
    data = response.json()
    titles = []
    for result in data["pages"]:
        titles.append(result["title"])
    return titles


def fetch_wikipedia_article(title):
    wiki_wiki = wikipediaapi.Wikipedia(USER_AGENT)
    page = wiki_wiki.page(title)
    if page.exists():
        return page.text
    else:
        return None


def preprocess_text(text):
    import re
    text = re.sub(r'\[\d+\]', '', text)  # Remove references like [1], [2], etc.
    return text




class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearch"
    description = "Searches Wikipedia for articles related to a query"

    def _run(self, input: str):
        query = input
        return search_wikipedia(query, NUMBER_OF_SEARCH_RESULTS)


class WikipediaFetchTool(BaseTool):
    name = "WikipediaFetch"
    description = "Fetches a Wikipedia article by title"

    def _run(self, title: str):
        article = fetch_wikipedia_article(title)
        if article:
            return preprocess_text(article)
        else:
            return ARTICLE_NOT_FOUND_MESSAGE


class UserInputValidationTool(BaseTool):
    name = "UserInputValidationTool"
    description = "Validates the user input format"

    def _run(self, user_input: str):
        return validate_input(user_input)


class SelectionTool(BaseTool):
    name = "SelectionTool"
    description = "Prompts user to select a Wikipedia article title"

    def _run(self, titles: list):
        return select_wikipedia_title(titles)


