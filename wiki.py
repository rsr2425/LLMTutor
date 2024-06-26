"""
Convenience functions for interacting with Wikipedia's API.
"""
import requests
import wikipediaapi
from langchain.agents import AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool

ARTICLE_NOT_FOUND_MESSAGE = "Article not found or disambiguation error."
FAKE_WIKI_CONTENT = "This is a fake Wikipedia article. It is not real."
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


class WikipediaChain(SimpleSequentialChain):
    """Fetches data from wikipedia based on user input."""
    def __init__(self, agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent
        self._previous_titles = []

    def _call(self, inputs):
        user_input = inputs['user_input']

        if len(self._previous_titles) == 0:
            validation_message, is_valid = validate_input(user_input)
            if is_valid:
                topic = validation_message
                search_results = self._agent.run(f"Search Wikipedia for: {topic}")
                return {"message": f"Search results for '{topic}':\n{search_results}\nEnter the exact title of the article you want to read:"}
            else:
                return {"message": validation_message}
        else:
            # Handle fetching the article based on previous titles
            if user_input in self.previous_titles:
                article_content = self.agent.run(f"Fetch Wikipedia article: {user_input}")
                if ARTICLE_NOT_FOUND_MESSAGE in article_content:
                    return {"message": f"'{user_input}' does not exist. Please choose another article or topic."}
                else:
                    self.previous_titles = []  # Reset the titles after successful fetch
                    return {"message": f"Content found.", "content": f"{article_content}"}
            else:
                # TODO make it so that user can choose a new topic with 'Topic: [topic name]'
                return {"message": "Please choose a valid article from the search results."}


class WikipediaSearchAgent(AgentExecutor):
    def __init__(self, agent, tools, memory, *args, **kwargs):
        super().__init__(agent, tools, *args, **kwargs)
        self.memory = memory

    def _plan(self, user_input):
        parsed_input, is_topic = validate_input(user_input)
        if is_topic:
            topic = parsed_input
            return [AgentAction(name="WikipediaSearch", input=topic)]
        else:
            return [AgentFinish(return_values={"output": parsed_input}, log="")]

    def _postprocess(self, agent_action: AgentAction, action_result: str):
        if agent_action.name == "WikipediaSearch":
            self.memory["article_titles"] = action_result.split('\n')
            return {"message": f"Search results for '{agent_action.input}':\n{action_result}\nEnter the exact title of the article you want to read:"}
        else:
            return {"message": action_result}



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


def search_wikipedia(query, results_limit=5):
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