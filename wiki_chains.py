"""
Module containing the major components needed to support a Question
Answering pipeline built around an LLM model.
"""
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from prompts import qa_generator_template, tutor_template


store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


def build_qa_generator_chain(model, temperature, openai_api_key):
    """
    Builds a chain for generating questions and answers using an LLM model.
    """
    llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=openai_api_key)
    json_runnable = RunnableParallel(passed=RunnablePassthrough(), output=JsonOutputParser())
    return qa_generator_template | llm | json_runnable


def build_tutor_chain(model, temperature, openai_api_key):
    """
    Builds a chain for tutoring a user on a given topic using an LLM model.
    """
    llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=openai_api_key)
    return tutor_template | llm | StrOutputParser()
