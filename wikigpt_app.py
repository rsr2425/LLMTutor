import logging
from textwrap import dedent
from typing import List, Tuple

import streamlit as st
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from prompts import qa_generator_template, tutor_template
from wiki import validate_input, search_wikipedia, fetch_wikipedia_article


TEMPERATURE = 0.7
MODEL_NAME = "gpt-4o"


openai_api_key = st.secrets["OPENAI_API_KEY"]

show_token_usage = st.sidebar.checkbox("Close Token Usage", value=False)
close_warning = st.sidebar.checkbox("Close Warning", value=False)

st.title("WikiGPT: A Simple LLM Tutor")
if not close_warning:
    st.warning("This is a simple AI chatbot using OpenAI's API. Honestly make no guarantees around quality. YMMV.")

if "topic_chosen" not in st.session_state:
    st.session_state.topic_chosen = False

WELCOME_MESSAGE = (
        "ai",
        dedent(
            """
            Hello! I am an AI chatbot tutor. What topic would you like to learn? I will ask you a series
            of five questions and then grade you on your responses.
            
            The questions will be based on the topic of your choosing, fetched from Wikipedia.
    
            Please provide the name of the topic you would like to learn using the format: Topic: [topic name]
            """)
    )

if "token_count" not in st.session_state:
    st.session_state.token_count = 0

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)

    json_runnable = RunnableParallel(passed=RunnablePassthrough(), output=JsonOutputParser())
    st.session_state.qa_generator_chain = qa_generator_template | llm | json_runnable
    st.session_state.tutor_chain = tutor_template | llm

    st.session_state.tutor_conversation = []
    st.session_state.questions = None
    st.session_state.content = None
    st.session_state.topic = None
    st.session_state.articles = []
    st.session_state.begun_tutoring = False


def generate_wiki_response(input_text: str) -> Tuple[str, bool]:
    if not st.session_state.topic_chosen:
        validation_result, is_valid = validate_input(input_text)
        if not is_valid:
            return validation_result, False

        # if is_valid, validation_result is topic
        st.session_state.articles = search_wikipedia(validation_result)
        if len(st.session_state.articles) == 0:
            return "No search results found. Please select another topic.", False

        st.session_state.topic_chosen = True
        return "Please select a topic from the list below: \n" + ",\n".join(st.session_state.articles), False
    elif user_input not in st.session_state.articles:
        return "Please select a topic from the list below: \n" + ",\n".join(st.session_state.articles), False
    else:
        # TODO hard-coded for now to make sure I don't spend a bunch of money sending useless context
        return fetch_wikipedia_article(input_text)[:8000], True


def generate_qa_response(content: str, num_questions=5) -> List[str]:
    response = st.session_state.qa_generator_chain.invoke(
        {
            "content": content,
            "num_questions": num_questions
        }
    )
    st.session_state.token_count += response['passed'].usage_metadata['total_tokens']
    return response['output']['questions']


def generate_tutor_response(content: str, questions: List[str]) -> BaseMessage:
    response = st.session_state.tutor_chain.invoke(
        {
            "content": content,
            "questions": questions,
            "conversations": st.session_state.tutor_conversation
        }
    )
    st.session_state.token_count += response.usage_metadata['total_tokens']
    return response


# 1) Fetch content from Wikipedia
# 2) Generate questions
# 3) Ask questions and grade them
with st.form("my_form"):
    user_input = st.text_area("Enter text:", "Topic: [topic name]")
    st.info(f"ℹ️ Powered by OpenAI's {MODEL_NAME} model")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API key", icon="⚠")

    if submitted and openai_api_key.startswith("sk-"):
        if st.session_state.content is None:
            wiki_response, content_retrieved = generate_wiki_response(user_input)
            if content_retrieved:
                st.session_state.topic = user_input
                st.session_state.content = wiki_response
            else:
                st.session_state.tutor_conversation = [("ai", wiki_response)]

        if st.session_state.content is not None:
            if st.session_state.questions is None:
                st.session_state.questions = generate_qa_response(st.session_state.content)
            # follow-ups
            if st.session_state.begun_tutoring:
                st.session_state.tutor_conversation.append(("human", user_input))
            # initial chat
            else:
                st.session_state.begun_tutoring = True
                st.session_state.tutor_conversation.append(("human", f"Topic: {st.session_state.topic}"))

            ai_msg = generate_tutor_response(st.session_state.content, st.session_state.questions)
            st.session_state.tutor_conversation.append(("ai", ai_msg.content))


conversation = [WELCOME_MESSAGE] + st.session_state.tutor_conversation
for msg in conversation[::-1]:
    if msg[0] == "human":
        st.info(f"👤 {msg[1]}")
    elif msg[0] == "ai":
        st.info(f"🤖 {msg[1]}")
if show_token_usage:
    st.info(f"Running token count: {st.session_state.token_count}")
