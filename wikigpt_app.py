import logging
from textwrap import dedent
from typing import List, Tuple

import streamlit as st
from langchain_core.messages import BaseMessage

from users import UserState
from wiki_chains import build_qa_generator_chain, build_tutor_chain
from wiki import validate_input, search_wikipedia, fetch_wikipedia_article

TEMPERATURE = 0.7
MODEL_NAME = "gpt-4o"

openai_api_key = st.secrets["OPENAI_API_KEY"]

show_token_usage = st.sidebar.checkbox("Show Token Usage", value=False)
close_warning = st.sidebar.checkbox("Close Warning", value=False)

st.title("WikiGPT: A Simple LLM Tutor")
if not close_warning:
    st.warning("This is a simple AI chatbot using OpenAI's API. Honestly make no guarantees around quality. YMMV.")

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

if "user_state" not in st.session_state:
    st.session_state.user_state = UserState()

    st.session_state.qa_generator_chain = build_qa_generator_chain(MODEL_NAME, TEMPERATURE, openai_api_key)
    st.session_state.tutor_chain = build_tutor_chain(MODEL_NAME, TEMPERATURE, openai_api_key)


def generate_wiki_response(input_text: str) -> Tuple[str, bool]:
    if not st.session_state.user_state.topic_chosen:
        validation_result, is_valid = validate_input(input_text)
        if not is_valid:
            return validation_result, False

        # if is_valid, validation_result is topic
        st.session_state.user_state.articles = search_wikipedia(validation_result)
        if len(st.session_state.user_state.articles) == 0:
            return "No search results found. Please select another topic.", False

        st.session_state.user_state.topic_chosen = True
        return "Please select a topic from the list below: \n" + ",\n".join(st.session_state.user_state.articles), False
    elif user_input not in st.session_state.user_state.articles:
        return "Please select a topic from the list below: \n" + ",\n".join(st.session_state.user_state.articles), False
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
    st.session_state.user_state.token_count += response['passed'].usage_metadata['total_tokens']
    return response['output']['questions']


def generate_tutor_response(content: str, questions: List[str]) -> BaseMessage:
    response = st.session_state.tutor_chain.invoke(
        {
            "content": content,
            "questions": questions,
            "conversations": st.session_state.user_state.tutor_conversation
        }
    )
    st.session_state.user_state.token_count += response.usage_metadata['total_tokens']
    return response


# 1) Fetch content from Wikipedia
# 2) Generate questions
# 3) Ask questions and grade them
with st.form("my_form"):
    user_input = st.text_area("Enter text:", "Topic: [topic name]")
    st.info(f"ℹ️ Powered by OpenAI's {MODEL_NAME} model")
    submitted = st.form_submit_button("Submit")

    if submitted and openai_api_key.startswith("sk-"):
        if st.session_state.user_state.content is None:
            wiki_response, content_retrieved = generate_wiki_response(user_input)
            if content_retrieved:
                st.session_state.user_state.topic = user_input
                st.session_state.user_state.content = wiki_response
            else:
                st.session_state.user_state.tutor_conversation = [("ai", wiki_response)]

        if st.session_state.user_state.content is not None:
            if st.session_state.user_state.questions is None:
                st.session_state.user_state.questions = generate_qa_response(st.session_state.user_state.content)
            # follow-ups
            if st.session_state.user_state.begun_tutoring:
                st.session_state.user_state.tutor_conversation.append(("human", user_input))
            # initial chat
            else:
                st.session_state.user_state.begun_tutoring = True
                st.session_state.user_state.tutor_conversation.append(("human", f"Topic: {st.session_state.user_state.topic}"))

            ai_msg = generate_tutor_response(st.session_state.user_state.content, st.session_state.user_state.questions)
            st.session_state.user_state.tutor_conversation.append(("ai", ai_msg.content))


conversation = [WELCOME_MESSAGE] + st.session_state.user_state.tutor_conversation
for msg in conversation[::-1]:
    if msg[0] == "human":
        st.info(f"👤 {msg[1]}")
    elif msg[0] == "ai":
        st.info(f"🤖 {msg[1]}")
if show_token_usage:
    st.info(f"Running token count: {st.session_state.user_state.token_count}")
