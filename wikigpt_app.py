import logging
from textwrap import dedent
from typing import List

import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from prompts import qa_generator_template, tutor_template
from wiki import WikipediaSearchTool, WikipediaFetchTool, WikipediaChain, FAKE_WIKI_CONTENT

openai_api_key = st.secrets["OPENAI_API_KEY"]

show_system_prompt = st.sidebar.radio("Show System Prompt?", ("No", "Yes"))
close_warning = st.sidebar.checkbox("Close Warning", value=False)

st.title("WikiGPT: A Simple LLM Tutor")
if not close_warning:
    st.warning("This is a simple AI chatbot using OpenAI's API. Honestly make no guarantees around quality. YMMV.")

TEMPERATURE = 0.7
MODEL_NAME = "gpt-4o"

if "topic_chosen" not in st.session_state:
    st.session_state.topic_chosen = False

WELCOME_MESSAGE = (
        "ai",
        dedent(
            """
            Hello! I am an AI chatbot tutor. What topic would you like to learn? I will ask you a series
            of five questions and then grade you on your responses.
    
            Please provide the name of the topic you would like to learn using the format: Topic: [topic name]
            """)
    )

if "tutor_conversation" not in st.session_state:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)
    st.session_state.qa_generator_chain = qa_generator_template | llm | JsonOutputParser()
    st.session_state.tutor_chain = tutor_template | llm

    st.session_state.tutor_conversation = []
    st.session_state.questions = None
    st.session_state.content = None

    st.session_state.counter = 0

    # search_tool = WikipediaSearchTool()
    # fetch_tool = WikipediaFetchTool()
    # tools = [search_tool, fetch_tool]
    # st.session_state.wiki_agent = initialize_agent(
    #     tools,
    #     st.session_state.llm,
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )
    # st.session_state.wikipediapipeline = WikipediaChain(st.session_state.wiki_agent)


def generate_wiki_response(input_text: str) -> str:
    # TODO fake for now
    return "The importance of renewable energy sources like solar and wind power is increasing as the world seeks to reduce its carbon footprint and combat climate change. Renewable energy sources are not only environmentally friendly but also provide a sustainable alternative to fossil fuels."


def generate_qa_response(content: str, num_questions=5) -> List[str]:
    response = st.session_state.qa_generator_chain.invoke(
        {
            "content": content,
            "num_questions": num_questions
        }
    )
    return response["questions"]


def generate_tutor_response(content: str, questions: List[str]) -> BaseMessage:
    return st.session_state.tutor_chain.invoke(
        {
            "content": content,
            "questions": questions,
            "conversations": st.session_state.tutor_conversation
        }
    )


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
        st.session_state.content = generate_wiki_response(user_input)
        if st.session_state.questions is None:
            st.session_state.questions = generate_qa_response(st.session_state.content)
        if len(st.session_state.tutor_conversation) > 0:
            st.session_state.tutor_conversation.append(("human", user_input))
        # TODO hard-coded for now
        if len(st.session_state.tutor_conversation) == 0:
            st.session_state.tutor_conversation.append(("human", "Topic: Renewable energy sources"))

        ai_msg = generate_tutor_response(st.session_state.content, st.session_state.questions)
        st.session_state.tutor_conversation.append(("ai", ai_msg.content))


conversation = [WELCOME_MESSAGE] + st.session_state.tutor_conversation
for msg in conversation[::-1]:
    if msg[0] == "human":
        st.info(f"👤 {msg[1]}")
    elif msg[0] == "ai":
        st.info(f"🤖 {msg[1]}")


st.write(f"Debugging! Tutor Convo: {st.session_state.tutor_conversation}")
