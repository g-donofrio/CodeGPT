from engine import SingleModelCode
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")
REPO_NAME = os.environ.get("REPO_NAME")
REPO_DEST = os.environ.get("REPO_DEST")
CHAT_TITLE = os.environ.get("CHAT_TITLE")
DB_PERSIST_PATH = os.environ.get("DB_PERSIST_PATH")


# initialize Chat Engine
@st.cache_resource(show_spinner=False) 
def get_engine(openai_api_key, repo_url, repo_path, repo_dest, db_persist):
    return SingleModelCode(openai_api_key, repo_url, repo_path, repo_dest, db_persist)


with st.status(
    "Loading engine...",
    expanded=st.session_state.get("expanded", True),
) as status:
    engine = get_engine(OPENAI_API_KEY, REPO_URL, REPO_PATH, REPO_DEST, DB_PERSIST_PATH)
    status.update(label="Engine loaded.", state="complete", expanded=False)
st.session_state["expanded"] = False

st.header(CHAT_TITLE, divider="rainbow")

# Create a text input box for the user
prompt = st.text_input(f"Use this to search answers for {REPO_NAME}")

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    with st.spinner("Waiting"):
        result = engine(prompt)
    # ...and write it out to the screen
    st.write(result["answer"])

    st.write("Sources:")
    for source in result["sources"]:
        with st.expander(source["metadata"]["source"]):
            st.text(source["content"])