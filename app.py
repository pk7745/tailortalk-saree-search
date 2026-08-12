import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

import config
from agent import build_agent_executor
from search_tool import index_size, load_image_from_bytes, load_image_from_url

st.set_page_config(page_title="TailorTalk - Saree Visual Search", page_icon="🥻", layout="wide")

# Pull the free Gemini key from Streamlit secrets if present (deployed) and
# expose it as an env var for langchain_google_genai to pick up.
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("🥻 TailorTalk")
st.caption(
    "Upload a saree photo (or paste an image link) and chat naturally -- "
    "e.g. *\"find me sarees similar to this\"*."
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[HumanMessage | AIMessage]
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# ---------------------------------------------------------------------
# Sidebar: image input + catalogue stats
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Query image")
    uploaded = st.file_uploader("Upload a saree image", type=["jpg", "jpeg", "png", "webp"])
    url_input = st.text_input("...or paste an image URL")

    if uploaded is not None:
        st.session_state.current_image = load_image_from_bytes(uploaded.read())
    elif url_input:
        try:
            st.session_state.current_image = load_image_from_url(url_input)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't load that URL: {e}")

    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="Current query image", use_column_width=True)
        if st.button("Clear image"):
            st.session_state.current_image = None
            st.rerun()

    st.divider()
    try:
        st.metric("Sarees indexed", index_size())
    except Exception:
        st.warning(
            "Index not found yet. Run `python ingest.py` locally to build "
            "data/saree_index.faiss before deploying."
        )

# ---------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if st.session_state.last_results:
    st.subheader("Matches")
    cols = st.columns(min(5, len(st.session_state.last_results)))
    for i, r in enumerate(st.session_state.last_results):
        with cols[i % len(cols)]:
            st.image(r["image_url"], use_column_width=True)
            st.markdown(f"**{r['name']}**")
            st.caption(f"Similarity: {r['score']:.2f} · ₹{r['price']}")
            if r.get("product_link"):
                st.markdown(f"[View product]({r['product_link']})")

# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask me to find similar sarees...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    captured_results = []

    def _capture(results):
        captured_results.extend(results)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                executor = build_agent_executor(st.session_state.current_image, _capture)
                response = executor.invoke(
                    {"input": user_input, "chat_history": st.session_state.chat_history[:-1]}
                )
                answer = response["output"]
            except Exception as e:  # noqa: BLE001
                answer = f"Something went wrong: {e}"
        st.markdown(answer)

    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.last_results = captured_results
    if captured_results:
        st.rerun()
