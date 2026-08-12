"""
The conversational agent.

Design: the LLM's ONLY job is (a) decide whether the user is asking for a
visual similarity search vs. just chatting, and (b) call the search tool
with a clean schema, and (c) narrate the results nicely. It never sees raw
image bytes -- the currently-uploaded image lives server-side in Streamlit
session state, and the tool is bound to it per-turn via a closure. This
mirrors how real production agents handle attachments (reference by id/
context, not by stuffing binary data through function-calling args) and
keeps the tool's input schema clean and LLM-friendly.

Tool schema
-----------
Input:  { top_k: int (1-10, default 5) }
Output: list of {name, price, image_url, product_link, score}

Swappable LLM: this file only touches Gemini. To use Groq or OpenAI
instead, replace `_build_llm()` -- everything else (tool, prompt, executor)
is provider-agnostic LangChain.
"""
from __future__ import annotations

import os
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from PIL import Image

from search_tool import search_similar_sarees
import config

SYSTEM_PROMPT = """You are TailorTalk, a friendly shopping assistant for a saree \
catalogue. You specialize in ONE thing: finding visually similar sarees when a \
user shares/uploads an image and asks to find similar items, matches, \
alternatives, or "more like this".

Rules:
- If the user has uploaded an image AND is asking (directly or implicitly) to \
find similar sarees, matches, or alternatives -- call the find_similar_sarees \
tool. Default top_k=5 unless the user asks for a specific number (max 10).
- If the user uploaded an image but is just asking a general question about it \
(e.g. "what color is this"), you may still offer to search for similar ones, \
but don't call the tool unless they want a search.
- If there's no image uploaded yet and the user wants a similarity search, ask \
them to upload one or paste an image link.
- After the tool returns results, do NOT re-list every field in prose (the UI \
already shows an image grid). Just give a short, natural one- or two-sentence \
summary of what you found (e.g. common fabric/colour theme across the matches).
- Stay in character as a saree shopping assistant; for unrelated requests, \
politely redirect.
"""


class FindSimilarInput(BaseModel):
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1,
        le=config.MAX_TOP_K,
        description="How many similar sarees to return, 1-10.",
    )


def _make_search_tool(current_image: Optional[Image.Image], on_results):
    """
    Build a StructuredTool bound to whichever image is currently active in
    the Streamlit session. `on_results` is a callback the app uses to grab
    the raw result list for rendering the image grid (the LLM only gets a
    compact text summary back, keeping token usage sane).
    """

    def _run(top_k: int = config.DEFAULT_TOP_K) -> str:
        if current_image is None:
            return "ERROR: no image is currently uploaded. Ask the user to upload one."
        results = search_similar_sarees(current_image, top_k=top_k)
        on_results(results)
        if not results:
            return "No similar sarees were found."
        lines = [
            f"- {r['name']} (score={r['score']}, price={r['price']})" for r in results
        ]
        return "Top matches:\n" + "\n".join(lines)

    return StructuredTool.from_function(
        func=_run,
        name="find_similar_sarees",
        description=(
            "Search the saree catalogue's vector index for items visually "
            "similar to the image the user just uploaded. Use this whenever "
            "the user wants a similarity/visual search, 'find matches', "
            "'more like this', etc. Takes only top_k; the query image is "
            "already known from context."
        ),
        args_schema=FindSimilarInput,
    )


def _build_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and set it as an env var "
            "(locally) or a Streamlit secret (when deployed)."
        )
    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0.3, google_api_key=api_key)


def build_agent_executor(current_image: Optional[Image.Image], on_results) -> AgentExecutor:
    llm = _build_llm()
    tool = _make_search_tool(current_image, on_results)

    img_ctx = (
        "Current session state: An image is currently active and attached in the sidebar. "
        "If the user asks to find matches, similar sarees, or mentions their uploaded/linked image, call the find_similar_sarees tool."
        if current_image is not None
        else "Current session state: No image is currently uploaded. Ask the user to upload or link an image if they want a similarity search."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"{SYSTEM_PROMPT}\n\n{img_ctx}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=False, handle_parsing_errors=True)
