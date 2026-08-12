"""
Conversational agent for TailorTalk Saree Search.

Combines:
- Multi-modal visual similarity (OpenCLIP + Color Histogram + FAISS)
- Deterministic attribute & budget filtering (max_price, min_price, color, fabric)
- Natural language saree stylist consultation with strict groundedness and conversational memory
"""
from __future__ import annotations

from typing import Callable, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

import config
from search_tool import search_similar_sarees

SYSTEM_PROMPT = """You are TailorTalk's expert Saree Stylist and Personal Shopper.

Your goal is to help users find sarees matching their exact visual and stylistic preferences from our verified catalogue.

Active Context: An image query is provided in the current session.
When the user asks for similar sarees (e.g. 'find me sarees like this', 'show similar ones', 'find sarees'), or specifies constraints (colors, fabrics, price):
1. Extract any budget constraints (e.g. 'under 3000' -> max_price=3000, 'above 2000' -> min_price=2000).
2. Extract any color preferences (e.g. 'pink', 'red', 'black', 'navy blue', 'green') -> color.
3. Extract any fabric preferences (e.g. 'banarasi', 'organza', 'pashmina', 'linen', 'satin', 'munga') -> fabric.
4. Extract the requested number of items (default 5, up to 20) -> top_k.
5. ALWAYS call `find_similar_sarees` with all extracted filter parameters.

Conversational Context & Pronoun References:
- When the user asks follow-up questions referencing previous results by position or pronouns (e.g., 'what's the price of the second one?', 'is that one machine washable?', 'tell me more about the first saree'), use the sarees listed in the conversation history to identify the exact item.
- Answer questions about a specific saree (price, fabric, material, saree length, blouse included/length, wash care instructions, etc.) truthfully based ONLY on the metadata present in the conversation/tool results.
- If a detail is missing/null in our catalogue records (e.g. occasion or care instructions when not recorded), state that this detail is not available in our verified catalogue records for this saree, rather than guessing.

For general chit-chat (e.g. 'hi', 'how are you?') without saree requests, respond politely without calling tools.
"""


class FindSimilarInput(BaseModel):
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        description="Number of similar sarees to return (between 1 and 20).",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum price budget in INR (e.g. 3000 for 'under ₹3000').",
    )
    min_price: Optional[float] = Field(
        default=None,
        description="Minimum price in INR (e.g. 2000 for 'above ₹2000').",
    )
    color: Optional[str] = Field(
        default=None,
        description="Target color filter (e.g. 'pink', 'red', 'black', 'blue', 'green', 'yellow', 'gold').",
    )
    fabric: Optional[str] = Field(
        default=None,
        description="Target fabric weave filter (e.g. 'banarasi', 'organza', 'pashmina', 'linen', 'satin', 'munga', 'silk').",
    )


def make_search_tool(
    query_image: Optional[Image.Image],
    on_results: Optional[Callable[[list[dict]], None]] = None,
):
    @tool("find_similar_sarees", args_schema=FindSimilarInput)
    def find_similar_sarees_tool(
        top_k: int = config.DEFAULT_TOP_K,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        color: Optional[str] = None,
        fabric: Optional[str] = None,
    ) -> list[dict]:
        """Search the catalogue for sarees matching the uploaded image and/or specified color, fabric, and price constraints."""
        results = search_similar_sarees(
            query_image=query_image,
            top_k=top_k,
            max_price=max_price,
            min_price=min_price,
            color=color,
            fabric=fabric,
        )
        if on_results is not None:
            on_results(results)
        return results

    return find_similar_sarees_tool


def build_agent_executor(
    query_image: Optional[Image.Image],
    on_results: Optional[Callable[[list[dict]], None]] = None,
) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=0.1,
    )
    tools = [make_search_tool(query_image, on_results)]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
