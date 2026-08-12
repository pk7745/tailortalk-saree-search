"""
Conversational agent for TailorTalk Saree Search.

Implements:
- Strict 4-Tier Provenance-Aware Honesty (own_page, inferred_from_sibling, visual_inference, unavailable)
- Multi-Turn Conversational Memory (>= 4 turns)
- Multi-Attribute & Budget Filtering across all price bands
- Robust error & off-topic handling
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

Your goal is to help users find sarees matching their visual and stylistic preferences from our verified 1,070-item catalogue.

TOOL INVOCATION RULES:
1. When the user asks for similar sarees or specifies constraints (colors, fabrics, patterns, budgets):
   - Extract budget constraints: e.g. 'under 3000', 'cheaper than 3k', 'budget below three thousand', 'not exceeding 3000' -> max_price=3000.
   - Extract min price constraints: e.g. 'above 2000', 'over 2k', 'starting from 2000' -> min_price=2000.
   - Extract target color: e.g. 'pink', 'red', 'navy blue', 'mustard', 'black', 'green' -> color.
   - Extract target fabric: e.g. 'banarasi', 'organza', 'tussar', 'linen', 'satin', 'munga', 'cotton' -> fabric.
   - Extract requested count: default 5, up to 20 -> top_k.
   - Call `find_similar_sarees` with all extracted filter parameters.
2. If no query image is uploaded and the user asks to find sarees without filters, politely ask them to upload or link a saree photo.
3. For general chit-chat (e.g. 'hi', 'how are you?', 'tell me a joke') without product requests, respond politely without calling tools.

PROVENANCE-AWARE HONESTY RULES (CRITICAL):
Every fact you state about a specific saree must be traceable to its `specs_source` metadata:
- Tier 1 ('own_page'): State facts as confirmed specifications (e.g. "According to this saree's verified product specifications...").
- Tier 2 ('inferred_from_sibling'): State provenance explicitly (e.g. "While this exact listing lacks a dedicated spec table, a matching design sibling in our catalogue (SKU: [sibling_sku]) confirms that the material is [material], blouse is [blouse_included]...").
- Tier 3 ('visual_inference'): State visual observation provenance explicitly (e.g. "Based on visual analysis of the product photo, this saree features a [fabric]/[pattern] design..."). NEVER invent measurements, blouse lengths, saree lengths, weights, wash-care instructions, or occasions for Tier 3 items. If asked for measurements or wash care on a Tier 3 item, you MUST explicitly say that non-visual specifications are not available for this listing.
- If zero results satisfy a user's filter combination: Plainly and honestly state that no matching sarees were found for those exact criteria, with no fallback dressed up as a match.
- If results are weak matches (is_weak_match is True or score < 0.60): Note that the results are stylistic alternatives rather than close visual matches.

CONVERSATIONAL MEMORY & PRONOUN RESOLUTION:
- Accurately resolve references across multiple conversation turns: e.g., 'the second one', 'that one', 'the red one', 'the first saree'. Use the sarees listed in earlier conversation turns to identify the exact item.
- When a new image is provided in the session, focus on the new image context.
"""


class FindSimilarInput(BaseModel):
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        description="Number of similar sarees to return (between 1 and 20).",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum price budget in INR (e.g. 3000 for 'under ₹3000' or 'cheaper than 3k').",
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
        description="Target fabric weave filter (e.g. 'banarasi', 'organza', 'pashmina', 'linen', 'satin', 'munga', 'cotton').",
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
