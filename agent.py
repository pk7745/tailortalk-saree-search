"""
The conversational agent for TailorTalk Saree Visual Search.

Equipped with:
1. Multi-modal similarity search (FAISS + CLIP + Color Histogram)
2. Fine-grained multi-attribute filtering (Colors, Fabrics, Patterns, Price, Keywords)
3. Direct catalogue discovery & querying
"""
from __future__ import annotations

import os
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from PIL import Image

from search_tool import search_sarees, search_similar_sarees
import config

SYSTEM_PROMPT = """You are TailorTalk's expert AI Saree Stylist & Luxury Catalogue Consultant. \
You provide accurate visual similarity search, fine-grained attribute filtering (color, fabric, pattern, price), \
and authentic styling advice across a curated catalogue of 1,070 authentic sarees.

Key Rules & Behaviors:
1. Extract & Apply All Filters Accurately:
   - When the user asks for colors (e.g. "show me only pink colour", "blue sarees", "green with gold"), pass `color` to the search tool.
   - When the user asks for fabrics (e.g. "Banarasi", "Organza", "Ajrakh", "Pashmina", "Linen", "Satin", "Tussar", "Munga Crape", "Silk", "Cotton"), pass `fabric`.
   - When the user specifies budget (e.g. "under 3000", "below 5000"), pass `max_price` and/or `min_price`.
   - When the user asks for patterns/work (e.g. "zari border", "lotus print", "madhubani", "checks", "applique work"), pass `pattern`.
   - Respect user requests for count (e.g. "show me 8 instead") via `top_k`.
2. Exact Truthfulness & Grounding:
   - Only speak to items and attributes returned by the search tool. Never invent false products or fake prices.
   - Provide a natural, polished 1-2 sentence stylist summary highlighting the matched colors, fabric weave, and craftwork.
3. Conversational Context:
   - When an image is active in session, searching seamlessly blends visual similarity with any requested color/fabric/budget filters.
   - If no image is uploaded and user asks for sarees by description, search the catalogue directly with the tool.
   - For general chit-chat (e.g. "hi", "who are you"), respond courteously in character as TailorTalk's saree shopping assistant without invoking the tool.
"""


class FindSimilarInput(BaseModel):
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="How many sarees to return (1-20).",
    )
    color: Optional[str] = Field(
        default=None,
        description="Filter by color (e.g. 'pink', 'navy blue', 'green', 'yellow', 'red', 'black', 'white', 'maroon', 'purple', etc.).",
    )
    fabric: Optional[str] = Field(
        default=None,
        description="Filter by fabric (e.g. 'banarasi', 'organza', 'ajrakh', 'pashmina', 'linen', 'satin', 'tussar', 'munga crape', 'silk', 'cotton', etc.).",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Filter by pattern or work (e.g. 'zari border', 'lotus print', 'madhubani', 'applique work', 'checks', 'printed', etc.).",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum price budget in INR (e.g. 3000, 5000).",
    )
    min_price: Optional[float] = Field(
        default=None,
        description="Minimum price in INR.",
    )
    keyword: Optional[str] = Field(
        default=None,
        description="Specific style or title keyword.",
    )


def _make_search_tool(current_image: Optional[Image.Image], on_results):
    """
    Build a StructuredTool bound to the active session image and result callback.
    """

    def _run(
        top_k: int = config.DEFAULT_TOP_K,
        color: Optional[str] = None,
        fabric: Optional[str] = None,
        pattern: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        keyword: Optional[str] = None,
    ) -> str:
        results = search_sarees(
            query_image=current_image,
            color=color,
            fabric=fabric,
            pattern=pattern,
            min_price=min_price,
            max_price=max_price,
            keyword=keyword,
            top_k=top_k,
        )
        on_results(results)
        if not results:
            active_filters = []
            if color:
                active_filters.append(f"color='{color}'")
            if fabric:
                active_filters.append(f"fabric='{fabric}'")
            if max_price:
                active_filters.append(f"max_price=₹{max_price}")
            filter_str = ", ".join(active_filters) if active_filters else "requested criteria"
            return f"No sarees found matching {filter_str}. Try broadening your search filters."

        lines = [
            f"- {r['name']} | Similarity: {r['score']} | Fabric: {r['fabric']} | Color: {r['color']} | Price: ₹{r['price']} | Link: {r['product_link']}"
            for r in results
        ]
        return f"Retrieved {len(results)} matching sarees:\n" + "\n".join(lines)

    return StructuredTool.from_function(
        func=_run,
        name="find_similar_sarees",
        description=(
            "Search and filter the 1,070-item saree catalogue. "
            "Supports visual similarity (using the active image), color filtering ('pink', 'blue', etc.), "
            "fabric filtering ('banarasi', 'organza', 'linen', etc.), pattern filtering ('zari border', 'lotus print', etc.), "
            "and price budget filtering (min_price/max_price). "
            "Use this whenever the user asks for similar sarees, color/fabric filters, budget options, or catalogue search."
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
    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0.2, google_api_key=api_key)


def build_agent_executor(current_image: Optional[Image.Image], on_results) -> AgentExecutor:
    llm = _build_llm()
    tool = _make_search_tool(current_image, on_results)

    img_ctx = (
        "Current session state: An image is currently active and attached in the sidebar. "
        "When the user asks for matches, similar sarees, or filtered sarees (e.g. 'show only pink colour'), "
        "call find_similar_sarees with appropriate color/fabric/price filters."
        if current_image is not None
        else "Current session state: No image is currently attached. If the user asks for sarees by text/color/fabric/budget, "
        "call find_similar_sarees to search the catalogue directly. If they want pure visual similarity, invite them to upload an image."
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
