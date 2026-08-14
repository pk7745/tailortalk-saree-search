"""
Conversational agent for TailorTalk Saree Search.

Implements:
- Strict 4-Tier Provenance-Aware Honesty (own_page, inferred_from_sibling, visual_inference, unavailable)
- Multi-Turn Conversational Memory (>= 4 turns)
- Multi-Attribute & Budget Filtering across all price bands
- Robust error & off-topic handling
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

import config
from search_tool import search_similar_sarees
from web_verifier import fetch_official_product_details

BASE_SYSTEM_PROMPT = """You are TailorTalk's expert Saree Stylist and Personal Shopper.

Your goal is to help users find sarees matching their visual and stylistic preferences from our verified 1,070-item catalogue.

{image_state_instruction}

STRICT EVIDENCE-BASED ANSWERING & SOURCE ROUTING RULES:
1. GEMINI IS AN ORCHESTRATION & REASONING LAYER, NOT THE SOURCE OF TRUTH FOR PRODUCT FACTS:
   - Always derive product-specific facts from authoritative retrieved evidence.
   - Use this source priority for product attributes:
     * Current Price & Availability -> Official Webpage (`fetch_official_product_details`), then Catalogue Metadata
     * Fabric & Detailed Specifications -> Official Webpage (`fetch_official_product_details`), then Catalogue Metadata
     * Product SKU & Identity -> Catalogue Metadata / Official Webpage
     * Visual Similarity & Vector Ranking -> Qdrant (1024d fused embeddings)
     * Observable Color & Pattern -> Image & Search Representation

2. TOOL INVOCATION RULES:
   - Call `find_similar_sarees` whenever searching the catalogue, applying filters (colors, fabrics, patterns, budgets), or requesting similar sarees.
   - Call `fetch_official_product_details` whenever verifying live product details, exact price, availability, or specifications for a specific saree URL (`product_link`).

3. CONTEXTUAL REFERENCE RESOLUTION & MULTI-TURN MEMORY:
   - Maintain active product identity across conversation turns.
   - Resolve references like "the first one", "second saree", "that red one", "is it available?", "how much is it?" against the latest displayed search results. Do NOT perform an unrelated new search when answering follow-up questions about a previously displayed product.

4. EXACT CONCISE ANSWERS (NO EXTRA UNREQUESTED FLUFF):
   - Answer the user's specific question directly, concisely, and accurately.
   - Do NOT dump unrequested extra specifications. If asked about fabric, state ONLY the fabric. If asked about price, state ONLY the price.
   - Provide full specifications ONLY when explicitly requested.
   - When providing store links, use the original website link (`product_link`) as a markdown hyperlink (e.g. "[View on Original Store ↗](product_link)").

5. DISCREPANCY & CONFLICT RESOLUTION:
   - If catalogue metadata and official webpage disagree (e.g. Catalogue lists ₹3,150 but Official Webpage lists ₹3,499), state both explicitly:
     "The catalogue lists ₹3,150, while the official product page currently lists ₹3,499."

6. STRICT FALLBACK FOR UNVERIFIED & OUT-OF-SCOPE QUESTIONS:
   - If an attribute is not present in catalogue metadata or official webpage evidence, state plainly:
     "I couldn't verify the [attribute] from the available product information."
   - Never infer non-visible specs (fabric weave, weight, wash care, price) solely from visual appearance.
   - If a prompt is completely outside saree shopping or catalogue lookup, state:
     "I apologize, but I cannot process or provide information for that request. I am specialized only in assisting you with finding and answering questions about sarees from our 1,070-item catalogue."
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
    pattern: Optional[str] = Field(
        default=None,
        description="Target pattern, border, pallu, or work type filter (e.g. 'zari border', 'golden zari', 'temple border', 'kadiyal border', 'contrast border', 'floral', 'embroidery', 'applique work', 'geometric zari', 'pallu').",
    )


class WebVerifierInput(BaseModel):
    url: str = Field(
        description="The official product webpage URL (product_link) to verify details from."
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
        pattern: Optional[str] = None,
    ) -> list[dict]:
        """Search the catalogue for sarees matching the uploaded image and/or specified color, fabric, pattern/border/work, and price constraints."""
        results = search_similar_sarees(
            query_image=query_image,
            top_k=top_k,
            max_price=max_price,
            min_price=min_price,
            color=color,
            fabric=fabric,
            pattern=pattern,
        )
        if on_results is not None:
            on_results(results)
        return results

    return find_similar_sarees_tool


def make_web_verifier_tool():
    @tool("fetch_official_product_details", args_schema=WebVerifierInput)
    def fetch_official_product_details_tool(url: str) -> dict:
        """Fetches live verified product specifications, exact price, availability, and details directly from the official merchant product page."""
        return fetch_official_product_details(url)

    return fetch_official_product_details_tool


@lru_cache(maxsize=1)
def _get_llm():
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=0.1,
        max_retries=6,
    )


def build_agent_executor(
    query_image: Optional[Image.Image],
    on_results: Optional[Callable[[list[dict]], None]] = None,
) -> AgentExecutor:
    if query_image is not None:
        image_instruction = (
            "ACTIVE QUERY IMAGE STATUS: The user has uploaded an active query image for this session. "
            "Whenever the user asks for similar sarees, matching sarees, or applies filters, "
            "ALWAYS immediately call the `find_similar_sarees` tool."
        )
    else:
        image_instruction = (
            "ACTIVE QUERY IMAGE STATUS: No query image is currently uploaded. "
            "If the user asks for visually similar sarees without any filters, politely ask them to upload or link a saree photo. "
            "If they specify text filters (e.g. 'find pink banarasi sarees under 3000'), search the catalogue by calling `find_similar_sarees`."
        )

    system_prompt = BASE_SYSTEM_PROMPT.format(image_state_instruction=image_instruction)

    llm = _get_llm()
    tools = [make_search_tool(query_image, on_results), make_web_verifier_tool()]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

