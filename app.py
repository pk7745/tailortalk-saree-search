import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

import config
from agent import build_agent_executor
from search_tool import index_size, load_image_from_bytes, load_image_from_url

st.set_page_config(
    page_title="TailorTalk — AI-Powered Saree Discovery",
    page_icon="🥻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom Premium Fashion-Oriented CSS Design System
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #2D3748;
    }
    
    .stApp {
        background-color: #FAF9F6;
    }

    .brand-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A202C;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        color: #8B0032;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .brand-desc {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 24px;
    }

    .welcome-card {
        background: linear-gradient(135deg, #FFF8FA 0%, #F5F7FA 100%);
        border: 1px solid #F1E2E7;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(139,0,50,0.03);
    }

    .welcome-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #1A202C;
        margin-bottom: 8px;
    }

    .welcome-text {
        font-size: 0.95rem;
        color: #64748B;
        max-width: 560px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }

    .saree-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 18px;
        transition: all 0.2s ease-in-out;
    }

    .saree-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border-color: #CBD5E1;
    }

    .verified-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: #059669;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 2px 8px;
        border-radius: 6px;
        margin-top: 6px;
    }

    .price-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #8B0032;
    }

    .match-score {
        font-size: 0.75rem;
        font-weight: 700;
        color: #0284C7;
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        padding: 2px 6px;
        border-radius: 6px;
    }

    /* Style Streamlit Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Pull API key from Streamlit secrets if running deployed
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource(show_spinner=False)
def _warmup_backend_engine():
    try:
        from embeddings import _lazy_load_clip
        from search_tool import _load_index_and_meta

        _lazy_load_clip()
        _load_index_and_meta()
    except Exception:
        pass


_warmup_backend_engine()

# ---------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# ---------------------------------------------------------------------
# Header & Branding Section
# ---------------------------------------------------------------------
st.markdown(
    """
<div style="text-align: center; padding: 10px 0 20px 0;">
    <div class="brand-title">TailorTalk</div>
    <div class="brand-subtitle">AI-Powered Saree Discovery</div>
    <div class="brand-desc">Find visually similar sarees and get accurate product information using text or images.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar: Image input + Search Guide
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📸 Visual Search")
    uploaded = st.file_uploader("Upload a saree image to find visually similar products", type=["jpg", "jpeg", "png", "webp"])
    url_input = st.text_input("...or paste an image URL")

    if uploaded is not None:
        st.session_state.current_image = load_image_from_bytes(uploaded.read())
    elif url_input:
        try:
            st.session_state.current_image = load_image_from_url(url_input)
        except Exception as e:
            st.error(f"Couldn't load image from URL: {e}")

    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="Active Reference Image", use_column_width=True)
        if st.button("🗑️ Clear Reference Image", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.last_results = []
            st.rerun()

    st.divider()

    st.markdown("### 💡 How to Search")
    st.markdown(
        """
        - **Text Search**: Ask about colours, fabrics, designs, prices, borders, or pallu.
        - **Visual Search**: Upload a photo to find visually matching sarees.
        - **Combined Search**: Upload an image and add text filters (e.g. *"under ₹5,000"*).
        - **Live Verification**: Ask about price or availability to fetch live merchant webpage evidence.
        """
    )

    st.divider()

    try:
        total = index_size()
        st.caption(f"**Verified Catalogue Size**: {total:,} Sarees")
        st.caption("**Vector Engine**: Qdrant (1024d Fused Embedding)")
    except Exception:
        pass

# ---------------------------------------------------------------------
# Welcome Hero (Displayed when chat is empty)
# ---------------------------------------------------------------------
quick_prompt = None

if not st.session_state.display_messages:
    st.markdown(
        """
    <div class="welcome-card">
        <div class="welcome-title">Discover your perfect saree</div>
        <div class="welcome-text">Search by description or upload an image to find visually similar sarees from our verified catalogue.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<p style='text-align:center; font-weight:600; color:#64748B; font-size:0.88rem; margin-bottom:12px;'>TRY EXAMPLE SEARCHES:</p>", unsafe_allow_html=True)
    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        if st.button("🧵 Show silk sarees under ₹5,000", use_container_width=True):
            quick_prompt = "Show silk sarees under ₹5,000"
    with e_col2:
        if st.button("✨ Find red sarees with zari border", use_container_width=True):
            quick_prompt = "Find red sarees with zari border"
    with e_col3:
        if st.button("🌸 Find pink banarasi sarees", use_container_width=True):
            quick_prompt = "Find pink banarasi sarees"

# ---------------------------------------------------------------------
# Helper functions to deduplicate and render product card grid
# ---------------------------------------------------------------------
def _get_product_id(r: dict) -> str:
    """Returns stable unique product identifier for deduplication."""
    return r.get("product_link") or r.get("image_url") or r.get("sku") or r.get("name", "")


def _deduplicate_saree_results(results: list[dict]) -> list[dict]:
    """Deduplicates product list by stable product ID, preserving highest final_score."""
    if not results:
        return []
    unique_map = {}
    for item in results:
        pid = _get_product_id(item)
        if pid not in unique_map or item.get("final_score", 0) > unique_map[pid].get("final_score", 0):
            unique_map[pid] = item
    deduped = list(unique_map.values())
    deduped.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return deduped


def render_saree_cards(results: list[dict]):
    deduped = _deduplicate_saree_results(results)
    if not deduped:
        return

    st.markdown(f"#### Similar Sarees <span style='font-size:0.85rem; font-weight:500; color:#64748B;'>({len(deduped)} matches found)</span>", unsafe_allow_html=True)

    num_cols = min(4, max(1, len(deduped)))
    cols = st.columns(num_cols)
    for i, r in enumerate(deduped):
        with cols[i % num_cols]:
            st.image(r["image_url"], use_column_width=True)
            st.markdown(f"**{r['name']}**")

            badges = []
            if r.get("fabric") and r["fabric"] != "Silk Blend":
                badges.append(f"🧵 {r['fabric']}")
            if r.get("color") and r["color"] != "Multicolor":
                badges.append(f"🎨 {r['color']}")
            if r.get("pattern") and r["pattern"] != "Classic Pattern":
                badges.append(f"✨ {r['pattern']}")

            if badges:
                st.caption(" · ".join(badges))

            sim_score = r.get("score", 0.0)
            if sim_score > 0 and sim_score <= 1.0 and st.session_state.current_image is not None:
                st.markdown(f"<span class='price-text'>₹{r['price']}</span> · <span class='match-score'>🎯 {sim_score:.2f} match</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='price-text'>₹{r['price']}</span>", unsafe_allow_html=True)

            if r.get("product_link"):
                st.markdown(f"[View Product ↗]({r['product_link']})")

            if r.get("specs_source") == "own_page" or r.get("web_verified"):
                st.markdown("<div class='verified-badge'>✓ Verified from official product page</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Persistent Chat History Render
# ---------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("results"):
            render_saree_cards(msg["results"])

# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask about sarees, colours, fabrics, designs, prices, borders, pallu...")
actual_input = quick_prompt or user_input

if actual_input:
    # 1. Append human message
    st.session_state.display_messages.append({"role": "user", "content": actual_input, "results": None})
    st.session_state.chat_history.append(HumanMessage(content=actual_input))
    with st.chat_message("user"):
        st.markdown(actual_input)

    captured_results = []

    def _capture(results):
        for item in _deduplicate_saree_results(results):
            pid = _get_product_id(item)
            existing_ids = [_get_product_id(x) for x in captured_results]
            if pid not in existing_ids:
                captured_results.append(item)
            else:
                idx = existing_ids.index(pid)
                if item.get("final_score", 0) > captured_results[idx].get("final_score", 0):
                    captured_results[idx] = item

    with st.chat_message("assistant"):
        with st.spinner("Consulting TailorTalk catalogue..."):
            try:
                executor = build_agent_executor(
                    st.session_state.current_image, _capture, st.session_state.last_results
                )
                response = executor.invoke(
                    {"input": actual_input, "chat_history": st.session_state.chat_history[:-1]}
                )
                answer = response["output"]
            except Exception as e:
                from search_tool import parse_query_intent, search_similar_sarees

                intent = parse_query_intent(actual_input)
                fallback_results = search_similar_sarees(
                    query_image=st.session_state.current_image,
                    color=intent["color"],
                    fabric=intent["fabric"],
                    min_price=intent["min_price"],
                    max_price=intent["max_price"],
                    top_k=intent["top_k"],
                )
                _capture(fallback_results)
                if fallback_results:
                    clauses = []
                    if intent["color"]:
                        clauses.append(f"color: {intent['color'].title()}")
                    if intent["fabric"]:
                        clauses.append(f"fabric: {intent['fabric'].title()}")
                    if intent["max_price"]:
                        clauses.append(f"budget under ₹{int(intent['max_price']):,}")
                    if intent["min_price"]:
                        clauses.append(f"price above ₹{int(intent['min_price']):,}")
                    desc = " (" + ", ".join(clauses) + ")" if clauses else ""
                    if st.session_state.current_image is not None:
                        answer = f"Found {len(captured_results)} visually matching sarees{desc} from our catalogue."
                    else:
                        answer = f"Found {len(captured_results)} authentic sarees{desc} from our catalogue."
                else:
                    answer = "No matching sarees found for those specific filters in our 1,070-item catalogue. Try broadening your criteria."

        # Guaranteed auto-trigger for active images if tool was not called
        if st.session_state.current_image is not None and not captured_results:
            from search_tool import parse_query_intent, search_similar_sarees

            intent = parse_query_intent(actual_input)
            if any(k in actual_input.lower() for k in ["similar", "find", "show", "match", "saree", "like"]) or intent["color"] or intent["fabric"] or intent["max_price"]:
                auto_results = search_similar_sarees(
                    query_image=st.session_state.current_image,
                    color=intent["color"],
                    fabric=intent["fabric"],
                    min_price=intent["min_price"],
                    max_price=intent["max_price"],
                    top_k=intent["top_k"],
                )
                _capture(auto_results)

        st.markdown(answer)
        if captured_results:
            render_saree_cards(captured_results)

    # 2. Append assistant message with its permanently attached results
    final_unique_results = _deduplicate_saree_results(captured_results)
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "results": final_unique_results if final_unique_results else None}
    )
    st.session_state.chat_history.append(AIMessage(content=answer))
    if final_unique_results:
        st.session_state.last_results = final_unique_results
    st.rerun()
