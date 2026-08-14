import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

import config
from agent import build_agent_executor
from search_tool import index_size, load_image_from_bytes, load_image_from_url

st.set_page_config(
    page_title="TailorTalk - Saree Visual & Attribute Search",
    page_icon="🥻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
<style>
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #1E1E2F;
        margin-bottom: 0px;
    }
    .saree-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #EDEDF2;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .saree-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .tag-badge {
        display: inline-block;
        background-color: #F0F2F6;
        color: #31333F;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .price-badge {
        font-size: 16px;
        font-weight: 700;
        color: #D9383A;
    }
    .score-badge {
        font-size: 11px;
        font-weight: 700;
        color: #0E7090;
        background-color: #E0F2FE;
        padding: 2px 8px;
        border-radius: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Pull the free Gemini key from Streamlit secrets if present (deployed) and
# expose it as an env var for langchain_google_genai to pick up.
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

st.title("🥻 TailorTalk Saree Search")
st.caption(
    "Search across **1,070 authentic sarees** with multi-modal visual similarity, "
    "fine-grained color & fabric filtering, and natural shopping consultation."
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # list[dict] with role, content, results
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[HumanMessage | AIMessage] for LangChain
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# ---------------------------------------------------------------------
# Sidebar: Image input + Catalogue breakdown
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📸 Query Image")
    uploaded = st.file_uploader("Upload a saree photo", type=["jpg", "jpeg", "png", "webp"])
    url_input = st.text_input("...or paste an image URL")

    if uploaded is not None:
        st.session_state.current_image = load_image_from_bytes(uploaded.read())
    elif url_input:
        try:
            st.session_state.current_image = load_image_from_url(url_input)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't load that URL: {e}")

    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="Active Query Image", use_column_width=True)
        if st.button("🗑️ Clear Query Image"):
            st.session_state.current_image = None
            st.session_state.last_results = []
            st.rerun()

    st.divider()
    st.subheader("📊 Catalogue Insights")
    try:
        total = index_size()
        st.metric("Total Indexed Sarees", f"{total:,}")
    except Exception:
        st.warning("Index not found yet. Run ingest pipeline locally.")

    with st.expander("🧵 Available Fabrics", expanded=False):
        st.markdown(
            "- **Banarasi / Pashmina**\n"
            "- **Pure Organza & Tissue**\n"
            "- **Ajrakh Handblock Prints**\n"
            "- **Linen Silk & Cotton**\n"
            "- **Satin & Munga Crape**\n"
            "- **Tussar & Mysore Silk**"
        )

    with st.expander("🎨 Popular Color Palettes", expanded=False):
        st.markdown(
            "- **Pinks**: Rani, Baby, Peach, Dusty\n"
            "- **Blues**: Navy, Sky, Royal Blue\n"
            "- **Greens**: Mint, Bottle, Pista Green\n"
            "- **Yellows & Gold**: Mustard, Lemon\n"
            "- **Reds & Maroons**: Crimson, Wine\n"
            "- **Neutrals**: White, Cream, Black, Silver"
        )

# ---------------------------------------------------------------------
# Quick suggestions
# ---------------------------------------------------------------------
st.markdown("**Quick Prompts:**")
col1, col2, col3, col4 = st.columns(4)
quick_prompt = None
with col1:
    if st.button("🔍 Find Similar Sarees", use_container_width=True):
        quick_prompt = "Find similar sarees to this one"
with col2:
    if st.button("🌸 Show Only Pink Sarees", use_container_width=True):
        quick_prompt = "Show me only pink colour sarees"
with col3:
    if st.button("🧵 Banarasi under ₹4,000", use_container_width=True):
        quick_prompt = "Find Banarasi sarees under ₹4,000"
with col4:
    if st.button("✨ Show 8 Matches", use_container_width=True):
        quick_prompt = "Show me 8 similar sarees"

# ---------------------------------------------------------------------
# Helper function to render a product card grid
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
                st.markdown(f"**₹{r['price']}** · 🎯 `{sim_score:.2f} match`")
            else:
                st.markdown(f"**₹{r['price']}**")

            if r.get("product_link"):
                st.markdown(f"[View on Store ↗]({r['product_link']})")

            if r.get("specs_source") == "own_page" or r.get("web_verified"):
                st.caption("✓ Verified from official page")



# ---------------------------------------------------------------------
# Persistent Chat History Render (with stored product cards per turn)
# ---------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("results"):
            render_saree_cards(msg["results"])

# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask about colors, fabrics, patterns, prices, or finding matches...")
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
                executor = build_agent_executor(st.session_state.current_image, _capture)
                response = executor.invoke(
                    {"input": actual_input, "chat_history": st.session_state.chat_history[:-1]}
                )
                answer = response["output"]
            except Exception as e:  # noqa: BLE001
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

