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

st.title("🥻 TailorTalk Saree Search")
st.caption(
    "Search across **1,070 authentic sarees** with multi-modal visual similarity, "
    "fine-grained color & fabric filtering, and natural shopping consultation."
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
# Chat history render
# ---------------------------------------------------------------------
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------------------------------------------------------------------
# Results Grid Render
# ---------------------------------------------------------------------
if st.session_state.last_results:
    st.subheader(f"✨ Matching Sarees ({len(st.session_state.last_results)} results)")
    num_cols = min(5, max(1, len(st.session_state.last_results)))
    cols = st.columns(num_cols)
    for i, r in enumerate(st.session_state.last_results):
        with cols[i % len(cols)]:
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

# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask about colors, fabrics, patterns, prices, or finding matches...")
actual_input = quick_prompt or user_input

if actual_input:
    st.session_state.chat_history.append(HumanMessage(content=actual_input))
    with st.chat_message("user"):
        st.markdown(actual_input)

    captured_results = []

    def _capture(results):
        captured_results.extend(results)

    with st.chat_message("assistant"):
        with st.spinner("Consulting TailorTalk catalogue..."):
            try:
                executor = build_agent_executor(st.session_state.current_image, _capture)
                response = executor.invoke(
                    {"input": actual_input, "chat_history": st.session_state.chat_history[:-1]}
                )
                answer = response["output"]
            except Exception as e:  # noqa: BLE001
                err_str = str(e)
                if "429" in err_str or "Quota" in err_str or "ResourceExhausted" in err_str or "rate" in err_str.lower():
                    from search_tool import search_sarees
                    inp_lower = actual_input.lower()
                    color_match = None
                    for c in ['baby pink', 'rani pink', 'peach pink', 'pink', 'navy blue', 'sky blue', 'blue', 'mint green', 'green', 'yellow', 'mustard', 'red', 'maroon', 'white', 'cream', 'black', 'purple', 'orange', 'gold']:
                        if c in inp_lower:
                            color_match = c
                            break
                    fabric_match = None
                    for f in ['banarasi', 'organza', 'ajrakh', 'pashmina', 'linen', 'satin', 'tussar', 'munga', 'silk', 'cotton', 'tissue']:
                        if f in inp_lower:
                            fabric_match = f
                            break
                    fallback_results = search_sarees(
                        query_image=st.session_state.current_image,
                        color=color_match,
                        fabric=fabric_match,
                        top_k=5,
                    )
                    captured_results.extend(fallback_results)
                    if fallback_results:
                        filter_desc = f" ({color_match or fabric_match})" if (color_match or fabric_match) else ""
                        if st.session_state.current_image is not None:
                            answer = f"Found {len(fallback_results)} visually matching sarees from our catalogue{filter_desc} based on your query."
                        else:
                            answer = f"Found {len(fallback_results)} sarees matching your request{filter_desc}."
                    else:
                        answer = "No matching sarees found for those specific filters. Try adjusting your search query."
                else:
                    answer = f"Something went wrong: {e}"
        st.markdown(answer)

    st.session_state.chat_history.append(AIMessage(content=answer))
    if captured_results:
        st.session_state.last_results = captured_results
    st.rerun()
