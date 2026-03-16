"""
Pokemon Vector Search App — powered by Zilliz Cloud.
Find the right Pokemon using natural language descriptions, colors, types, and more.
"""

import streamlit as st
from zilliz_db import setup_database, search_pokemon

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Pokemon Finder", page_icon="⚡", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .pokemon-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .pokemon-name {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .pokemon-score {
        font-size: 14px;
        opacity: 0.85;
        margin-bottom: 12px;
    }
    .pokemon-detail {
        font-size: 14px;
        margin: 4px 0;
    }
    .type-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Type color map ───────────────────────────────────────────
TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0",
    "Electric": "#F8D030", "Grass": "#78C850", "Ice": "#98D8D8",
    "Fighting": "#C03028", "Poison": "#A040A0", "Ground": "#E0C068",
    "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8",
    "Dark": "#705848", "Steel": "#B8B8D0", "Fairy": "#EE99AC",
}


def type_badges_html(types_str: str) -> str:
    """Generate colored badge HTML for Pokemon types."""
    badges = []
    for t in types_str.split(", "):
        color = TYPE_COLORS.get(t.strip(), "#888")
        badges.append(f'<span class="type-badge" style="background:{color};color:white">{t.strip()}</span>')
    return " ".join(badges)


# ── Init database connection (cached) ───────────────────────
@st.cache_resource
def init_db():
    return setup_database()


try:
    client, model = init_db()
    db_connected = True
except Exception as e:
    db_connected = False
    db_error = str(e)

# ── Header ───────────────────────────────────────────────────
st.title("Pokemon Finder")
st.markdown("*Semantic search powered by Zilliz Cloud vector database*")

if not db_connected:
    st.error(f"Could not connect to Zilliz Cloud: {db_error}")
    st.info(
        "**Setup required:**\n"
        "1. Create a free cluster at [cloud.zilliz.com](https://cloud.zilliz.com)\n"
        "2. Set environment variables:\n"
        "```bash\n"
        "export ZILLIZ_URI='https://your-cluster.zillizcloud.com'\n"
        "export ZILLIZ_TOKEN='your-api-key'\n"
        "```\n"
        "3. Restart the app"
    )
    st.stop()

# ── Sidebar filters ─────────────────────────────────────────
st.sidebar.header("Filters")

color_options = ["Any", "red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "brown", "gray"]
color_filter = st.sidebar.selectbox("Color", color_options)
if color_filter == "Any":
    color_filter = None

type_options = ["Any"] + sorted(TYPE_COLORS.keys())
type_filter = st.sidebar.selectbox("Type / Element", type_options)
if type_filter == "Any":
    type_filter = None

legendary_only = st.sidebar.checkbox("Legendary only")

top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)

# ── Search ───────────────────────────────────────────────────
st.markdown("---")
query = st.text_input(
    "Describe the Pokemon you're looking for:",
    placeholder="e.g. a big blue dragon that lives in the ocean",
)

# Example queries
st.markdown("**Try these:**")
example_cols = st.columns(4)
examples = [
    "small cute yellow electric mouse",
    "big scary fire-breathing dragon",
    "elegant psychic fairy dancer",
    "dark spooky ghost with purple flames",
]
for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        query = example

if query:
    with st.spinner("Searching Zilliz..."):
        results = search_pokemon(
            client, model, query,
            top_k=top_k,
            color_filter=color_filter,
            type_filter=type_filter,
            legendary_only=legendary_only,
        )

    if not results:
        st.warning("No Pokemon found matching your criteria. Try adjusting the filters.")
    else:
        st.markdown(f"### Top {len(results)} matches")

        for i, pokemon in enumerate(results):
            similarity_pct = pokemon["score"] * 100
            legendary_tag = " ★ LEGENDARY" if pokemon.get("is_legendary") else ""

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="pokemon-card">
                    <div class="pokemon-name">#{i+1} {pokemon['name']}{legendary_tag}</div>
                    <div class="pokemon-score">Match: {similarity_pct:.1f}%</div>
                    <div>{type_badges_html(pokemon['types'])}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"**Color:** {pokemon['color'].title()}")
                st.markdown(f"**Shape:** {pokemon['shape'].title()}")
                st.markdown(f"**Size:** {pokemon['height_m']}m / {pokemon['weight_kg']}kg")
                st.markdown(f"**Generation:** {pokemon['generation']}")
                st.markdown(f"*{pokemon['description']}*")

            st.markdown("---")

# ── Footer ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:**\n"
    "Your query is converted to a vector embedding and compared against "
    "Pokemon descriptions stored in Zilliz Cloud using cosine similarity."
    "Try Zilliz for free on cloud.zilliz.com"
)
