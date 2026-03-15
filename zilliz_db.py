"""
Zilliz Cloud vector database module.
Handles connection, collection creation, embedding, and search.
"""

import os
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from pokemon_data import POKEMON, build_search_text

COLLECTION_NAME = "pokemon_search"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model for generating embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_zilliz_client() -> MilvusClient:
    """Connect to Zilliz Cloud using environment variables."""
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")

    if not uri or not token:
        raise ValueError(
            "Set ZILLIZ_URI and ZILLIZ_TOKEN environment variables.\n"
            "Get these from your Zilliz Cloud console: https://cloud.zilliz.com"
        )

    return MilvusClient(uri=uri, token=token)


def create_collection(client: MilvusClient) -> None:
    """Create the Pokemon collection in Zilliz if it doesn't exist."""
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="types", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="shape", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="height_m", datatype=DataType.FLOAT)
    schema.add_field(field_name="weight_kg", datatype=DataType.FLOAT)
    schema.add_field(field_name="generation", datatype=DataType.INT64)
    schema.add_field(field_name="is_legendary", datatype=DataType.BOOL)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")


def index_pokemon(client: MilvusClient, model: SentenceTransformer) -> None:
    """Generate embeddings and insert all Pokemon into Zilliz."""
    texts = [build_search_text(p) for p in POKEMON]
    embeddings = model.encode(texts, show_progress_bar=True)

    data = []
    for pokemon, embedding in zip(POKEMON, embeddings):
        data.append({
            "id": pokemon["id"],
            "name": pokemon["name"],
            "types": ", ".join(pokemon["types"]),
            "color": pokemon["color"],
            "shape": pokemon["shape"],
            "height_m": pokemon["height_m"],
            "weight_kg": pokemon["weight_kg"],
            "generation": pokemon["generation"],
            "is_legendary": pokemon["is_legendary"],
            "description": pokemon["description"],
            "embedding": embedding.tolist(),
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Inserted {len(data)} Pokemon into Zilliz.")


def search_pokemon(
    client: MilvusClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
    color_filter=None,
    type_filter=None,
    legendary_only=False,
):
    """
    Semantic search for Pokemon matching a natural language query.

    Args:
        client: Zilliz client
        model: Sentence transformer model
        query: Natural language search query
        top_k: Number of results to return
        color_filter: Optional color filter (e.g., "blue")
        type_filter: Optional type filter (e.g., "Fire")
        legendary_only: Only return legendary Pokemon

    Returns:
        List of matching Pokemon with similarity scores
    """
    query_embedding = model.encode([query])[0].tolist()

    # Build filter expression
    filters = []
    if color_filter:
        filters.append(f'color == "{color_filter.lower()}"')
    if type_filter:
        filters.append(f'types like "%{type_filter}%"')
    if legendary_only:
        filters.append("is_legendary == true")

    filter_expr = " and ".join(filters) if filters else ""

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["name", "types", "color", "shape", "height_m", "weight_kg",
                        "generation", "is_legendary", "description"],
        filter=filter_expr if filter_expr else None,
    )

    matches = []
    for hit in results[0]:
        entity = hit["entity"]
        entity["score"] = hit["distance"]
        matches.append(entity)

    return matches


def setup_database() -> tuple[MilvusClient, SentenceTransformer]:
    """Full setup: connect, create collection, index data. Returns client and model."""
    print("Connecting to Zilliz Cloud...")
    client = get_zilliz_client()

    print("Loading embedding model...")
    model = get_embedding_model()

    create_collection(client)

    # Check if data is already indexed
    count = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["count(*)"],
    )
    if count and count[0].get("count(*)", 0) > 0:
        print(f"Collection already has {count[0]['count(*)']} records.")
    else:
        print("Indexing Pokemon data...")
        index_pokemon(client, model)

    return client, model


if __name__ == "__main__":
    client, model = setup_database()
    # Quick test search
    results = search_pokemon(client, model, "a cute small yellow electric mouse")
    for r in results:
        print(f"  {r['name']} ({r['types']}) - Score: {r['score']:.4f}")
