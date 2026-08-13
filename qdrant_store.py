"""
Dedicated Qdrant Operations Module for TailorTalk Saree Search.
Handles client connection, collection creation, batch upserting, similarity searching,
and payload retrieval while keeping Qdrant-specific logic cleanly isolated.
"""
import os
import uuid
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue

import config


_QDRANT_CLIENT_INSTANCE = None

def get_qdrant_client() -> QdrantClient:
    """
    Returns a QdrantClient instance (cached singleton).
    Priority:
    1. Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY)
    2. Remote Qdrant server (QDRANT_URL)
    3. Local embedded Qdrant disk storage (QDRANT_LOCAL_PATH)
    """
    global _QDRANT_CLIENT_INSTANCE
    if _QDRANT_CLIENT_INSTANCE is not None:
        return _QDRANT_CLIENT_INSTANCE

    url = config.QDRANT_URL
    api_key = config.QDRANT_API_KEY

    if url and api_key:
        _QDRANT_CLIENT_INSTANCE = QdrantClient(url=url, api_key=api_key)
    elif url:
        _QDRANT_CLIENT_INSTANCE = QdrantClient(url=url)
    else:
        # Local embedded persistent Qdrant storage
        os.makedirs(config.QDRANT_LOCAL_PATH, exist_ok=True)
        _QDRANT_CLIENT_INSTANCE = QdrantClient(path=config.QDRANT_LOCAL_PATH)

    return _QDRANT_CLIENT_INSTANCE



def collection_exists(client: QdrantClient, collection_name: str = config.QDRANT_COLLECTION_NAME) -> bool:
    """Check if a Qdrant collection exists."""
    try:
        cols = client.get_collections().collections
        return any(c.name == collection_name for c in cols)
    except Exception:
        return False


def create_collection(client: QdrantClient, collection_name: str = config.QDRANT_COLLECTION_NAME) -> bool:
    """Create a new Qdrant collection configured for 1024d vectors and COSINE distance."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        # Create payload index for numerical price range filtering
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="price_numeric",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")
        return False


def ensure_collection(client: QdrantClient, collection_name: str = config.QDRANT_COLLECTION_NAME) -> bool:
    """Ensure the Qdrant collection exists and has necessary payload indexes."""
    if not collection_exists(client, collection_name):
        create_collection(client, collection_name)

    # Ensure price_numeric payload index exists for Qdrant Cloud filtering
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="price_numeric",
            field_schema=models.PayloadSchemaType.FLOAT,
        )
    except Exception:
        pass

    return True



def delete_collection(client: QdrantClient, collection_name: str = config.QDRANT_COLLECTION_NAME) -> bool:
    """Delete a Qdrant collection if it exists."""
    try:
        if collection_exists(client, collection_name):
            client.delete_collection(collection_name=collection_name)
            return True
        return False
    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")
        return False


def get_collection_info(client: QdrantClient, collection_name: str = config.QDRANT_COLLECTION_NAME) -> dict:
    """Retrieve summary information about the collection."""
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {"error": str(e)}


def health_check(client: QdrantClient) -> bool:
    """Ping Qdrant connection to verify health."""
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def generate_point_id(record_idx: int, image_url: str) -> int:
    """Generate a stable, deterministic integer ID for Qdrant point."""
    # Uses 1-indexed record number (1..1070)
    return record_idx + 1


def prepare_payload(row: pd.Series) -> Dict[str, Any]:
    """Convert metadata row into a clean JSON-serializable Qdrant payload dictionary."""
    payload = {}
    for col in row.index:
        val = row[col]
        if pd.isnull(val) or val is None:
            payload[col] = None
        elif isinstance(val, (np.integer, int)):
            payload[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            payload[col] = float(val)
        elif isinstance(val, (np.bool_, bool)):
            payload[col] = bool(val)
        else:
            payload[col] = str(val)

    # Parse numeric price float for payload filtering
    try:
        price_clean = float(str(row.get("price", "0")).replace("₹", "").replace(",", "").strip())
        payload["price_numeric"] = price_clean
    except Exception:
        payload["price_numeric"] = 0.0

    return payload


def upsert_sarees(
    client: QdrantClient,
    vectors: np.ndarray,
    meta_df: pd.DataFrame,
    collection_name: str = config.QDRANT_COLLECTION_NAME,
    batch_size: int = 100,
) -> int:
    """
    Batch upserts sarees into Qdrant idempotently with stable point IDs.
    Returns the total number of points successfully upserted.
    """
    ensure_collection(client, collection_name)
    total_points = len(meta_df)
    print(f"Upserting {total_points} points into Qdrant collection '{collection_name}'...")

    points = []
    for i in range(total_points):
        point_id = generate_point_id(i, meta_df.iloc[i]["image_url"])
        vector = vectors[i].tolist()
        payload = prepare_payload(meta_df.iloc[i])

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)

    print(f"Successfully upserted {total_points} points to Qdrant!")
    return total_points


def build_price_filter(min_price: Optional[float] = None, max_price: Optional[float] = None) -> Optional[Filter]:
    """Build a Qdrant payload filter for price range constraints."""
    if min_price is None and max_price is None:
        return None

    range_kwargs = {}
    if min_price is not None:
        range_kwargs["gte"] = float(min_price)
    if max_price is not None:
        range_kwargs["lte"] = float(max_price)

    return Filter(must=[FieldCondition(key="price_numeric", range=Range(**range_kwargs))])


def search_sarees(
    client: QdrantClient,
    query_vector: np.ndarray,
    limit: int = 100,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    collection_name: str = config.QDRANT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Executes vector similarity search on Qdrant with optional payload price filter.
    Returns list of dictionaries containing point_id, score, and payload metadata.
    """
    query_list = query_vector.flatten().tolist()
    q_filter = build_price_filter(min_price, max_price)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_list,
        query_filter=q_filter,
        limit=limit,
        with_payload=True,
    )

    results = []
    for hit in search_result:
        results.append({
            "point_id": hit.id,
            "score": float(hit.score),
            "payload": hit.payload,
        })
    return results
