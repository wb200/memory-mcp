#!/usr/bin/env python3
"""
Droid Memory MCP Server - LanceDB Hybrid Search Implementation

Provides persistent memory with TRUE hybrid search (vector + BM25) using:
- FastMCP for clean, idiomatic MCP server patterns
- LanceDB for vector + full-text hybrid search with RRF fusion
- CrossEncoderReranker for neural reranking
- Ollama/qwen3-embedding for local embeddings (1024-dim), Google Gemini fallback
- Google Gemini for summarization
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CrossEncoderReranker
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from google.genai import Client as GenAIClient

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class Config:
    """Server configuration with sensible defaults."""

    db_path: Path = Path(os.environ.get("DROID_MEMORY_DB_PATH", Path.home() / ".memory-mcp" / "lancedb-memory"))
    table_name: str = "memories"
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "1024"))
    embedding_provider: str = os.environ.get("EMBEDDING_PROVIDER", "ollama")  # ollama | google
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = "gemini-3-flash-preview"
    ttl_days: int = 365
    dedup_threshold: float = 0.90
    default_limit: int = 5
    max_limit: int = 50
    fts_weight: float = 0.3  # Weight for FTS in hybrid fusion (vector gets 1 - this)


CONFIG = Config()

VALID_CATEGORIES = frozenset(
    {"PATTERN", "CONFIG", "DEBUG", "PERF", "PREF", "INSIGHT", "API", "AGENT"}
)
ID_PREFIX_MATCH_LIMIT = 100
CLEANUP_INTERVAL_HOURS = 24


# =============================================================================
# LanceDB Schema
# =============================================================================


class Memory(LanceModel):
    """LanceDB schema for memories with vector embeddings."""

    id: str  # UUID string - thread-safe, no race conditions
    content: str  # Indexed for FTS
    vector: Vector(CONFIG.embedding_dim)  # type: ignore[valid-type]
    category: str
    tags: str  # JSON array as string
    project_id: str
    user_id: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None


# =============================================================================
# Thread-Safety Lock (for singleton initialization)
# =============================================================================

_lock = threading.RLock()  # RLock allows reentrant calls (get_table -> get_db)

# =============================================================================
# Gemini Client (Lazy Singleton)
# =============================================================================

_genai_client: GenAIClient | None = None


def _get_api_key() -> str:
    """Get API key from environment or secrets file."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    secrets_path = Path.home() / ".secrets" / "GOOGLE_API_KEY"
    if secrets_path.exists():
        return secrets_path.read_text().strip()
    raise ValueError(
        "GOOGLE_API_KEY not found. Set environment variable or create ~/.secrets/GOOGLE_API_KEY"
    )


def get_genai_client() -> GenAIClient:
    """Get or create the GenAI client singleton (thread-safe)."""
    global _genai_client
    if _genai_client is None:
        with _lock:
            if _genai_client is None:  # Double-check after acquiring lock
                from google import genai

                _genai_client = genai.Client(api_key=_get_api_key())
    return _genai_client


# =============================================================================
# Database Layer (LanceDB)
# =============================================================================

_db: lancedb.DBConnection | None = None
_table: lancedb.table.Table | None = None
_reranker: CrossEncoderReranker | None = None
_cleanup_task: asyncio.Task | None = None


def get_db() -> lancedb.DBConnection:
    """Get or create LanceDB connection (thread-safe)."""
    global _db
    if _db is None:
        with _lock:
            if _db is None:  # Double-check after acquiring lock
                CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)
                _db = lancedb.connect(str(CONFIG.db_path))
    return _db


def get_table() -> lancedb.table.Table:
    """Get or create the memories table (thread-safe)."""
    global _table
    if _table is None:
        with _lock:
            if _table is None:  # Double-check after acquiring lock
                db = get_db()
                try:
                    _table = db.open_table(CONFIG.table_name)
                except Exception:
                    try:
                        table_names = db.list_tables()
                    except AttributeError:
                        table_names = db.table_names()
                    except Exception:
                        table_names = []
                    if CONFIG.table_name in table_names:
                        raise
                    _table = db.create_table(CONFIG.table_name, schema=Memory)
    return _table


def get_reranker() -> CrossEncoderReranker:
    """Get or create the CrossEncoder reranker (thread-safe)."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:  # Double-check after acquiring lock
                _reranker = CrossEncoderReranker(
                    model_name="mixedbread-ai/mxbai-reranker-base-v2"
                )
    return _reranker


def _escape_filter_value(value: str) -> str:
    """Escape single quotes in filter values to prevent injection."""
    return value.replace("'", "''")


def _normalize_category(category: str | None) -> tuple[str | None, str | None]:
    """Normalize and validate a category string."""
    if category is None:
        return None, None
    normalized = category.upper()
    if normalized not in VALID_CATEGORIES:
        return None, f"Error: Invalid category '{category}'. Valid: {sorted(VALID_CATEGORIES)}"
    return normalized, None


def _find_memory_by_id(
    table: lancedb.table.Table, memory_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Find a memory by full or partial UUID, with ambiguity detection."""
    safe_id = _escape_filter_value(memory_id)
    if len(memory_id) < 32:
        results = (
            table.search()
            .where(f"id LIKE '{safe_id}%'")
            .limit(ID_PREFIX_MATCH_LIMIT)
            .to_list()
        )
        if not results:
            return None, f"Memory {memory_id} not found"
        if len(results) > 1:
            ids = ", ".join(r["id"] for r in results)
            if len(results) >= ID_PREFIX_MATCH_LIMIT:
                ids = f"{ids}..."
                return (
                    None,
                    "Error: Ambiguous ID prefix. Showing first "
                    f"{ID_PREFIX_MATCH_LIMIT} matches: {ids}. Provide full 32-char ID.",
                )
            return (
                None,
                f"Error: Ambiguous ID prefix. Matches: {ids}. Provide full 32-char ID.",
            )
        return results[0], None

    results = table.search().where(f"id = '{safe_id}'").limit(1).to_list()
    if not results:
        return None, f"Memory {memory_id} not found"
    return results[0], None


async def init_database() -> None:
    """Initialize database with FTS and vector indexes, warm up models."""
    table = get_table()

    # Create FTS index on content (for BM25 search) - check if exists first
    indices = []
    try:
        indices = table.list_indices()
        has_fts = any("fts" in str(idx).lower() or "content" in str(idx).lower() for idx in indices)
        if not has_fts:
            table.create_fts_index("content", use_tantivy=True, replace=True)
            print("[memory-mcp] Tantivy-based FTS index (BM25) created on 'content'", file=sys.stderr)
        else:
            print("[memory-mcp] FTS index already exists", file=sys.stderr)
    except Exception as e:
        print(f"[memory-mcp] FTS index warning: {e}", file=sys.stderr)

    # Create IVF-PQ index for ANN search
    try:
        has_vector_idx = any("ivf" in str(idx).lower() for idx in indices)
        if has_vector_idx:
            print("[memory-mcp] Vector index already exists", file=sys.stderr)
        else:
            row_count = table.count_rows()
            if row_count >= 10:
                table.create_index(
                    metric="cosine",
                    num_partitions=max(4, int(row_count**0.5)),
                    num_sub_vectors=48,
                    index_type="IVF_PQ",
                    replace=True,
                )
                print("[memory-mcp] IVF-PQ index created", file=sys.stderr)
    except Exception as e:
        print(f"[memory-mcp] Vector index warning: {e}", file=sys.stderr)

    # Warm up CrossEncoder reranker (loads model ~2-3s on first use)
    print("[memory-mcp] Warming up CrossEncoder reranker...", file=sys.stderr)
    get_reranker()
    print("[memory-mcp] Server ready", file=sys.stderr)


# =============================================================================
# Embedding & Summarization
# =============================================================================


def _compute_embedding_ollama(text: str) -> list[float] | None:
    """Generate embedding using Ollama (local, fallback option)."""
    try:
        import requests

        response = requests.post(
            f"{CONFIG.ollama_base_url}/api/embeddings",
            json={
                "model": CONFIG.embedding_model,
                "prompt": text,
            },
            timeout=30,
        )
        response.raise_for_status()
        embedding = np.array(response.json().get("embedding", []))

        # Handle dimension mismatch by truncation/padding
        if len(embedding) != CONFIG.embedding_dim:
            if len(embedding) > CONFIG.embedding_dim:
                embedding = embedding[:CONFIG.embedding_dim]
            else:
                # Pad with zeros
                padding = np.zeros(CONFIG.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])

        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else embedding.tolist()
    except Exception as e:
        print(f"[memory-mcp] Ollama embedding error: {e}", file=sys.stderr)
        return None


def _compute_embedding_google(text: str, task_type: str) -> list[float] | None:
    """Generate embedding using Google Genai API."""
    try:
        from google.genai import types

        client = get_genai_client()
        response = client.models.embed_content(
            model=CONFIG.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=CONFIG.embedding_dim
            ),
        )
        embedding = np.array(response.embeddings[0].values)
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else embedding.tolist()
    except Exception as e:
        print(f"[memory-mcp] Google embedding error: {e}", file=sys.stderr)
        return None


def _compute_embedding_hash_fallback(text: str) -> list[float]:
    """
    Last-resort fallback: deterministic hash-based embedding.
    Not real semantic meaning, but ensures memory can be saved.
    """
    import hashlib

    # Create deterministic hash of content
    h = hashlib.sha256(text.encode()).digest()
    # Use first 768 bytes and normalize to [-1, 1]
    values = [(b - 128) / 128.0 for b in h[:CONFIG.embedding_dim]]

    # Normalize to unit length
    norm = np.linalg.norm(values)
    if norm > 0:
        values = [v / norm for v in values]

    return values


def _compute_embedding_sync(
    text: str, task_type: str = "SEMANTIC_SIMILARITY"
) -> list[float] | None:
    """Synchronous embedding computation with provider fallback chain."""
    provider = CONFIG.embedding_provider.lower()

    # Try Ollama first if configured as primary
    if provider == "ollama":
        result = _compute_embedding_ollama(text)
        if result:
            return result
        print("[memory-mcp] Ollama failed, falling back to Google", file=sys.stderr)

    # Try Google
    result = _compute_embedding_google(text, task_type)
    if result:
        return result

    # Last resort: hash-based fallback (never fails, but poor semantic quality)
    print("[memory-mcp] Using hash fallback embedding (poor semantic quality)", file=sys.stderr)
    return _compute_embedding_hash_fallback(text)


@lru_cache(maxsize=128)
def _compute_embedding_cached(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> tuple[float, ...] | None:
    """Cached embedding computation to avoid redundant API calls."""
    result = _compute_embedding_sync(text, task_type)
    return tuple(result) if result else None


async def get_embedding(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float] | None:
    """Generate embedding asynchronously with LRU cache."""
    cached = _compute_embedding_cached(text, task_type)
    if cached is None:
        print(f"[memory-mcp] Embedding cache miss/failed for content length={len(text)}", file=sys.stderr)
    return list(cached) if cached else None


def _summarize_sync(content: str, category: str) -> dict[str, Any]:
    """Synchronous summarization (for thread pool)."""
    try:
        client = get_genai_client()
        prompt = f"""Extract the key insight from this content. Return JSON only.

CONTENT: {content}
CATEGORY: {category}

Return: {{"summary": "CATEGORY - concise insight with context", "tags": ["tag1", "tag2"]}}"""

        response = client.models.generate_content(model=CONFIG.llm_model, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        result = json.loads(text)
        return {"summary": result.get("summary", content[:500]), "tags": result.get("tags", [])}
    except Exception as e:
        print(f"[memory-mcp] Summarization error: {e}", file=sys.stderr)
        return {"summary": content[:500] if len(content) > 500 else content, "tags": []}


async def smart_summarize(content: str, category: str = "INSIGHT") -> dict[str, Any]:
    """Summarize content asynchronously using LLM."""
    return await asyncio.to_thread(_summarize_sync, content, category)


# =============================================================================
# Utilities
# =============================================================================


@lru_cache(maxsize=1)
def get_project_id() -> str:
    """Get current project identifier from git or cwd. Cached per session."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:  # noqa: S110 - git may not be available
        pass
    return str(Path.cwd())


def now_iso() -> str:
    """Get current timestamp as ISO string."""
    return datetime.now().isoformat()


def expires_iso() -> str:
    """Get expiration timestamp as ISO string."""
    return (datetime.now() + timedelta(days=CONFIG.ttl_days)).isoformat()


def _candidates_to_arrow(candidates: list[dict]) -> pa.Table:
    """Convert candidate dictionaries to PyArrow table for reranker.

    Note: CrossEncoderReranker expects 'text' field, not 'content'.
    We include both 'text' (for reranker) and 'content' (for display).
    Also preserve _distance for similarity display in results.
    """
    return pa.table({
        "id": [r["id"] for r in candidates],
        "text": [r["content"] for r in candidates],  # Reranker expects 'text'
        "content": [r["content"] for r in candidates],  # Keep for result display
        "vector": [r["vector"] for r in candidates],
        "category": [r["category"] for r in candidates],
        "tags": [r["tags"] for r in candidates],
        "project_id": [r["project_id"] for r in candidates],
        "created_at": [r["created_at"] for r in candidates],
        "updated_at": [r["updated_at"] for r in candidates],
        "_distance": [r.get("_distance", 0.0) for r in candidates],  # Preserve for display
    })


def _rrf_fusion(vector_results: list[dict], fts_results: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion to combine vector and FTS results."""
    scores: dict[str, float] = {}
    all_results: dict[str, dict] = {}

    # Score from vector results (weight: 1 - fts_weight)
    vector_weight = 1 - CONFIG.fts_weight
    for rank, r in enumerate(vector_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + vector_weight / (k + rank + 1)
        all_results[rid] = r

    # Score from FTS results (weight: fts_weight)
    for rank, r in enumerate(fts_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + CONFIG.fts_weight / (k + rank + 1)
        if rid not in all_results:
            all_results[rid] = r

    # Sort by combined score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [all_results[rid] for rid in sorted_ids]


# =============================================================================
# FastMCP Server
# =============================================================================

mcp = FastMCP(
    "droid-memory",
    instructions="Persistent memory with LanceDB TRUE hybrid search (vector + BM25 RRF fusion) and neural reranking",
)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    }
)
async def memory_save(
    content: str,
    category: str = "INSIGHT",
    tags: list[str] | None = None,
    summarize: bool = False,
) -> str:
    """Save a memory with semantic embedding. Use after learning something valuable.

    Args:
        content: Memory content. Format: '[CATEGORY] - [insight]. Context: [where]. Rationale: [why]'
        category: One of PATTERN, CONFIG, DEBUG, PERF, PREF, INSIGHT, API, AGENT
        tags: Optional tags for categorization (auto-extracted if summarize=True)
        summarize: Use LLM to intelligently summarize verbose content
    """
    if not content.strip():
        return "Error: content is required"

    # Validate category
    normalized_category, error = _normalize_category(category)
    if error:
        return error
    category = normalized_category or category

    tags = tags or []
    original_len = len(content)

    if summarize:
        result = await smart_summarize(content, category)
        content = result["summary"]
        tags = list(dict.fromkeys(tags + result.get("tags", [])))

    embedding = await get_embedding(content)
    if embedding is None:
        return f"Error: Failed to generate embedding. Check API key configuration and network connectivity."

    table = get_table()

    # Deduplication check
    try:
        results = table.search(embedding).metric("cosine").limit(1).to_list()
        if results:
            similarity = 1 - results[0]["_distance"]
            if similarity >= CONFIG.dedup_threshold:
                return (
                    f"Duplicate detected ({similarity:.0%} similar)\n"
                    f"Existing ID: {results[0]['id']}\n"
                    f"Existing: {results[0]['content'][:200]}..."
                )
    except Exception as e:
        print(f"[memory-mcp] Dedup check warning: {e}", file=sys.stderr)

    # Create memory with UUID (thread-safe, no race conditions)
    memory_id = uuid.uuid4().hex
    timestamp = now_iso()
    memory = Memory(
        id=memory_id,
        content=content,
        vector=embedding,
        category=category,
        tags=json.dumps(tags),
        project_id=get_project_id(),
        created_at=timestamp,
        updated_at=timestamp,
        expires_at=expires_iso(),
    )
    table.add([memory.model_dump()])

    parts = [f"Saved (ID: {memory_id[:8]}..., {category})"]
    if summarize:
        parts.append(f"Summarized: {original_len} → {len(content)} chars")
    parts.append(f"Tags: {tags}")
    return "\n".join(parts)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    }
)
async def memory_recall(
    query: str,
    category: str | None = None,
    limit: int = 5,
) -> str:
    """Hybrid search (vector + BM25) across ALL projects with neural reranking.

    Args:
        query: Search query - works with both keywords and semantic concepts
        category: Optional filter: PATTERN, CONFIG, DEBUG, PERF, PREF, INSIGHT, API, AGENT
        limit: Max results (default 5, max 50)
    """
    return await _recall(query, category, limit, project_scope=False)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    }
)
async def memory_recall_project(
    query: str,
    category: str | None = None,
    limit: int = 5,
) -> str:
    """Hybrid search in CURRENT project only with neural reranking.

    Args:
        query: Search query - works with both keywords and semantic concepts
        category: Optional category filter
        limit: Max results (default 5, max 50)
    """
    return await _recall(query, category, limit, project_scope=True)


async def _recall(query: str, category: str | None, limit: int, project_scope: bool) -> str:
    """Core recall with TRUE hybrid search: Vector + FTS with RRF fusion."""
    if not query.strip():
        return "Error: query is required"

    # Validate limit
    if limit <= 0:
        return f"Error: limit must be positive, got {limit}"
    if limit > CONFIG.max_limit:
        return f"Error: limit cannot exceed {CONFIG.max_limit}, got {limit}"
    limit = min(limit, CONFIG.max_limit)

    # Validate category if provided
    normalized_category, error = _normalize_category(category)
    if error:
        return error
    category = normalized_category

    # TRUE HYBRID SEARCH: Run both vector and FTS searches
    fetch_limit = limit * 3
    vector_results = []
    fts_results = []
    embedding = await get_embedding(query, task_type="RETRIEVAL_QUERY")
    
    table = get_table()
    reranker = get_reranker()

    # Build filter expression with escaped values
    filters = []
    now = datetime.now().isoformat()
    filters.append(f"(expires_at IS NULL OR expires_at > '{_escape_filter_value(now)}')")
    if project_scope:
        project_id = _escape_filter_value(get_project_id())
        filters.append(f"project_id = '{project_id}'")
    if category:
        filters.append(f"category = '{_escape_filter_value(category)}'")
    filter_expr = " AND ".join(filters) if filters else None

    # 1. Vector search (graceful degradation if embedding fails)
    if embedding is not None:
        vector_search = table.search(embedding).metric("cosine")
        if filter_expr:
            vector_search = vector_search.where(filter_expr)
        vector_results = vector_search.limit(fetch_limit).to_list()
    else:
        print("[memory-mcp] Embedding failed, using FTS-only search", file=sys.stderr)

    # 2. Full-text search (BM25)
    fts_results = []
    try:
        fts_search = table.search(query, query_type="fts")
        if filter_expr:
            fts_search = fts_search.where(filter_expr)
        fts_results = fts_search.limit(fetch_limit).to_list()
    except Exception as e:
        print(
            f"[memory-mcp] FTS search warning (falling back to vector-only): {e}", file=sys.stderr
        )

    # 3. Fuse results using Reciprocal Rank Fusion
    if fts_results:
        candidates = _rrf_fusion(vector_results, fts_results)
        search_type = "hybrid (vector + BM25 RRF)"
    else:
        candidates = vector_results
        search_type = "vector"

    # 4. Neural reranking with CrossEncoder
    if candidates:
        try:
            arrow_table = _candidates_to_arrow(candidates)
            reranked = reranker.rerank_vector(query, arrow_table)[:limit]
            results = reranked.to_pylist()
        except Exception as e:
            print(f"[memory-mcp] Reranker error: {e}", file=sys.stderr)
            results = candidates[:limit]
    else:
        results = []

    if not results:
        scope = "current project" if project_scope else "all projects"
        return f"No memories found for '{query}' in {scope}"

    # Format results
    scope = "project" if project_scope else "global"
    lines = [f"Found {len(results)} memories ({scope}, {search_type} + neural rerank):\n"]

    for i, row in enumerate(results, 1):
        tags_list = json.loads(row["tags"]) if row["tags"] else []
        proj = row["project_id"][:40] + "..." if len(row["project_id"]) > 40 else row["project_id"]

        lines.append(f"[{i}] {row['category']} (ID: {row['id'][:8]}...)")
        lines.append(f"    {row['content']}")
        if tags_list:
            lines.append(f"    Tags: {', '.join(tags_list)}")
        lines.append(f"    Project: {proj} | {row['created_at'][:19]}")

        if "_relevance_score" in row:
            lines.append(f"    Relevance: {row['_relevance_score']:.2f}")
        elif "_distance" in row:
            lines.append(f"    Similarity: {1 - row['_distance']:.0%}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
    }
)
async def memory_delete(memory_id: str) -> str:
    """Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete (full or partial UUID)
    """
    table = get_table()
    existing, error = _find_memory_by_id(table, memory_id)
    if error:
        return error
    full_id = existing["id"]
    table.delete(f"id = '{_escape_filter_value(full_id)}'")
    return f"Deleted memory {full_id[:8]}..."


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    }
)
async def memory_update(
    memory_id: str,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing memory.

    Args:
        memory_id: The ID of the memory to update (full or partial UUID)
        content: New content (re-embeds if changed)
        category: New category
        tags: New tags (replaces existing)
    """
    table = get_table()
    existing, error = _find_memory_by_id(table, memory_id)
    if error:
        return error
    full_id = existing["id"]

    # Validate category if provided
    normalized_category, error = _normalize_category(category)
    if error:
        return error
    category = normalized_category

    new_content = content if content is not None else existing["content"]
    new_category = category if category is not None else existing["category"]
    new_tags = json.dumps(tags) if tags is not None else existing["tags"]

    if content is not None:
        embedding = await get_embedding(new_content)
        if embedding is None:
            return "Error: Failed to generate embedding for updated content"
        new_vector = embedding
    else:
        new_vector = existing["vector"]

    old_updated_at = existing["updated_at"]
    old_created_at = existing["created_at"]
    new_updated_at = now_iso()
    if new_updated_at == old_updated_at:
        new_updated_at = (datetime.now() + timedelta(microseconds=1)).isoformat()

    # Add first, then delete only the old row
    memory = Memory(
        id=full_id,
        content=new_content,
        vector=new_vector,
        category=new_category,
        tags=new_tags,
        project_id=existing["project_id"],
        user_id=existing.get("user_id"),
        created_at=old_created_at,
        updated_at=new_updated_at,
        expires_at=existing.get("expires_at"),
    )
    table.add([memory.model_dump()])
    table.delete(
        " AND ".join(
            [
                f"id = '{_escape_filter_value(full_id)}'",
                f"updated_at = '{_escape_filter_value(old_updated_at)}'",
                f"created_at = '{_escape_filter_value(old_created_at)}'",
            ]
        )
    )

    return f"Updated memory {full_id[:8]}..."


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    }
)
async def memory_stats() -> str:
    """Get memory system statistics - total, by category, by project."""
    table = get_table()
    total = table.count_rows()

    if total == 0:
        return "No memories stored yet."

    category_counts: dict[str, int] = {}
    project_counts: dict[str, int] = {}

    arrow_table = None
    try:
        arrow_table = table.to_arrow(columns=["category", "project_id"])
    except (TypeError, AttributeError):
        try:
            arrow_table = table.to_arrow()
            if hasattr(arrow_table, "select"):
                arrow_table = arrow_table.select(["category", "project_id"])
        except Exception:
            arrow_table = None

    if arrow_table is not None:
        arrow_data = arrow_table.to_pydict()
        for category in arrow_data.get("category", []):
            if category is None:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1
        for project in arrow_data.get("project_id", []):
            if project is None:
                continue
            project_counts[project] = project_counts.get(project, 0) + 1
    else:
        try:
            data = table.to_pydict()
            for category in data.get("category", []):
                if category is None:
                    continue
                category_counts[category] = category_counts.get(category, 0) + 1
            for project in data.get("project_id", []):
                if project is None:
                    continue
                project_counts[project] = project_counts.get(project, 0) + 1
        except Exception:
            try:
                rows = table.to_pylist()
            except Exception:
                rows = table.search().limit(total).to_list()
            for row in rows:
                category = row.get("category")
                project = row.get("project_id")
                if category is not None:
                    category_counts[category] = category_counts.get(category, 0) + 1
                if project is not None:
                    project_counts[project] = project_counts.get(project, 0) + 1

    # Sort and get top 5 projects
    by_project = dict(sorted(project_counts.items(), key=lambda x: x[1], reverse=True)[:5])

    # Try to get indices
    has_vector_index = False
    has_fts_index = False
    try:
        indices = table.list_indices()
        has_vector_index = any("ivf" in str(idx).lower() for idx in indices)
        has_fts_index = any("fts" in str(idx).lower() or "content" in str(idx).lower() for idx in indices)
    except Exception:
        pass

    db_size = sum(f.stat().st_size for f in CONFIG.db_path.rglob("*") if f.is_file()) / 1024

    lines = [
        "=== Memory Statistics (LanceDB) ===",
        f"Total: {total} memories",
        f"Database: {db_size:.1f} KB",
        f"Vector Index: {'Yes (IVF-PQ)' if has_vector_index else 'Flat (brute-force)'}",
        f"FTS Index: {'Yes (BM25)' if has_fts_index else 'No'}",
        "Search: Hybrid (Vector + BM25 RRF) + CrossEncoder reranking",
        "",
        "By Category:",
    ]
    for cat, count in sorted(category_counts.items()):
        lines.append(f"  {cat}: {count}")

    lines.append("\nBy Project (top 5):")
    for proj, count in by_project.items():
        proj_display = proj[:40] + "..." if len(proj) > 40 else proj
        lines.append(f"  {proj_display}: {count}")

    return "\n".join(lines)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    }
)
async def memory_health() -> str:
    """Get memory system health status - indexes, database size, configuration."""
    table = get_table()
    total = table.count_rows()

    has_fts = False
    has_vector = False
    try:
        indices = table.list_indices()
        has_fts = any("fts" in str(idx).lower() or "content" in str(idx).lower() for idx in indices)
        has_vector = any("ivf" in str(idx).lower() for idx in indices)
    except Exception:
        pass

    db_size = sum(f.stat().st_size for f in CONFIG.db_path.rglob("*") if f.is_file()) / 1024

    lines = [
        "=== Memory Health Status ===",
        f"\nTotal memories: {total}",
        f"Database size: {db_size:.1f} KB",
        f"FTS index: {'✓ BM25 enabled' if has_fts else '✗ Not enabled (requires tantivy-py)'}",
        f"Vector index: {'✓ IVF-PQ' if has_vector else 'Flat (brute-force)'}",
        f"TTL: {CONFIG.ttl_days} days",
        f"Search: Hybrid (vector + BM25 RRF) + CrossEncoder rerank",
        f"Embedding cache: LRU (maxsize=128)",
    ]

    # Check if cleanup task is running
    global _cleanup_task
    if _cleanup_task is not None and not _cleanup_task.done():
        lines.append(f"TTL cleanup: ✓ Active (every {CLEANUP_INTERVAL_HOURS}h)")
    else:
        lines.append(f"TTL cleanup: ✗ Not active")

    return "\n".join(lines)


# =============================================================================
# TTL Cleanup
# =============================================================================


async def _cleanup_expired_memories():
    """Periodically delete expired memories."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        try:
            table = get_table()
            now = datetime.now().isoformat()
            deleted = table.delete(f"expires_at IS NOT NULL AND expires_at < '{now}'")
            print(f"[memory-mcp] Cleaned expired memories at {now}", file=sys.stderr)
        except Exception as e:
            print(f"[memory-mcp] Cleanup error: {e}", file=sys.stderr)


# =============================================================================
# Server Entry Point
# =============================================================================


async def run_server():
    """Run the MCP server with database initialization and background tasks."""
    await init_database()
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_expired_memories())
    await mcp.run_stdio_async()


def main():
    """Entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
