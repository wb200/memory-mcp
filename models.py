"""Shared data models for memory-mcp."""

import os

from lancedb.pydantic import LanceModel, Vector

# Configuration
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))


class Memory(LanceModel):
    """Memory table schema for LanceDB.

    IMPORTANT: Any changes to this schema require migration of existing data.
    Dynamic Vector dimension is based on CONFIG.embedding_dim.
    """

    id: str  # UUID string - thread-safe, no race conditions
    content: str  # Indexed for FTS
    vector: Vector(EMBEDDING_DIM)  # type: ignore[valid-type] - Dynamic dimension from config
    category: str
    tags: str  # JSON array as string
    project_id: str
    user_id: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None
