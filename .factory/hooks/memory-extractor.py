#!/home/wb200/projects/memory-mcp/.venv/bin/python3
"""
Real-time Memory Extractor Hook for Droid - LanceDB Version

Automatically extracts and saves memory-worthy insights from Droid actions
using Gemini Flash as a selective judge.

Shares database with memory MCP server for seamless auto-save + retrieval.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import lancedb
import numpy as np
from lancedb.pydantic import LanceModel, Vector

# =============================================================================
# Configuration (matches MCP server - uses env vars)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Config:
    db_path: Path = Path(
        os.environ.get("LANCEDB_MEMORY_PATH", Path.home() / ".memory-mcp" / "lancedb-memory")
    )
    table_name: str = "memories"
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "1024"))
    embedding_provider: str = os.environ.get("EMBEDDING_PROVIDER", "ollama")  # ollama | google
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = "gemini-3-flash-preview"
    ttl_days: int = 365
    dedup_threshold: float = 0.90
    rate_limit_seconds: int = 30


CONFIG = Config()

LOG_PATH = Path.home() / ".factory" / "logs" / "memory-extractor.log"
RATE_LIMIT_FILE = Path.home() / ".factory" / "hooks" / ".last_extraction"

VALID_CATEGORIES = frozenset(
    {"PATTERN", "CONFIG", "DEBUG", "PERF", "PREF", "INSIGHT", "API", "AGENT"}
)


# =============================================================================
# LanceDB Schema (matches MCP server)
# =============================================================================


class Memory(LanceModel):
    id: str
    content: str
    vector: Vector(CONFIG.embedding_dim)  # type: ignore[valid-type]
    category: str
    tags: str  # JSON array as string
    project_id: str
    user_id: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None


# =============================================================================
# Gemini Client (Lazy Singleton)
# =============================================================================

_genai_client = None


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    secrets_path = Path.home() / ".secrets" / "GOOGLE_API_KEY"
    if secrets_path.exists():
        return secrets_path.read_text().strip()
    raise ValueError("GOOGLE_API_KEY not found")


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=_get_api_key())
    return _genai_client


# =============================================================================
# Database Layer
# =============================================================================

_db = None
_table = None


def get_db():
    global _db
    if _db is None:
        CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(str(CONFIG.db_path))
    return _db


def get_table():
    global _table
    if _table is None:
        db = get_db()
        try:
            _table = db.open_table(CONFIG.table_name)
        except Exception:
            _table = db.create_table(CONFIG.table_name, schema=Memory)
    return _table


def _normalize_git_url(url: str) -> str:
    """Normalize git URLs to canonical format: provider.com/owner/repo"""
    import re
    
    url = url.removesuffix('.git')
    
    # SSH format: git@github.com:owner/repo -> github.com/owner/repo
    ssh_match = re.match(r'git@([^:]+):(.+)', url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"
    
    # HTTPS format: https://github.com/owner/repo -> github.com/owner/repo
    https_match = re.match(r'https?://(.+)', url)
    if https_match:
        return https_match.group(1)
    
    return url


def get_project_id():
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _normalize_git_url(result.stdout.strip())
    except Exception:  # noqa: S110 - git may not be available
        pass
    return str(Path.cwd())


def now_iso():
    return datetime.now().isoformat()


def expires_iso():
    return (datetime.now() + timedelta(days=CONFIG.ttl_days)).isoformat()


def log(message: str, level: str = "INFO"):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    with LOG_PATH.open("a") as f:
        f.write(f"[{timestamp}] {level}: {message}\n")


def check_rate_limit():
    if not RATE_LIMIT_FILE.exists():
        return False
    try:
        last_time = float(RATE_LIMIT_FILE.read_text().strip())
        if time.time() - last_time < CONFIG.rate_limit_seconds:
            return True
    except (ValueError, OSError):
        pass
    return False


def update_rate_limit():
    RATE_LIMIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RATE_LIMIT_FILE.write_text(str(time.time()))


# =============================================================================
# Embedding (supports Ollama and Google with fallback chain)
# =============================================================================


def _compute_embedding_ollama(text: str) -> list[float] | None:
    """Generate embedding using Ollama (local, primary for qwen3-embedding)."""
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
                embedding = embedding[: CONFIG.embedding_dim]
            else:
                padding = np.zeros(CONFIG.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])

        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else embedding.tolist()
    except Exception as e:
        log(f"Ollama embedding error: {e}", "ERROR")
        return None


def _compute_embedding_google(text: str, task_type: str) -> list[float] | None:
    """Generate embedding using Google Genai API (fallback)."""
    try:
        from google.genai import types

        client = get_genai_client()
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=CONFIG.embedding_dim
            ),
        )
        embedding = np.array(response.embeddings[0].values)
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else embedding.tolist()
    except Exception as e:
        log(f"Google embedding error: {e}", "ERROR")
        return None


def compute_embedding(text: str, task_type: str = "SEMANTIC_SIMILARITY"):
    """Compute embedding with provider fallback chain."""
    provider = CONFIG.embedding_provider.lower()

    # Try Ollama first if configured as primary
    if provider == "ollama":
        result = _compute_embedding_ollama(text)
        if result:
            return result
        log("Ollama failed, falling back to Google", "WARNING")

    # Try Google
    result = _compute_embedding_google(text, task_type)
    if result:
        return result

    log("All embedding providers failed", "ERROR")
    return None


# =============================================================================
# Memory Operations
# =============================================================================


def read_recent_context(transcript_path: str, num_messages: int = 5) -> str:
    try:
        messages = []
        with Path(transcript_path).open() as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "message":
                        msg = entry.get("message", {})
                        role = msg.get("role", "unknown")
                        content = msg.get("content", [])
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", "")[:500])
                        if text_parts:
                            messages.append(f"[{role}]: {' '.join(text_parts)[:800]}")
                except json.JSONDecodeError:
                    continue
        return "\n".join(messages[-num_messages:])
    except Exception as e:
        log(f"Error reading transcript: {e}", "ERROR")
        return ""


def judge_memory_worthiness(
    tool_name: str, tool_input: dict, tool_response: dict, context: str
) -> dict | None:
    prompt = f"""Analyze this Droid action and determine if it contains a memory-worthy insight.

ACTION:
Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)[:2000]}
Result: {json.dumps(tool_response, indent=2)[:2000]}

Recent Context:
{context[:3000]}

CRITERIA (save ONLY if it clearly matches one of these):
- Bug fix with non-obvious cause/solution
- New coding pattern or architecture insight
- Configuration that took effort to get right
- Error resolution with reusable fix
- Performance optimization technique
- User preference explicitly stated

DO NOT save:
- Simple file reads or directory listings
- Trivial edits or formatting changes
- Commands that just check status
- Actions without learning value

If NOT worthy, respond exactly: {{"worthy": false}}

If worthy, respond with valid JSON:
{{
  "worthy": true,
  "category": "PATTERN|CONFIG|DEBUG|PERF|PREF|INSIGHT|API|AGENT",
  "content": "[CATEGORY] - [Concise insight in 1-2 sentences]. Context: [project/situation]. Rationale: [why this matters]",
  "tags": ["tag1", "tag2", "tag3"]
}}

Respond ONLY with JSON, no other text."""

    try:
        client = get_genai_client()
        response = client.models.generate_content(model=CONFIG.llm_model, contents=prompt)
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        result = json.loads(response_text)
        if result.get("worthy"):
            return result
        return None
    except Exception as e:
        log(f"Gemini judge error: {e}", "ERROR")
        return None


def check_similar_exists(content: str):
    embedding = compute_embedding(content)
    if embedding is None:
        return False, None, None

    try:
        table = get_table()
        results = table.search(embedding).metric("cosine").limit(1).to_list()
        if results:
            similarity = 1 - results[0]["_distance"]
            if similarity >= CONFIG.dedup_threshold:
                return True, similarity, results[0]["content"][:100]
        return False, similarity, None
    except Exception as e:
        log(f"Similarity check error: {e}", "ERROR")
        return False, None, None


def save_memory(content: str, category: str, tags: list, project_id: str) -> str | None:
    embedding = compute_embedding(content)
    if embedding is None:
        log(f"Failed to generate embedding for: {content[:100]}...", "ERROR")
        return None

    table = get_table()
    memory_id = uuid.uuid4().hex

    memory = Memory(
        id=memory_id,
        content=content,
        vector=embedding,
        category=category,
        tags=json.dumps(tags),
        project_id=project_id,
        created_at=now_iso(),
        updated_at=now_iso(),
        expires_at=expires_iso(),
    )

    try:
        table.add([memory.model_dump()])
        log(f"Saved memory {memory_id[:8]}...: [{category}] {content[:100]}...", "INFO")
        return memory_id
    except Exception as e:
        log(f"Database error: {e}", "ERROR")
        return None


# =============================================================================
# Main Hook Entry Point
# =============================================================================


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        log(f"Invalid JSON input: {e}", "ERROR")
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    tool_response = input_data.get("tool_response", {})
    transcript_path = input_data.get("transcript_path", "")
    cwd = input_data.get("cwd", str(Path.cwd()))

    allowed_tools = ["Edit", "Write", "Bash", "MultiEdit"]
    is_tiger_mcp = tool_name.startswith("mcp__tiger__")
    if tool_name not in allowed_tools and not is_tiger_mcp:
        sys.exit(0)

    if check_rate_limit():
        log(f"Skipping due to rate limit (tool: {tool_name})")
        sys.exit(0)

    context = read_recent_context(transcript_path) if transcript_path else ""

    log(f"Judging action: {tool_name}")
    result = judge_memory_worthiness(tool_name, tool_input, tool_response, context)

    if result is None:
        log(f"Action not memory-worthy: {tool_name}")
        sys.exit(0)

    similar_exists, similarity, existing_snippet = check_similar_exists(result["content"])
    if similar_exists:
        log(f"Skipping duplicate (similarity: {similarity:.2%}): {existing_snippet}...")
        sys.exit(0)

    memory_id = save_memory(
        content=result["content"],
        category=result["category"],
        tags=result.get("tags", []),
        project_id=cwd,
    )

    if memory_id:
        update_rate_limit()
        log(f"Auto-saved memory {memory_id[:8]}... from {tool_name}")

    sys.exit(0)


if __name__ == "__main__":
    main()
