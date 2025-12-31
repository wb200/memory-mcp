# Memory-MCP Server

**Version**: 3.2.0  
**Status**: Production-Ready  
**License**: MIT

A state-of-the-art persistent memory system for AI agents using hybrid search (vector embeddings + BM25 FTS), neural reranking, and optional LLM-driven automated memory extraction.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [MCP Tools Reference](#mcp-tools-reference)
- [Hook System (Auto-Save)](#hook-system-auto-save)
- [Search Technology](#search-technology)
- [Memory Lifecycle](#memory-lifecycle)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [FAQ](#faq)

---

## Overview

Memory-MCP is a production-grade persistent memory system for AI coding agents (Claude Code, Cursor, Windsurf, custom agents, etc.) that stores and retrieves valuable insights across sessions. It combines semantic vector search with keyword matching (BM25) for optimal retrieval accuracy.

### What Problems Does This Solve?

- **Lost Knowledge**: Valuable insights from debugging sessions, configurations, and patterns are forgotten between sessions
- **Context Switching**: Hard to recall what worked in previous projects
- **Duplicate Effort**: Solving the same problems repeatedly
- **Scattered Notes**: Knowledge lives in different formats across different projects

### How It Works

```
Save Memory → Embedding Generation → Duplicate Check → Store in LanceDB
                    ↓
Recall Memory → Hybrid Search (Vector + BM25) → RRF Fusion → Neural Rerank → Results
```

1. **Intelligent Storage**: Stores insights with 1024-dimensional semantic embeddings
2. **Hybrid Retrieval**: Searches using both semantic similarity AND keyword matching
3. **Neural Reranking**: CrossEncoder re-ranks results for maximum relevance
4. **Optional Auto-Save**: Hook analyzes agent actions and extracts memories automatically

---

## Key Features

| Feature | Description |
|---------|-------------|
| **7 MCP Tools** | Full CRUD operations + stats + health monitoring |
| **Hybrid Search** | Vector (70%) + BM25 FTS (30%) with RRF fusion |
| **Neural Reranking** | CrossEncoder (mxbai-reranker-base-v2, BEIR SOTA) |
| **Local Embeddings** | Ollama support for privacy and speed |
| **GPU Acceleration** | Works with any CUDA-capable GPU |
| **Duplicate Prevention** | 90% similarity threshold prevents redundant saves |
| **TTL Management** | 365-day expiry with automatic cleanup |
| **Fallback Chain** | Ollama → Google → Hash (always available) |
| **Project Scoping** | Search across all projects or project-specific |
| **Auto-Save Hook** | Optional PostToolUse hook for automatic extraction |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY-MCP v3.2.0                           │
└─────────────────────────────────────────────────────────────────┘

   Agent/User
       │
       ├──────────────────────────────────────┐
       │                                      │
   ┌───▼────┐                          ┌──────▼───────┐
   │ Manual │                          │  Auto-Save   │
   │ MCP    │                          │  Hook        │
   │ Tools  │                          │  (Optional)  │
   └───┬────┘                          └──────┬───────┘
       │                                      │
       │              ┌───────────────────────▼──────────┐
       │              │ LLM Judge (Gemini Flash)         │
       │              │ - Determines worthiness          │
       │              │ - Extracts category & tags       │
       │              └───────────────────────┬──────────┘
       │                                      │
       └──────────────┬───────────────────────┘
                      │
           ┌──────────▼──────────┐
           │ Embedding Generation │
           │ Primary: Ollama      │
           │ Fallback: Google     │
           │ Last: Hash-based     │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │ Duplicate Check     │
           │ (90% similarity)    │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │      LanceDB        │
           │ - 1024-dim vectors  │
           │ - BM25 FTS index    │
           │ - TTL (365 days)    │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  Query Pipeline     │
           │ 1. Vector Search    │
           │ 2. FTS Search       │
           │ 3. RRF Fusion       │
           │ 4. Neural Rerank    │
           │ 5. TTL Filter       │
           └─────────────────────┘
```

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/memory-mcp.git
cd memory-mcp
uv sync

# Configure MCP client (add to your mcp.json)
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/memory-mcp/.venv/bin/python",
      "args": ["/path/to/memory-mcp/server.py"],
      "env": {
        "GOOGLE_API_KEY": "your-api-key"
      }
    }
  }
}

# Test it works
memory_health()  # Should show system status
memory_save(content="Test memory", category="DEBUG")
memory_recall(query="test")
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Ollama (optional, for local embeddings) OR Google API key

### Step 1: Clone and Setup

```bash
git clone https://github.com/yourusername/memory-mcp.git
cd memory-mcp

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Install Ollama (Recommended for Privacy)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the embedding model
ollama pull qwen3-embedding:0.6b

# Start Ollama server
ollama serve
```

**With GPU (systemd service)**:

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama LLM Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_NUM_GPU=all"
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now ollama
```

### Step 3: Configure MCP Client

Add to your MCP configuration file:

**Claude Code / Factory** (`~/.factory/mcp.json`):
```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "/path/to/memory-mcp/.venv/bin/python",
      "args": ["/path/to/memory-mcp/server.py"],
      "env": {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "EMBEDDING_DIM": "1024",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
        "LANCEDB_MEMORY_PATH": "/home/youruser/.memory-mcp/lancedb-memory"
      }
    }
  }
}
```

**Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/memory-mcp/.venv/bin/python",
      "args": ["/path/to/memory-mcp/server.py"],
      "env": {
        "GOOGLE_API_KEY": "your-api-key"
      }
    }
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LANCEDB_MEMORY_PATH` | `~/.memory-mcp/lancedb-memory` | Database location |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `google` |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |
| `EMBEDDING_DIM` | `1024` | Embedding dimensions |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GOOGLE_API_KEY` | - | Google Gemini API key |

### Server Configuration

These are set in the `Config` class in `server.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | `gemini-3-flash-preview` | LLM for summarization |
| `ttl_days` | `365` | Memory time-to-live |
| `dedup_threshold` | `0.90` | Duplicate similarity threshold |
| `fts_weight` | `0.3` | FTS weight in RRF fusion |
| `default_limit` | `5` | Default results per query |
| `max_limit` | `50` | Maximum results per query |

---

## MCP Tools Reference

### Overview

| Tool | Description | Read-Only |
|------|-------------|-----------|
| `memory_save` | Save a memory with semantic embedding | No |
| `memory_recall` | Search across ALL projects | Yes |
| `memory_recall_project` | Search in CURRENT project only | Yes |
| `memory_delete` | Delete a memory by ID | No |
| `memory_update` | Update an existing memory | No |
| `memory_stats` | Get statistics by category/project | Yes |
| `memory_health` | Get system health status | Yes |

### memory_save

Save a new memory with automatic embedding and duplicate detection.

**Parameters:**
- `content` (required): Memory content
- `category`: One of `PATTERN`, `CONFIG`, `DEBUG`, `PERF`, `PREF`, `INSIGHT`, `API`, `AGENT`
- `tags`: List of tags for categorization
- `summarize`: Use LLM to summarize verbose content

**Example:**
```python
memory_save(
    content="[DEBUG] - RuntimeError: CUDA out of memory solved with gradient_checkpointing=True. Context: Fine-tuning transformer. Rationale: Trades compute for memory.",
    category="DEBUG",
    tags=["pytorch", "cuda", "memory"]
)
# Response: Saved (ID: a1b2c3d4..., DEBUG)
#           Tags: ['pytorch', 'cuda', 'memory']
```

### memory_recall

Search across all projects using hybrid search.

**Parameters:**
- `query` (required): Search query (semantic + keywords)
- `category`: Optional category filter
- `limit`: Max results (default 5, max 50)

**Example:**
```python
memory_recall(
    query="CUDA out of memory pytorch",
    category="DEBUG",
    limit=3
)
# Response: Found 2 memories (global, hybrid + neural rerank):
#
# [1] DEBUG (ID: a1b2c3d4...)
#     RuntimeError: CUDA out of memory solved with gradient_checkpointing...
#     Tags: pytorch, cuda, memory
#     Similarity: 94%
```

### memory_recall_project

Same as `memory_recall` but scoped to current project only.

### memory_delete

Delete a memory by full or partial ID.

**Example:**
```python
memory_delete(memory_id="a1b2c3d4")  # Full ID
memory_delete(memory_id="a1b2")      # Partial prefix (must be unambiguous)
```

### memory_update

Update content, category, or tags of an existing memory.

**Example:**
```python
memory_update(
    memory_id="a1b2c3d4",
    content="Updated content here",
    category="CONFIG",
    tags=["new", "tags"]
)
```

### memory_stats

Get memory statistics.

**Example:**
```python
memory_stats()
# Response: === Memory Statistics (LanceDB) ===
#           Total: 47 memories
#           Database: 185.7 KB
#           ...
#           By Category:
#             CONFIG: 15
#             DEBUG: 12
#             ...
```

### memory_health

Get system health and configuration status.

**Example:**
```python
memory_health()
# Response: === Memory Health Status ===
#           Total memories: 47
#           Database size: 185.7 KB
#           FTS index: ✓ BM25 enabled
#           Vector index: ✓ IVF-PQ
#           TTL: 365 days
#           TTL cleanup: ✓ Active (every 24h)
```

---

## Hook System (Auto-Save)

The hook system enables **automatic memory extraction** from agent actions. This is optional but recommended for hands-free learning.

### How It Works

The `hooks/memory-extractor.py` hook:

1. **Triggers** after tool executions (Edit, Write, Bash, MultiEdit)
2. **Analyzes** the action using an LLM judge (Gemini Flash)
3. **Extracts** category, content, and tags if memory-worthy
4. **Checks** for duplicates (90% similarity threshold)
5. **Saves** automatically to the same LanceDB database

### Installation

**For Factory/Droid users:**

1. **Update the shebang** in `hooks/memory-extractor.py` to point to your venv:
```bash
# Edit the first line of hooks/memory-extractor.py to:
#!/path/to/memory-mcp/.venv/bin/python3
```
> **Important**: The hook requires dependencies (lancedb, numpy, etc.) from the project's virtual environment. Using `#!/usr/bin/env python3` will fail with `ModuleNotFoundError` unless those packages are installed system-wide.

2. Symlink or copy the hook:
```bash
ln -s /path/to/memory-mcp/hooks/memory-extractor.py ~/.factory/hooks/
```

3. Configure in `~/.factory/settings.json`:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|Bash|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "~/.factory/hooks/memory-extractor.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

4. Make executable:
```bash
chmod +x ~/.factory/hooks/memory-extractor.py
```

### LLM Judge Criteria

The judge saves memories ONLY when they match:

- Bug fix with non-obvious cause/solution
- New coding pattern or architecture insight
- Configuration that took effort
- Error resolution with reusable fix
- Performance optimization
- User preference explicitly stated

It SKIPS:
- Simple file reads/listings
- Trivial edits or formatting
- Status checks
- Actions without learning value

### Memory Categories

| Category | When to Use |
|----------|-------------|
| `PATTERN` | Coding patterns, architectures, design decisions |
| `CONFIG` | Tool configurations, environment settings |
| `DEBUG` | Error resolutions, debugging techniques |
| `PERF` | Performance optimizations |
| `PREF` | User preferences, coding style |
| `INSIGHT` | Cross-project learnings |
| `API` | LLM/external API usage patterns |
| `AGENT` | Agent design patterns, workflows |

### Hook Limits

| Setting | Value | Description |
|---------|-------|-------------|
| Rate Limit | 30s | Minimum time between extractions |
| Timeout | 30s | Max execution time |
| Context | 5 messages | Recent transcript context |

### Hook Logs

Monitor hook activity:
```bash
tail -f ~/.factory/logs/memory-extractor.log
```

---

## Search Technology

### Hybrid Search Pipeline

Memory-MCP uses **true hybrid search** combining multiple retrieval methods:

```
Query
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 ▼
┌─────────────────┐       ┌─────────────────┐
│ Vector Search   │       │ FTS Search      │
│ (70% weight)    │       │ (30% weight)    │
│ Cosine similarity│      │ BM25 keywords   │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │ RRF Fusion       │
          │ 1/(k + rank)     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Neural Reranking │
          │ CrossEncoder     │
          │ mxbai-reranker   │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ TTL Filter       │
          │ expires_at > NOW │
          └────────┬─────────┘
                   │
                   ▼
              Top K Results
```

### Components

1. **Vector Search** (70% weight)
   - 1024-dimensional embeddings (qwen3-embedding or Gemini)
   - Cosine similarity
   - Captures semantic meaning

2. **BM25 FTS** (30% weight)
   - Tantivy-based full-text search
   - TF-IDF keyword matching
   - Catches exact phrases and rare terms

3. **RRF Fusion**
   - Reciprocal Rank Fusion combines results
   - Weighted scoring prevents either method dominating

4. **Neural Reranking**
   - CrossEncoder: `mixedbread-ai/mxbai-reranker-base-v2`
   - BEIR benchmark SOTA (reinforcement learning trained)
   - Improves relevance 10-15%

### Performance

| Operation | Time |
|-----------|------|
| Embedding (GPU) | ~10ms |
| Embedding (CPU) | ~30-50ms |
| Vector Search | 20-30ms |
| FTS Search | 2-5ms |
| RRF Fusion | <1ms |
| Neural Rerank | 20-50ms |
| **Total Recall** | **50-130ms** |

---

## Memory Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    CREATION                                  │
├─────────────────────────────────────────────────────────────┤
│ Source: memory_save() or Auto-Save Hook                     │
│    ↓                                                        │
│ Embedding Generation (Ollama → Google → Hash fallback)      │
│    ↓                                                        │
│ Duplicate Check (90% similarity threshold)                  │
│    ↓                                                        │
│ Store to LanceDB with TTL (365 days)                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    ACTIVE (365 days)                         │
├─────────────────────────────────────────────────────────────┤
│ Available for recall                                         │
│ TTL checked on each query                                    │
│ Can be updated via memory_update()                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    EXPIRATION                                │
├─────────────────────────────────────────────────────────────┤
│ Background cleanup runs every 24 hours                       │
│ Expired memories deleted automatically                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Basic Usage

```python
# Save a debugging insight
memory_save(
    content="[DEBUG] - Python multiprocessing on macOS requires 'spawn' start method. Use: mp.set_start_method('spawn'). Context: ML training. Rationale: 'fork' causes CUDA issues.",
    category="DEBUG",
    tags=["python", "multiprocessing", "macos", "cuda"]
)

# Search for it later
memory_recall(query="multiprocessing macos cuda")

# Check system health
memory_health()
```

### With Category Filtering

```python
# Only search CONFIG memories
memory_recall(query="database connection", category="CONFIG")
```

### Project-Specific Search

```python
# Search only in current project
memory_recall_project(query="authentication pattern")
```

### Using Summarization

```python
# Let LLM summarize verbose content
memory_save(
    content="Very long error log with stack trace...",
    category="DEBUG",
    summarize=True  # LLM extracts key insight
)
```

### Auto-Save Example

With the hook configured, after you fix a bug:

```bash
# You run: Edit file to fix ImportError
# Hook automatically saves:
# "[DEBUG] - ImportError: No module named 'xyz' fixed by adding to PYTHONPATH. Context: Package structure issue."
```

---

## Testing

### Run All Tests

```bash
cd memory-mcp
uv run pytest -v
```

### Test Database

Tests use an isolated database separate from production:

| Variable | Default Location |
|----------|------------------|
| `LANCEDB_MEMORY_PATH` (production) | `~/.memory-mcp/lancedb-memory` |
| `LANCEDB_MEMORY_TEST_PATH` (tests) | `./lancedb-memory-test` (project folder) |

The test database is automatically created and wiped before each test run. It's excluded from git via `.gitignore`.

### Current Test Results

```
30 passed in 37s
```

### Test Suites

| Suite | Tests | Coverage |
|-------|-------|----------|
| TestMemorySave | 7 | Save validation, deduplication |
| TestMemoryRecall | 6 | Search, filtering, project scope |
| TestMemoryUpdate | 3 | Update operations |
| TestMemoryDelete | 3 | Delete by ID, partial match |
| TestMemoryStats | 1 | Statistics |
| TestEmbeddings | 2 | Generation, similarity |
| TestSummarization | 1 | LLM summarization |
| TestConcurrency | 3 | Thread safety |
| TestFullLifecycle | 1 | End-to-end CRUD |
| TestHookIntegration | 2 | Hook configuration |
| TestMCPConfig | 1 | Config validation |

---

## Troubleshooting

### "GOOGLE_API_KEY not found"

```bash
export GOOGLE_API_KEY="your-api-key"
# Or add to MCP config env section
```

### "Ollama connection refused"

```bash
# Start Ollama
ollama serve

# Or check systemd
sudo systemctl status ollama
```

### "Duplicate detected" too often

Lower the threshold in server.py:
```python
dedup_threshold: float = 0.85  # Try 85% instead of 90%
```

### Hook not triggering

1. Check permissions: `chmod +x ~/.factory/hooks/memory-extractor.py`
2. Check logs: `tail -f ~/.factory/logs/memory-extractor.log`
3. Verify settings.json configuration

### Embedding dimension mismatch

Reset database (will lose existing memories):
```bash
rm -rf ~/.memory-mcp/lancedb-memory
```

### Health Check

Always start troubleshooting with:
```python
memory_health()
```

---

## API Reference

### Memory Schema

```python
class Memory:
    id: str              # UUID
    content: str         # Memory text (FTS indexed)
    vector: Vector(1024) # Semantic embedding
    category: str        # PATTERN|CONFIG|DEBUG|PERF|PREF|INSIGHT|API|AGENT
    tags: str           # JSON array: '["tag1", "tag2"]'
    project_id: str     # Git remote URL or cwd
    user_id: str | None # Optional
    created_at: str     # ISO timestamp
    updated_at: str     # ISO timestamp
    expires_at: str     # ISO timestamp (TTL)
```

### Config Schema

```python
@dataclass(frozen=True, slots=True)
class Config:
    db_path: Path = Path.home() / ".memory-mcp" / "lancedb-memory"
    table_name: str = "memories"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024
    embedding_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemini-3-flash-preview"
    ttl_days: int = 365
    dedup_threshold: float = 0.90
    fts_weight: float = 0.3
    default_limit: int = 5
    max_limit: int = 50
```

---

## FAQ

**Q: Can I use this without Ollama?**  
A: Yes, set `EMBEDDING_PROVIDER=google` and provide `GOOGLE_API_KEY`.

**Q: Is GPU required?**  
A: No, but recommended. CPU embeddings are ~3x slower.

**Q: What happens if all embedding providers fail?**  
A: Hash-based fallback ensures saves always work (reduced semantic quality).

**Q: How do I backup my memories?**  
A: Copy `~/.memory-mcp/lancedb-memory/` directory.

**Q: Can multiple agents share the same database?**  
A: Yes, LanceDB supports concurrent access.

**Q: Is the hook required?**  
A: No, it's optional. You can use `memory_save()` manually.

---

## Contributing

1. Fork the repository
2. Run tests: `uv run pytest -v`
3. Format code: `uv run ruff format .`
4. Lint: `uv run ruff check .`
5. Submit PR

---

## License

MIT License - See LICENSE file.

---

## Acknowledgments

- [LanceDB](https://lancedb.com/) - Vector database
- [Tantivy](https://github.com/quickwit-oss/tantivy) - Full-text search
- [Sentence-Transformers](https://sbert.net/) - CrossEncoder reranking
- [Ollama](https://ollama.ai/) - Local embeddings
- [Google Gemini](https://ai.google.dev/) - LLM judge & embeddings
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

**Version History**:
- **v3.2.0** - SOTA reranker (mxbai-reranker-base-v2), path updates
- **v3.1.0** - Tantivy FTS, embedding cache, TTL cleanup
- **v3.0.0** - Ollama integration, 1024-dim embeddings
- **v2.0.0** - Hook system, LLM judge
- **v1.0.0** - Initial release
