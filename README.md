# Memory-MCP Server

**Version**: 3.8.0  
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
│                      MEMORY-MCP v3.8.0                           │
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
git clone https://github.com/wb200/memory-mcp.git
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
git clone https://github.com/wb200/memory-mcp.git
cd memory-mcp

# Using uv (recommended)
uv sync
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

The hook system enables **automatic memory extraction** from agent actions and **memory recall at session start** for context injection. This is optional but recommended for hands-free learning.

### Hooks Overview

| Hook Event | File | Trigger | Purpose |
|------------|------|---------|---------|
| **PostToolUse** | `.factory/hooks/memory-extractor.py` | After Edit/Write/Bash/MultiEdit | Auto-save memory-worthy insights |
| **SessionStart** | `.factory/hooks/session_start_recall.py` | On startup, `/resume`, `/clear`, compact | Inject memory context at session start |

### How They Work

**1. memory-extractor.py (PostToolUse)**

1. **Triggers** after tool executions (Edit, Write, Bash, MultiEdit, MCP tiger tools)
2. **Analyzes** the action using an LLM judge (Gemini Flash)
3. **Extracts** category, content, and tags if memory-worthy
4. **Checks** for duplicates (90% similarity threshold)
5. **Saves** automatically to the same LanceDB database

**2. session_start_recall.py (SessionStart)**

1. **Triggers** on session events: `startup`, `resume` (`/resume`), `clear` (`/clear`), `compact`
2. **Retrieves** memories for the current project from LanceDB
3. **Generates** a "Project Highlights" summary using Gemini
4. **Outputs** JSON with `additionalContext` field (injected into agent context)

### Installation

**For Factory/Droid users:**

**⭐ Recommended: Global Hooks Installation**

To enable memory capture across **all projects** (not just memory-mcp):

1. **Copy hooks to global Factory hooks directory**:
```bash
cp /path/to/memory-mcp/.factory/hooks/memory-extractor.py ~/.factory/hooks/
cp /path/to/memory-mcp/.factory/hooks/session_start_recall.py ~/.factory/hooks/
```

2. **Configure in `~/.factory/settings.json`**:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|Bash|MultiEdit|mcp__tiger__.*",
        "hooks": [
          {
            "type": "command",
            "command": "~/.factory/hooks/memory-extractor.py",
            "timeout": 30
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "~/.factory/hooks/session_start_recall.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**Why Global Installation?**
- ✅ **Works everywhere** - Captures memories from all your projects
- ✅ **No configuration per project** - Set up once, works forever
- ✅ **Automatic project tagging** - Each memory tagged with project_id (git remote or cwd)
- ✅ **Centralized updates** - Update hooks in one place

**Alternative: Project-Specific Hooks** (not recommended unless you only want memories from memory-mcp project):
- Hooks are also in `.factory/hooks/` within the project
- Use absolute path: `/path/to/memory-mcp/.factory/hooks/memory-extractor.py`
- Only triggers when working inside memory-mcp directory

### Factory/Droid Hook Events Reference

| Event | Matchers | When It Fires |
|-------|----------|---------------|
| `SessionStart` | `startup`, `resume`, `clear`, `compact` | New session, `/resume`, `/clear`, or context compaction |
| `PostToolUse` | Tool names (regex) | After any matched tool executes successfully |
| `PreToolUse` | Tool names (regex) | Before tool execution (can block/modify) |
| `UserPromptSubmit` | N/A | When user submits a prompt (NOT on slash commands) |
| `Stop` | N/A | When agent finishes responding |
| `Notification` | N/A | When agent sends notifications |

> **Note**: `/resume` triggers `SessionStart` with `resume` matcher, NOT `UserPromptSubmit`. This is a common configuration mistake.

### Project-Based Hooks

Hooks are stored in the project folder (version controlled):
```
memory-mcp/
├── .factory/
│   └── hooks/
│       ├── memory-extractor.py      # Auto-save after tool use
│       └── session_start_recall.py  # Memory recall at session start
└── server.py
```

This approach:
- Version controls hooks with the project
- Makes configuration portable (no hardcoded paths)
- Enables team sharing via git

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

## Project ID Migration

If you have existing memories from before v3.7.0, you may have duplicate project IDs in different formats:
- `git@github.com:owner/repo.git` (SSH)
- `https://github.com/owner/repo.git` (HTTPS)
- `/home/user/projects/repo` (path)

Run the migration script to normalize all git URLs:

```bash
# Preview changes (recommended first)
uv run migrate_project_ids.py --dry-run

# Apply migration
uv run migrate_project_ids.py
```

**Example output:**
```
2 memories: git@github.com:wb200/memory-mcp.git
     -> github.com/wb200/memory-mcp

✓ Migration complete! Updated 2 memories
```

After migration, all git URLs use canonical format: `github.com/owner/repo`

**Note:** Path-based project IDs (e.g., `/home/user/projects/repo`) are preserved for backward compatibility with legacy memories.

---

## Testing

### Run All Tests

```bash
cd memory-mcp
uv run pytest -v
```

### MCP Inspector Testing

Test the server interactively with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
# Start the server in a separate terminal
cd memory-mcp
uv run python server.py

# In another terminal, run the inspector
npx @modelcontextprotocol/inspector http://localhost:3000
```

Or test via stdio:
```bash
cd memory-mcp
uv run python -c "
import asyncio
from server import mcp
async def test():
    result = await mcp.tools['memory_health']()
    print(result)
asyncio.run(test())
"
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
30 passed in ~23s
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

## Web Viewer

A beautiful, always-on browser interface for browsing your memory database with advanced filtering and search capabilities.

### Features

- 🎨 **Color-coded categories** - Visual distinction between PATTERN, CONFIG, DEBUG, etc.
- 🔍 **Dual search modes** - Filter by project AND keyword simultaneously
- 📊 **Pagination** - Smooth navigation through large memory sets
- 🏷️ **Tag display** - See all tags and metadata at a glance
- ⚡ **Real-time updates** - Always reflects current database state
- 🌐 **Project filtering** - Quickly isolate memories from specific codebases

### Quick Start (Manual)

```bash
# Install Flask (one-time)
uv add --group optional flask

# Run the viewer
uv run memory-viewer

# Access at http://localhost:5000
```

### **⭐ Recommended: Always-On Service**

Run the memory viewer as a persistent background service that:
- ✅ **Survives terminal closures** - No more accidentally killing the viewer
- ✅ **Auto-starts on boot** - Available immediately after system restart  
- ✅ **Auto-restarts on crash** - Built-in systemd recovery
- ✅ **Zero maintenance** - Set it and forget it
- ✅ **Integrated logging** - All output captured in systemd journal

**One-command setup:**

```bash
# Install and start the service
./install-service.sh

# Enable 24/7 always-on mode (survives logout/reboot)
loginctl enable-linger $USER
```

**That's it!** Access your memories anytime at **http://localhost:5000** 🚀

### Service Management

```bash
# Check status and uptime
systemctl --user status memory-viewer

# Restart after code updates
systemctl --user restart memory-viewer

# View real-time logs
journalctl --user -u memory-viewer -f

# Stop the service
systemctl --user stop memory-viewer

# Uninstall completely
./uninstall-service.sh
```

### Performance Impact

The always-on service is lightweight and designed for 24/7 operation:

| Resource | Usage | Notes |
|----------|-------|-------|
| **Memory** | ~130MB | Flask app + Python runtime |
| **CPU** | <1% idle | Only active during page loads |
| **Disk** | Negligible | Reads from existing LanceDB |
| **Network** | Local only | Binds to 127.0.0.1:5000 |

### Advanced Configuration

See [SERVICE.md](SERVICE.md) for:
- Custom port configuration
- Production WSGI server setup (Gunicorn/uWSGI)
- Troubleshooting service issues
- Log rotation and monitoring
- Security considerations

### Why Use the Service vs Manual?

| Scenario | Manual Run | Always-On Service |
|----------|-----------|-------------------|
| Quick check | ✅ Perfect | 🔶 Overkill |
| Daily use | 🔶 Annoying to restart | ✅ Always ready |
| Shared machine | ❌ Stops on logout | ✅ Keeps running |
| Development workflow | 🔶 Tab clutter | ✅ Clean workspace |
| Team access | ❌ Unreliable | ✅ Guaranteed uptime |

**Bottom line:** If you check memories more than once a day, the service pays for itself in convenience.

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

**For PostToolUse hooks (memory-extractor):**

1. Check logs: `tail -f ~/.factory/logs/memory-extractor.log`
2. Verify settings.json has correct `$FACTORY_PROJECT_DIR` path
3. Ensure you're using tools that match the hook matcher (Edit|Write|Bash|MultiEdit)

**For SessionStart hooks (session_start_recall):**

1. Check debug log: `cat memory-mcp/.factory/hooks/hook-debug.log`
2. The hook triggers on: startup, `/resume`, `/clear`, compact
3. Verify settings.json has `SessionStart` event with matcher `startup|resume|clear|compact`
4. Ensure hook outputs valid JSON with `hookSpecificOutput.additionalContext` field

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
- **v3.8.0** - R.A.S.I.R. code quality improvements: fixed SQL LIKE wildcard injection, added secret file permissions validation, extracted shared models (`models.py`) and utils (`utils.py`) for DRY, consolidated duplicate embedding normalization and git URL normalization, added Config validation with `__post_init__`, improved type hints, fixed all ruff violations. Documented global hooks installation for cross-project memory capture.
- **v3.7.0** - Project ID normalization: git URLs now use canonical format (github.com/owner/repo), migration script for existing memories, eliminates duplicate project fragmentation
- **v3.6.0** - Always-on web viewer with systemd service support, linger mode for 24/7 availability, legacy project ID fallback for backward compatibility
- **v3.5.0** - Renamed MCP server from `droid-memory` to `memory` (agent-agnostic), fixed `hookEventName` camelCase bug, silent context injection (removed verbose stderr)
- **v3.4.0** - Fixed hook configuration: SessionStart event (not UserPromptSubmit) for memory recall on `/resume`, JSON output format for context injection, comprehensive hook documentation
- **v3.3.0** - Project-based hooks, dual hooks (auto-save + session start recall)
- **v3.2.0** - SOTA reranker (mxbai-reranker-base-v2), path updates
- **v3.1.0** - Tantivy FTS, embedding cache, TTL cleanup
- **v3.0.0** - Ollama integration, 1024-dim embeddings
- **v2.0.0** - Hook system, LLM judge
- **v1.0.0** - Initial release
