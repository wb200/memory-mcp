#!/home/wb200/projects/memory-mcp/.venv/bin/python3
"""
Session Start Memory Recall Hook for Droid

Automatically recalls project-specific memories when starting a session,
generating a "Project Highlights" summary using Gemini.

This hook:
1. Identifies the current project
2. Recalls the 10 most recent memories
3. Generates an AI summary of ALL project memories
4. Outputs context for the agent to use

Output appears in transcript, providing context without manual prompting.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import lancedb
import numpy as np
from lancedb.pydantic import LanceModel, Vector

# =============================================================================
# Configuration (matches MCP server)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Config:
    db_path: Path = Path(
        os.environ.get("LANCEDB_MEMORY_PATH", Path.home() / ".memory-mcp" / "lancedb-memory")
    )
    table_name: str = "memories"
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "1024"))
    embedding_provider: str = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = "gemini-3-flash-preview"
    summary_cache_hours: int = 1  # Cache summary for 1 hour


CONFIG = Config()

CACHE_DIR = Path.home() / ".factory" / "hooks" / ".memory_cache"
SUMMARY_CACHE_FILE = CACHE_DIR / ".project_summary_cache"


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
    return None  # Will handle missing key gracefully


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = _get_api_key()
        if api_key:
            from google import genai
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client


# =============================================================================
# Database Layer
# =============================================================================


_db = None
_table = None


def get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(str(CONFIG.db_path))
    return _db


def get_table():
    global _table
    if _table is None:
        db = get_db()
        try:
            _table = db.open_table(CONFIG.table_name)
        except Exception:
            return None
    return _table


# =============================================================================
# Project Detection
# =============================================================================


def get_project_id() -> str:
    """Get current project identifier from git or cwd."""
    # Use FACTORY_PROJECT_DIR if available, otherwise use script's directory
    project_dir = os.environ.get("FACTORY_PROJECT_DIR") or str(Path(__file__).parent)
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=project_dir,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return project_dir


def get_project_name() -> str:
    """Get human-readable project name."""
    project_id = get_project_id()
    if "github.com" in project_id:
        # Extract owner/repo from URL
        parts = project_id.split("/")
        if len(parts) >= 2:
            return parts[-1].replace(".git", "")
    # Fallback to script directory name, not cwd
    project_dir = os.environ.get("FACTORY_PROJECT_DIR") or str(Path(__file__).parent)
    return Path(project_dir).name


# =============================================================================
# Memory Retrieval
# =============================================================================


def get_project_memories(project_id: str, limit: int = 100) -> list[dict]:
    """Get all memories for current project."""
    table = get_table()
    if table is None:
        return []

    # Get possible project identifiers (git remote or path)
    project_dir = os.environ.get("FACTORY_PROJECT_DIR") or str(Path(__file__).parent)
    project_ids = {project_id}
    
    # If using git remote, also check for path-based project_id (legacy memories)
    if "github.com" in project_id:
        project_ids.add(project_dir)

    try:
        # Use to_arrow() to get data as PyArrow table, then convert to list of dicts
        arrow_table = table.to_arrow()
        column_names = arrow_table.column_names
        column_data = arrow_table.to_pydict()
        all_memories = [
            {col: column_data[col][i] for col in column_names}
            for i in range(len(column_data.get(column_names[0], [])))
        ]
        # Match memories by any of the possible project_ids
        project_memories = [
            m for m in all_memories
            if m.get("project_id") in project_ids
        ]
        # Sort by created_at descending
        project_memories.sort(
            key=lambda m: m.get("created_at", ""), reverse=True
        )
        return project_memories[:limit]
    except Exception as e:
        print(f"[session-start-recall] Error fetching memories: {e}", file=sys.stderr)
        return []


def get_recent_memories(project_id: str, limit: int = 10) -> list[dict]:
    """Get N most recent memories for the project."""
    all_mem = get_project_memories(project_id, limit=limit)
    return all_mem[:limit]


# =============================================================================
# Summary Generation
# =============================================================================


def generate_project_summary(memories: list[dict], project_name: str) -> str:
    """Use Gemini to generate a concise summary of project memories."""
    if not memories:
        return "No memories found for this project yet."

    client = get_genai_client()
    if client is None:
        return "[Summary unavailable: No GOOGLE_API_KEY configured]"

    # Prepare memory content for summarization
    memory_texts = []
    for i, m in enumerate(memories, 1):
        cat = m.get("category", "UNKNOWN")
        content = m.get("content", "")
        tags = m.get("tags", "[]")
        memory_texts.append(f"[{i}] [{cat}] {content} (tags: {tags})")

    memories_block = "\n".join(memory_texts)

    prompt = f"""You are a memory assistant. Analyze these project memories and create a concise "Project Highlights" summary.

Project: {project_name}
Total memories: {len(memories)}

MEMORIES:
{memories_block}

Please create a summary with these sections (use markdown):

## Architecture
Key architectural patterns, design decisions, code structure

## Tech Stack
Languages, frameworks, libraries, tools used

## Setup & Configuration
Installation steps, environment variables, API keys locations, configuration files

## Common Patterns
Frequently used code patterns, conventions, best practices

## Debug History
Issues encountered and how they were fixed

## Quick Reference
Most important commands, shortcuts, or configurations to remember

Keep each section concise - focus on actionable information. Use bullet points.
Format for easy scanning by an AI agent.
"""

    try:
        response = client.models.generate_content(
            model=CONFIG.llm_model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[session-start-recall] Summary generation failed: {e}", file=sys.stderr)
        return f"[Summary unavailable: {e}]"


def get_cached_summary(project_id: str) -> tuple[str, datetime | None]:
    """Get cached summary and its timestamp."""
    if not SUMMARY_CACHE_FILE.exists():
        return None, None

    try:
        cache_data = json.loads(SUMMARY_CACHE_FILE.read_text())
        if cache_data.get("project_id") == project_id:
            cached_at = datetime.fromisoformat(cache_data.get("cached_at", ""))
            return cache_data.get("summary"), cached_at
    except Exception:
        pass

    return None, None


def save_cached_summary(project_id: str, summary: str):
    """Cache the summary with timestamp."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "project_id": project_id,
            "cached_at": datetime.now().isoformat(),
            "summary": summary,
        }
        SUMMARY_CACHE_FILE.write_text(json.dumps(cache_data))
    except Exception as e:
        print(f"[session-start-recall] Cache write failed: {e}", file=sys.stderr)


# =============================================================================
# Output Formatting
# =============================================================================


def format_recent_memories(memories: list[dict]) -> str:
    """Format recent memories for display."""
    if not memories:
        return "No memories yet."

    lines = ["=" * 60, "RECENT MEMORIES (Last 10)", "=" * 60, ""]

    for m in memories:
        cat = m.get("category", "UNKNOWN")[:1].upper()  # First letter
        content = m.get("content", "")
        tags = json.loads(m.get("tags", "[]"))
        created = m.get("created_at", "")[:19].replace("T", " ")

        # Truncate long content
        if len(content) > 150:
            content = content[:150] + "..."

        lines.append(f"[{cat}] {content}")
        if tags:
            lines.append(f"    Tags: {', '.join(tags[:5])}")
        lines.append(f"    {created}")
        lines.append("")

    return "\n".join(lines)


def print_project_highlights(summary: str, project_name: str):
    """Print the project highlights summary."""
    border = "═" * 60
    print(f"\n{border}")
    print(f"║  PROJECT HIGHLIGHTS - {project_name}")
    print(f"{border}")
    print(f"\n{summary}\n")
    print(f"{border}\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    start_time = time.time()

    # DEBUG: File-based logging to verify hook execution
    # Use the script's actual location to find the project
    script_path = Path(__file__).resolve()
    hooks_dir = script_path.parent
    project_dir = hooks_dir.parent.parent  # Go up: hooks -> .factory -> project
    
    debug_log = hooks_dir / "hook-debug.log"
    
    def log_debug(msg):
        timestamp = datetime.now().isoformat()
        try:
            with open(debug_log, "a") as f:
                f.write(f"[{timestamp}] {msg}\n")
        except Exception:
            pass
    
    log_debug("=== Hook invoked ===")
    
    # Read hook input from stdin (Droid passes session info)
    input_data = {}
    is_new_session = False
    try:
        input_data = json.load(sys.stdin)
        log_debug(f"Received stdin: {json.dumps(input_data)[:500]}")
        if input_data:
            hook_event = input_data.get("hookEventName", "")  # Factory uses camelCase
            
            # For SessionStart events
            if hook_event == "SessionStart":
                source = input_data.get("source", "")
                log_debug(f"SessionStart event (source: {source})")
                print(f"[session-start-recall] Received {hook_event} (source: {source})", file=sys.stderr)
                is_new_session = True
            
            # For UserPromptSubmit - detect session start patterns
            elif hook_event == "UserPromptSubmit":
                prompt = input_data.get("prompt", "").strip()
                log_debug(f"UserPromptSubmit event (prompt: {prompt})")
                # Check for resume/start patterns
                session_start_patterns = ["/resume", "/continue", "/start", "/sessions", "/session"]
                if prompt in session_start_patterns or prompt.lower().startswith("/resume") or prompt.lower().startswith("/continue"):
                    log_debug(f"Detected session start command: {prompt}")
                    print(f"[session-start-recall] Detected session start: {prompt}", file=sys.stderr)
                    is_new_session = True
                else:
                    # Not a session start - exit silently
                    log_debug("Not a session start command, exiting silently")
                    return
    except json.JSONDecodeError as e:
        log_debug(f"JSON decode error: {e}")
        pass  # No stdin data, continue anyway
    
    # Only run if this is a new session
    if not is_new_session:
        log_debug("No new session detected, exiting")
        return

    # Identify project
    project_id = get_project_id()
    project_name = get_project_name()

    print(f"[session-start-recall] Project: {project_name}", file=sys.stderr)

    # Get recent memories (last 10)
    recent_memories = get_recent_memories(project_id, limit=10)

    # Check for cached summary
    cached_summary, cached_at = get_cached_summary(project_id)

    # Determine if we need to regenerate summary
    should_summarize = True
    if cached_summary and cached_at:
        hours_since_cache = (datetime.now() - cached_at).total_seconds() / 3600
        if hours_since_cache < CONFIG.summary_cache_hours:
            print(f"[session-start-recall] Using cached summary ({hours_since_cache:.1f}h old)", file=sys.stderr)
            summary = cached_summary
            should_summarize = False

    # Generate summary if needed
    if should_summarize:
        print("[session-start-recall] Generating project summary...", file=sys.stderr)
        all_memories = get_project_memories(project_id, limit=100)
        summary = generate_project_summary(all_memories, project_name)
        save_cached_summary(project_id, summary)

    # Build context output
    context_parts = []
    
    # Add project highlights
    border = "═" * 60
    context_parts.append(f"\n{border}")
    context_parts.append(f"║  PROJECT HIGHLIGHTS - {project_name}")
    context_parts.append(f"{border}")
    context_parts.append(f"\n{summary}\n")
    context_parts.append(f"{border}\n")
    
    # Add recent memories
    context_parts.append(format_recent_memories(recent_memories))
    
    additional_context = "\n".join(context_parts)
    
    # Output JSON for SessionStart hook (per Factory docs)
    # This injects context into the agent via hookSpecificOutput.additionalContext
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context
        }
    }
    print(json.dumps(output))

    elapsed = time.time() - start_time
    print(f"[session-start-recall] Completed in {elapsed:.2f}s", file=sys.stderr)
    log_debug(f"Hook completed successfully in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
