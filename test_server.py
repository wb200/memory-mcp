#!/usr/bin/env python3
"""
Comprehensive test suite for Droid Memory MCP Server (LanceDB).

Run with: pytest test_server.py -v
Or manual: python test_server.py
"""
# ruff: noqa: E402

import asyncio
import json
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

# Load API key before imports
secrets_path = Path.home() / ".secrets" / "GOOGLE_API_KEY"
if secrets_path.exists():
    os.environ["GOOGLE_API_KEY"] = secrets_path.read_text().strip()

# TEST DATABASE ISOLATION (set before importing server)
# Use project-relative path, configurable via LANCEDB_MEMORY_TEST_PATH
TEST_DB_PATH = Path(
    os.environ.get("LANCEDB_MEMORY_TEST_PATH", Path(__file__).parent / "lancedb-memory-test")
)
os.environ["LANCEDB_MEMORY_PATH"] = str(TEST_DB_PATH)

import server as server_module
from server import (
    CONFIG,
    VALID_CATEGORIES,
    get_embedding,
    init_database,
    memory_delete,
    memory_recall,
    memory_recall_project,
    memory_save,
    memory_stats,
    memory_update,
    smart_summarize,
)

# CRITICAL: Reset CONFIG.db_path because the Config dataclass evaluates
# its default at class definition time, not at runtime.
# Use object.__setattr__ because Config is frozen=True.
object.__setattr__(server_module.CONFIG, "db_path", TEST_DB_PATH)

RANDOM_WORDS = [
    "quantum",
    "neural",
    "cosmic",
    "fractal",
    "entropy",
    "stellar",
    "velocity",
    "catalyst",
    "synthesis",
    "algorithm",
    "paradox",
    "spectrum",
    "resonance",
    "wavelength",
    "dimension",
    "threshold",
    "equilibrium",
    "trajectory",
]


def unique_content(base: str) -> str:
    """Generate semantically unique content to avoid deduplication."""
    # Use more words and a longer UUID to ensure uniqueness
    random_phrase = " ".join(random.sample(RANDOM_WORDS, 6))
    unique_id = uuid.uuid4().hex
    return f"{random_phrase} - {base} - unique context {unique_id}"


@pytest.fixture(autouse=True)
async def setup_db():
    """Initialize isolated test database."""
    import shutil

    server_module._db = None
    server_module._table = None
    if TEST_DB_PATH.exists():
        shutil.rmtree(TEST_DB_PATH)

    await init_database()


# =============================================================================
# Core CRUD Tests
# =============================================================================


class TestMemorySave:
    """Tests for memory_save tool."""

    async def test_save_basic(self):
        """Test basic memory saving with UUID."""
        result = await memory_save(
            content=unique_content("TEST - Basic save test."),
            category="DEBUG",
            tags=["test", "pytest"],
        )
        assert "Saved" in result
        assert "DEBUG" in result
        # UUID format: shows first 8 chars
        assert "..." in result

    async def test_save_empty_content_fails(self):
        """Empty content should fail."""
        result = await memory_save(content="")
        assert "Error" in result

    async def test_save_whitespace_only_fails(self):
        """Whitespace-only content should fail."""
        result = await memory_save(content="   \n\t  ")
        assert "Error" in result

    async def test_save_invalid_category_fails(self):
        """Invalid category should fail with helpful message."""
        result = await memory_save(content=unique_content("test"), category="INVALID")
        assert "Error" in result
        assert "Invalid category" in result

    async def test_save_all_valid_categories(self):
        """Test saving in all valid categories."""
        for category in VALID_CATEGORIES:
            result = await memory_save(
                content=unique_content(f"{category} - Test"),
                category=category,
            )
            assert "Saved" in result, f"Failed for {category}"

    async def test_save_with_summarize(self):
        """Test AI summarization."""
        uid = uuid.uuid4().hex
        long_content = f"Project {uid}: A complex investigation revealed critical issues. " * 10
        result = await memory_save(content=long_content, category="DEBUG", summarize=True)
        assert "Saved" in result or "Duplicate" in result
        if "Saved" in result:
            assert "Summarized" in result

    async def test_deduplication_detection(self):
        """Test that near-duplicates are detected."""
        content = unique_content("DEBUG - Deduplication test memory.")
        result1 = await memory_save(content=content, category="DEBUG")
        assert "Saved" in result1
        result2 = await memory_save(content=content, category="DEBUG")
        assert "Duplicate" in result2


class TestMemoryRecall:
    """Tests for memory_recall tool (hybrid search)."""

    async def test_recall_semantic(self):
        """Test semantic search."""
        await memory_save(
            content=unique_content("PATTERN - Use asyncio.to_thread for blocking I/O."),
            category="PATTERN",
        )
        result = await memory_recall(query="async blocking operations")
        assert "Found" in result or "No memories" in result
        assert "Error" not in result

    async def test_recall_empty_query_fails(self):
        """Empty query should fail."""
        result = await memory_recall(query="")
        assert "Error" in result

    async def test_recall_with_category_filter(self):
        """Test category filtering."""
        result = await memory_recall(query="python", category="PATTERN")
        assert "Error" not in result

    async def test_recall_invalid_category_fails(self):
        """Invalid category in recall should fail."""
        result = await memory_recall(query="test", category="INVALID")
        assert "Error" in result
        assert "Invalid category" in result

    async def test_recall_shows_hybrid_search_type(self):
        """Verify hybrid search is being used."""
        await memory_save(content=unique_content("LanceDB hybrid test"), category="DEBUG")
        result = await memory_recall(query="LanceDB")
        # Should show hybrid or vector in search type
        if "Found" in result:
            assert "hybrid" in result.lower() or "vector" in result.lower()

    async def test_project_scoped_search(self):
        """Test project-scoped vs global search."""
        content = unique_content("PROJECT SCOPE TEST")
        await memory_save(content=content, category="DEBUG")
        global_result = await memory_recall(query="project scope", limit=5)
        project_result = await memory_recall_project(query="project scope", limit=5)
        # Check that search succeeded (starts with "Found" or "No memories")
        assert global_result.startswith("Found") or global_result.startswith("No memories")
        assert project_result.startswith("Found") or project_result.startswith("No memories")


class TestMemoryUpdate:
    """Tests for memory_update tool."""

    async def test_update_content(self):
        """Test updating memory content."""
        save_result = await memory_save(
            content=unique_content("TEST - Original content."),
            category="DEBUG",
        )
        assert "Saved" in save_result
        # Extract partial UUID from "Saved (ID: abc12345..., DEBUG)"
        memory_id = save_result.split("ID: ")[1].split("...")[0]

        result = await memory_update(memory_id=memory_id, content=unique_content("Updated."))
        assert "Updated" in result

    async def test_update_nonexistent_fails(self):
        """Updating nonexistent memory should fail."""
        result = await memory_update(memory_id="nonexistent123", content="test")
        assert "not found" in result

    async def test_update_invalid_category_fails(self):
        """Updating with invalid category should fail."""
        save_result = await memory_save(content=unique_content("test"), category="DEBUG")
        memory_id = save_result.split("ID: ")[1].split("...")[0]
        result = await memory_update(memory_id=memory_id, category="INVALID")
        assert "Error" in result
        assert "Invalid category" in result


class TestMemoryDelete:
    """Tests for memory_delete tool."""

    async def test_delete_existing(self):
        """Test deleting a memory."""
        save_result = await memory_save(content=unique_content("To delete."), category="DEBUG")
        assert "Saved" in save_result
        memory_id = save_result.split("ID: ")[1].split("...")[0]
        result = await memory_delete(memory_id=memory_id)
        assert "Deleted" in result

    async def test_delete_nonexistent(self):
        """Deleting nonexistent memory should report not found."""
        result = await memory_delete(memory_id="nonexistent999")
        assert "not found" in result

    async def test_delete_with_partial_id(self):
        """Test deleting with partial UUID (first 8 chars)."""
        save_result = await memory_save(
            content=unique_content("Partial ID test."), category="DEBUG"
        )
        memory_id = save_result.split("ID: ")[1].split("...")[0]  # Gets first 8 chars
        result = await memory_delete(memory_id=memory_id)
        assert "Deleted" in result


class TestMemoryStats:
    """Tests for memory_stats tool."""

    async def test_stats_structure(self):
        """Stats should return expected structure."""
        result = await memory_stats()
        assert "Memory Statistics" in result or "No memories" in result
        if "Memory Statistics" in result:
            assert "Total:" in result
            assert "By Category:" in result
            assert "FTS Index:" in result  # New: verify FTS index reported


# =============================================================================
# Embedding & Hybrid Search Tests
# =============================================================================


class TestEmbeddings:
    """Tests for embedding generation."""

    async def test_embedding_generation(self):
        """Test embeddings are generated correctly."""
        embedding = await get_embedding("Test embedding generation")
        assert embedding is not None
        assert len(embedding) == CONFIG.embedding_dim
        import numpy as np

        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01, f"Embedding not normalized: {norm}"

    async def test_embedding_similarity(self):
        """Test similar texts have similar embeddings."""
        import numpy as np

        e1 = await get_embedding("Python async programming patterns")
        e2 = await get_embedding("Asynchronous Python code patterns")
        e3 = await get_embedding("Recipe for chocolate cake baking")
        assert all(e is not None for e in [e1, e2, e3])
        sim_12 = np.dot(e1, e2)
        sim_13 = np.dot(e1, e3)
        assert sim_12 > sim_13, f"Expected similar texts closer: {sim_12} vs {sim_13}"


class TestSummarization:
    """Tests for smart summarization."""

    async def test_summarize_long_content(self):
        """Test long content is summarized."""
        long_content = (
            f"Investigation {uuid.uuid4().hex}: " + " ".join(random.sample(RANDOM_WORDS, 10)) * 5
        )
        result = await smart_summarize(long_content, category="PERF")
        assert "summary" in result
        assert "tags" in result
        assert len(result["summary"]) < len(long_content)


# =============================================================================
# Integration Tests
# =============================================================================


class TestConcurrency:
    """Tests for thread-safety and concurrent operations."""

    async def test_concurrent_saves(self):
        """Test thread-safety of concurrent memory saves."""
        # Use very different content for each save to avoid duplicate detection
        unique_topics = [
            "quantum computing algorithms",
            "machine learning pipelines",
            "database optimization techniques",
            "network security protocols",
            "cloud infrastructure patterns",
            "microservice architectures",
            "real-time data streaming",
            "distributed systems design",
            "containerization strategies",
            "API gateway configurations",
        ]
        tasks = [
            memory_save(unique_content(f"{topic} - iteration {i}"), category="INSIGHT")
            for i, topic in enumerate(unique_topics)
        ]
        results = await asyncio.gather(*tasks)
        # Allow duplicates since concurrent saves may detect similarity
        success_count = sum(1 for r in results if "Saved" in r or "Duplicate" in r)
        assert success_count == len(results), f"Some saves failed unexpectedly: {results}"

    async def test_concurrent_mixed_operations(self):
        """Test concurrent saves, recalls, and stats."""
        # Save baseline memories
        await memory_save(unique_content("Baseline for stress test"), category="DEBUG")
        await memory_save(unique_content("Another baseline memory"), category="PATTERN")

        # Mix of operations - build list explicitly to avoid variable scope issues
        tasks = []
        for i in range(3):
            tasks.extend(
                [
                    memory_save(unique_content(f"Stress {i}"), category="INSIGHT"),
                    memory_recall("stress test"),
                    memory_stats(),
                ]
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"

    async def test_concurrent_recall_same_query(self):
        """Test multiple concurrent recalls with same query."""
        await memory_save(unique_content("Python async patterns"), category="PATTERN")

        tasks = [memory_recall("python async") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should return valid results (starts with "Found" or "No memories")
        assert all(r.startswith("Found") or r.startswith("No memories") for r in results), (
            f"Recall errors: {results}"
        )


class TestFullLifecycle:
    """Test complete memory lifecycle."""

    async def test_create_read_update_delete(self):
        """CRUD lifecycle with new UUID system."""
        # Create
        content = unique_content("LIFECYCLE - Test memory.")
        save_result = await memory_save(content=content, category="DEBUG", tags=["lifecycle"])
        assert "Saved" in save_result
        memory_id = save_result.split("ID: ")[1].split("...")[0]

        # Read via recall
        recall_result = await memory_recall(query="lifecycle", limit=10)
        assert "Found" in recall_result or "lifecycle" in recall_result.lower()

        # Update
        update_result = await memory_update(memory_id=memory_id, content=unique_content("Updated."))
        assert "Updated" in update_result

        # Delete
        delete_result = await memory_delete(memory_id=memory_id)
        assert "Deleted" in delete_result

        # Verify deleted
        verify = await memory_delete(memory_id=memory_id)
        assert "not found" in verify


class TestHookIntegration:
    """Tests for the memory-extractor hook (if exists)."""

    def test_hook_exists(self):
        """Test hook script exists and is executable."""
        hook_path = Path.home() / ".factory" / "hooks" / "memory-extractor.py"
        if not hook_path.exists():
            pytest.skip("Hook not installed")
        assert os.access(hook_path, os.X_OK), "Hook not executable"

    def test_hook_syntax(self):
        """Test hook has valid Python syntax."""
        hook_path = Path.home() / ".factory" / "hooks" / "memory-extractor.py"
        if not hook_path.exists():
            pytest.skip("Hook not installed")
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(hook_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestMCPConfig:
    """Tests for MCP configuration."""

    def test_mcp_config_valid(self):
        """Test MCP configuration is valid."""
        mcp_config_path = Path.home() / ".factory" / "mcp.json"
        if not mcp_config_path.exists():
            pytest.skip("mcp.json not found")
        with mcp_config_path.open() as f:
            config = json.load(f)
        assert "mcpServers" in config
        assert "memory" in config["mcpServers"]


# =============================================================================
# Manual Test Runner
# =============================================================================


async def run_manual_tests():
    """Run comprehensive manual tests."""
    print("=" * 70)
    print("DROID MEMORY MCP - TEST SUITE (LanceDB + Hybrid Search)")
    print("=" * 70)

    await init_database()
    print(f"\nDatabase: {CONFIG.db_path}")
    print(f"Embedding: {CONFIG.embedding_model} ({CONFIG.embedding_dim}D)")
    print(f"FTS Weight: {CONFIG.fts_weight}")
    print()

    tests_passed = 0
    tests_failed = 0

    async def run_test(name: str, test_fn):
        nonlocal tests_passed, tests_failed
        try:
            print(f"  [{name}]...", end=" ", flush=True)
            if asyncio.iscoroutinefunction(test_fn):
                await test_fn()
            else:
                test_fn()
            print("PASSED")
            tests_passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            tests_failed += 1

    # Run test classes
    print("\n--- Memory Save ---")
    save = TestMemorySave()
    await run_test("save_basic", save.test_save_basic)
    await run_test("save_empty_fails", save.test_save_empty_content_fails)
    await run_test("save_invalid_category", save.test_save_invalid_category_fails)
    await run_test("deduplication", save.test_deduplication_detection)

    print("\n--- Memory Recall (Hybrid Search) ---")
    recall = TestMemoryRecall()
    await run_test("recall_semantic", recall.test_recall_semantic)
    await run_test("recall_empty_fails", recall.test_recall_empty_query_fails)
    await run_test("recall_category_filter", recall.test_recall_with_category_filter)
    await run_test("recall_hybrid_type", recall.test_recall_shows_hybrid_search_type)

    print("\n--- Memory Update ---")
    update = TestMemoryUpdate()
    await run_test("update_content", update.test_update_content)
    await run_test("update_nonexistent", update.test_update_nonexistent_fails)

    print("\n--- Memory Delete ---")
    delete = TestMemoryDelete()
    await run_test("delete_existing", delete.test_delete_existing)
    await run_test("delete_partial_id", delete.test_delete_with_partial_id)

    print("\n--- Stats ---")
    stats = TestMemoryStats()
    await run_test("stats_structure", stats.test_stats_structure)

    print("\n--- Embeddings ---")
    embed = TestEmbeddings()
    await run_test("embedding_generation", embed.test_embedding_generation)
    await run_test("embedding_similarity", embed.test_embedding_similarity)

    print("\n--- Full Lifecycle ---")
    lifecycle = TestFullLifecycle()
    await run_test("crud_lifecycle", lifecycle.test_create_read_update_delete)

    # Summary
    print("\n" + "=" * 70)
    total = tests_passed + tests_failed
    print(f"RESULTS: {tests_passed}/{total} tests passed")
    if tests_failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILED: {tests_failed} tests")
    print("=" * 70)

    return tests_failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_manual_tests())
    sys.exit(0 if success else 1)
