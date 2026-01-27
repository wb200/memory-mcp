#!/usr/bin/env python3
"""Web viewer for memories - accessible in browser."""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
from flask import Flask, render_template_string, request

app = Flask(__name__)
DB_PATH = Path.home() / ".memory-mcp" / "lancedb-memory"
ITEMS_PER_PAGE = 10


def get_memories() -> list[dict]:
    """Fetch all memories, sorted by newest first."""
    db = lancedb.connect(str(DB_PATH))
    try:
        table = db.open_table("memories")
    except Exception:
        return []  # Empty database
    # to_arrow() limits to 10 by default, use search().limit() to get all
    all_rows = table.search().limit(1000).to_list()
    # Sort by created_at descending (newest first)
    return sorted(all_rows, key=lambda m: m.get("created_at", ""), reverse=True)


def get_page_links(current: int, total: int) -> list:
    """Generate smart pagination links with ellipsis for gaps."""
    if total <= 7:
        return list(range(1, total + 1))

    links = []
    for p in range(1, total + 1):
        show_page = (
            p <= 3  # First 3 pages
            or p >= total - 2  # Last 3 pages
            or abs(p - current) <= 1  # Pages around current
        )
        if show_page:
            links.append(p)
        elif links[-1] != "...":
            links.append("...")
    return links


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Memory Viewer</title>
    <style>
        body { font-family: system-ui; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d9ff; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }
        .pagination { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
        .pagination a, .pagination span { padding: 6px 12px; background: #0f3460; color: #00d9ff; text-decoration: none; border-radius: 5px; display: inline-block; }
        .pagination a:hover { background: #16213e; }
        .pagination a.current { background: #00d9ff; color: #1a1a2e; font-weight: bold; }
        .pagination span.ellipsis { color: #888; background: transparent; }
        .pagination a.disabled { color: #666; pointer-events: none; }
        .memory { background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #00d9ff; }
        .category { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .PATTERN { background: #4a90d9; }
        .CONFIG { background: #9b59b6; }
        .DEBUG { background: #e74c3c; }
        .PERF { background: #f39c12; }
        .PREF { background: #2ecc71; }
        .INSIGHT { background: #1abc9c; }
        .API { background: #e91e63; }
        .AGENT { background: #ff9800; }
        .tags { margin-top: 8px; }
        .tag { background: #0f3460; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 5px; }
        .meta { color: #888; font-size: 12px; margin-top: 8px; }
        .filters { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .search { flex: 1; min-width: 300px; }
        .project-filter { min-width: 200px; }
        input, select { padding: 10px; border-radius: 5px; border: none; background: #0f3460; color: #fff; font-size: 14px; }
        input { width: 100%; }
        select { cursor: pointer; }
        select:hover { background: #16213e; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Viewer</h1>
        <div class="pagination">
            {% if page > 1 %}
            <a href="/?page={{ page-1 }}&project={{ selected_project }}">← Prev</a>
            {% else %}
            <a class="disabled">← Prev</a>
            {% endif %}

            {% for p in page_links %}
            {% if p == "..." %}
            <span class="ellipsis">...</span>
            {% elif p == page %}
            <span class="current">{{ p }}</span>
            {% else %}
            <a href="/?page={{ p }}&project={{ selected_project }}">{{ p }}</a>
            {% endif %}
            {% endfor %}

            {% if page < total_pages %}
            <a href="/?page={{ page+1 }}&project={{ selected_project }}">Next →</a>
            {% else %}
            <a class="disabled">Next →</a>
            {% endif %}
        </div>
    </div>
    <p>{{ total_memories }} memories total</p>
    <div class="filters">
        <div class="project-filter">
            <select id="projectSelect" onchange="filterByProject()">
                <option value="all" {% if selected_project == "all" %}selected{% endif %}>All Projects</option>
                {% for proj in projects %}
                <option value="{{ proj.id }}" {% if selected_project == proj.id %}selected{% endif %}>
                    {{ proj.name }}
                </option>
                {% endfor %}
            </select>
        </div>
        <div class="search">
            <input type="text" id="search" placeholder="Search memories..." onkeyup="filterMemories()">
        </div>
    </div>
    <div id="memories">
        {% for m in memories %}
        <div class="memory" data-content="{{ m.content|lower }}">
            <span class="category {{ m.category }}">{{ m.category }}</span>
            <p>{{ m.content }}</p>
            <div class="tags">
                {% for t in m.tags_list %}
                <span class="tag">{{ t }}</span>
                {% endfor %}
            </div>
            <div class="meta">{{ m.project_id[:50] }}... | {{ m.created_at[:19] }}</div>
        </div>
        {% endfor %}
    </div>
    <script>
        function filterMemories() {
            const q = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.memory').forEach(el => {
                el.style.display = el.dataset.content.includes(q) ? 'block' : 'none';
            });
        }

        function filterByProject() {
            const project = document.getElementById('projectSelect').value;
            window.location.href = '/?project=' + encodeURIComponent(project);
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    all_memories = get_memories()
    for m in all_memories:
        m["tags_list"] = json.loads(m.get("tags", "[]"))

    # Get unique projects and create friendly names
    projects_raw = sorted({m.get("project_id", "unknown") for m in all_memories})
    projects = []
    for proj in projects_raw:
        # Create friendly name from project path
        friendly = Path(proj).name if "/" in proj else proj[:50]
        projects.append({"id": proj, "name": friendly})

    # Filter by selected project
    selected_project = request.args.get("project", "all")
    if selected_project != "all":
        all_memories = [m for m in all_memories if m.get("project_id") == selected_project]

    total = len(all_memories)
    page = int(request.args.get("page", 1))
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    memories = all_memories[start:end]
    total_pages = (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    page_links = get_page_links(page, total_pages)

    return render_template_string(
        HTML,
        memories=memories,
        page=page,
        total_pages=total_pages,
        total_memories=total,
        page_links=page_links,
        projects=projects,
        selected_project=selected_project,
    )


if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser")
    app.run(port=5000)
