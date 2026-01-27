"""Shared utility functions for memory-mcp."""

import re


def normalize_git_url(url: str) -> str:
    """Normalize git URLs to canonical format: provider.com/owner/repo

    Examples:
        git@github.com:wb200/memory-mcp.git -> github.com/wb200/memory-mcp
        https://github.com/wb200/memory-mcp.git -> github.com/wb200/memory-mcp
        git@gitlab.com:owner/project -> gitlab.com/owner/project
    """
    # Remove .git suffix
    url = url.removesuffix(".git")

    # SSH format: git@github.com:owner/repo -> github.com/owner/repo
    ssh_match = re.match(r"git@([^:]+):(.+)", url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"

    # HTTPS format: https://github.com/owner/repo -> github.com/owner/repo
    https_match = re.match(r"https?://(.+)", url)
    if https_match:
        return https_match.group(1)

    return url
