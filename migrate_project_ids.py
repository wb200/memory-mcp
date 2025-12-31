#!/usr/bin/env python3
"""
Migrate project IDs in existing memories to normalized format.

This script normalizes git URLs to a canonical format:
- git@github.com:owner/repo.git -> github.com/owner/repo
- https://github.com/owner/repo.git -> github.com/owner/repo

Usage:
    python migrate_project_ids.py --dry-run  # Preview changes
    python migrate_project_ids.py            # Apply migration
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import lancedb


def normalize_git_url(url: str) -> str:
    """Normalize git URLs to canonical format: provider.com/owner/repo"""
    # Remove .git suffix
    url = url.removesuffix('.git')
    
    # SSH format: git@github.com:owner/repo -> github.com/owner/repo
    ssh_match = re.match(r'git@([^:]+):(.+)', url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"
    
    # HTTPS format: https://github.com/owner/repo -> github.com/owner/repo
    https_match = re.match(r'https?://(.+)', url)
    if https_match:
        return https_match.group(1)
    
    # Already normalized or unknown format (paths, etc.)
    return url


def should_normalize(project_id: str) -> bool:
    """Check if project_id needs normalization (is a git URL)."""
    return (
        project_id.startswith('git@') or
        project_id.startswith('http://') or
        project_id.startswith('https://') or
        project_id.endswith('.git')
    )


def migrate_project_ids(dry_run: bool = True) -> None:
    """Migrate project IDs to normalized format."""
    db_path = Path.home() / ".memory-mcp" / "lancedb-memory"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    print(f"Opening database: {db_path}")
    db = lancedb.connect(str(db_path))
    
    try:
        table = db.open_table("memories")
    except Exception as e:
        print(f"Error: Could not open memories table: {e}")
        sys.exit(1)
    
    # Fetch all memories
    print("Fetching all memories...")
    all_memories = table.search().limit(10000).to_list()
    print(f"Found {len(all_memories)} total memories\n")
    
    # Analyze what needs migration
    migration_plan = defaultdict(list)
    unchanged = []
    
    for memory in all_memories:
        old_id = memory.get('project_id', '')
        
        if should_normalize(old_id):
            new_id = normalize_git_url(old_id)
            if new_id != old_id:
                migration_plan[old_id].append((memory['id'], new_id))
            else:
                unchanged.append(old_id)
        else:
            unchanged.append(old_id)
    
    # Display migration plan
    print("=" * 70)
    print("MIGRATION PLAN")
    print("=" * 70)
    
    if migration_plan:
        print("\nProject IDs to be normalized:\n")
        total_memories_to_migrate = 0
        for old_id, memories in sorted(migration_plan.items()):
            new_id = normalize_git_url(old_id)
            count = len(memories)
            total_memories_to_migrate += count
            print(f"  {count:3d} memories: {old_id}")
            print(f"       {'':3s}      -> {new_id}\n")
        
        print(f"Total memories to migrate: {total_memories_to_migrate}")
    else:
        print("\n✓ No git URLs need normalization!")
    
    # Show unchanged stats
    if unchanged:
        unchanged_counts = defaultdict(int)
        for pid in unchanged:
            unchanged_counts[pid] += 1
        
        print(f"\nUnchanged project IDs ({len(unchanged)} memories):\n")
        for pid, count in sorted(unchanged_counts.items()):
            print(f"  {count:3d} memories: {pid}")
    
    print("\n" + "=" * 70)
    
    # Execute migration if not dry run
    if not dry_run and migration_plan:
        print("\nApplying migration...")
        
        updates_made = 0
        for old_id, memories in migration_plan.items():
            new_id = normalize_git_url(old_id)
            
            for memory_id, _ in memories:
                # Fetch the memory
                results = table.search().where(f"id = '{memory_id}'").limit(1).to_list()
                if not results:
                    print(f"Warning: Memory {memory_id} not found, skipping")
                    continue
                
                memory = results[0]
                
                # Update project_id
                memory['project_id'] = new_id
                
                # Delete old and add updated
                table.delete(f"id = '{memory_id}'")
                table.add([memory])
                updates_made += 1
        
        print(f"\n✓ Migration complete! Updated {updates_made} memories")
    elif dry_run:
        print("\n⚠ DRY RUN MODE - No changes applied")
        print("Run without --dry-run to apply migration")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate project IDs to normalized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_project_ids.py --dry-run  # Preview changes
  python migrate_project_ids.py            # Apply migration
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    
    args = parser.parse_args()
    
    try:
        migrate_project_ids(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nMigration cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()
