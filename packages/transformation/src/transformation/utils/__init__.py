import os
from itertools import chain
from pathlib import Path

# --- utils/helpers ----
def _parse_env_sources(env_key: str) -> list[dict]:
    """Parse comma-separated SOURCES env var into list of dicts."""
    raw = os.getenv(env_key, "")
    if not raw.strip():
        return []
    
    sources = []
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            continue
        
        name, src_type, source = parts
        sources.append({"name": name, "src_type": src_type, "source": source})
    return sources

def _build_data_dirs(ingestion_root: Path, *source_lists: list[dict]) -> list[dict]:
    """Build data_dirs from multiple source lists."""
    return [
        {"path": ingestion_root / s["name"], "name": s["name"]}
        for s in chain(*source_lists)
    ]