"""
Shared cache utilities for Exp1 figure scripts.

Cache files are written to paper/fig/exp1/data/ as JSON.
Each cache embeds the abs logs_dir and good name so that a changed
logs path or good name triggers a full recompute.
"""

import glob
import json
import os
import time


def get_data_dir(output_path):
    """Return the data/ cache directory next to the figure output PDF."""
    fig_dir = os.path.dirname(os.path.abspath(output_path))
    return os.path.join(fig_dir, "data")


def get_cache_path(data_dir, script_stem, good):
    """E.g. paper/fig/exp1/data/exp1_heatmap_food.json"""
    return os.path.join(data_dir, f"{script_stem}_{good}.json")


def newest_run_mtime(run_dirs):
    """Return the newest mtime of any state_t*.json across all run_dirs."""
    newest = 0.0
    for d in run_dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "state_t*.json")):
            try:
                mtime = os.path.getmtime(f)
                if mtime > newest:
                    newest = mtime
            except OSError:
                pass
    return newest


def is_cache_fresh(cache_path, run_dirs, logs_dir, good):
    """
    Return True iff cache_path exists, metadata matches, and the cache
    file is at least as new as the newest state file in any run_dir.
    """
    if not os.path.isfile(cache_path):
        return False
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        meta = cached.get("_meta", {})
        if meta.get("logs_dir") != os.path.abspath(logs_dir):
            return False
        if meta.get("good") != good:
            return False
    except Exception:
        return False
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime >= newest_run_mtime(run_dirs)


def save_cache(cache_path, data_dict, logs_dir, good):
    """Write data_dict (JSON-serialisable) to cache_path with metadata."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "_meta": {
            "logs_dir": os.path.abspath(logs_dir),
            "good":     good,
            "created":  time.time(),
        },
        "data": data_dict,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)


def load_cache_data(cache_path):
    """Return the 'data' section of a cache file."""
    with open(cache_path) as f:
        return json.load(f)["data"]
