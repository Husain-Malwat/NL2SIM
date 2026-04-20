#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "community"
LOG_FILE = Path(__file__).parent.parent / "data" / "raw" / "community" / "_scrape_log.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordered by priority (HIGH first)
REPOS = [
    # (owner, repo, description)
    ("anirbanm93", "mumax3_simulation_MBG",     "Magnonic Bragg grating spin-wave dynamics"),
    ("mdai26",     "mumax3scripts",              "Python code and scripts for mumax3 simulation"),
    ("ricktjwong", "micromagnetic-simulations",  "Micromagnetic simulations with mumax+OOMMF"),
    ("nicholasfprestes", "Mumax_simulations",    "Mumax projects"),
    ("swapneelap", "Mumax-Scripts",              "Mumax simulation input files"),
    ("GetaCupOfAmericano", "MumaxSimulations",   "MuMax simulations"),
    ("nbeaver",    "mumax3-simulations",         "mumax3 simulations on Discovery cluster"),
    ("Donatobot",  "mumax",                      "Mumax repository for Micromagnetic Simulations"),
    ("IsobelClarke1", "EDSR-Mumax3-Simulations","MSci Project Simulations"),
    ("Wlybs",      "paso_mumax3_simulation",     "Paso mumax3 simulations"),
    ("ojoayomipo", "MUMAX-Simulation",           "MUMAX Simulation"),
    ("juhalinj",   "Mumax-simulations",          "Mumax simulations"),
    ("EQUBAL100",  "Mumax_simulation",           "Mumax simulation"),
    ("basicallyAlexOh", "FeGe-simulations",      "Static and Dynamic MuMax simulations of FeGe"),
    ("Qiuyuan-Wang","DW_mumax3",                 "Current- and field-driven domain wall motion"),
    ("kmcai",      "Mumax3",                     "Micromagnetic simulation"),
    ("kaiyang5029","mumax3",                     "Micromagnetic Simulator"),
    ("kaanyapici", "mumax3",                     "Micro magnetic simulations"),
    ("ninjha252",  "MuMax",                      "Permalloy MuMax3 studies"),
    ("joovon",     "mumax3.10",                  "GPU-accelerated micromagnetic simulator fork"),
    ("peytondmurray","mx3tools",                 "Python tools for mumax3 (may contain .mx3)"),
    ("JeroenMulkers","mumax3-tutorial",          "Tutorial (standardproblem4.mx3)"),
    ("neonh",      "mumaxpy",                    "Package for automated mumax3 simulations"),
    ("LekhaRam",   "MumaxAveragedSampleSolution","Mumax3.10Beta averaged sampling"),
]

GITHUB_API = "https://api.github.com"
RAW_BASE   = "https://raw.githubusercontent.com"

def get_headers(token: str = None):
    h = {"Accept": "application/vnd.github.v3+json",
         "User-Agent": "mumax3-dataset-builder/1.0"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def get_tree(owner, repo, headers):
    """Recursively fetch file tree for default branch."""
    # First get default branch
    r = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers, timeout=15)
    if r.status_code == 404:
        print(f"  [SKIP] {owner}/{repo} — 404 not found")
        return []
    r.raise_for_status()
    branch = r.json().get("default_branch", "master")

    # Then get full tree
    r2 = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
        headers=headers, timeout=30)
    if r2.status_code != 200:
        print(f"  [WARN] {owner}/{repo} tree fetch returned {r2.status_code}")
        return []
    tree = r2.json().get("tree", [])
    return [(item["path"], branch) for item in tree if item["type"] == "blob"]

def download_file(owner, repo, path, branch, out_dir, headers):
    """Download a single file and save it."""
    raw_url = f"{RAW_BASE}/{owner}/{repo}/{branch}/{path}"
    r = requests.get(raw_url, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    content = r.text
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # sanitize path for local filename
    safe_name = path.replace("/", "__")
    out_path = out_dir / safe_name
    out_path.write_text(content)

    return {
        "owner": owner,
        "repo": repo,
        "path": path,
        "branch": branch,
        "raw_url": raw_url,
        "local_file": str(out_path.relative_to(Path(__file__).parent.parent)),
        "content_hash": content_hash,
        "lines": len(content.splitlines()),
        "bytes": len(content.encode()),
    }

def scrape_repo(owner, repo, desc, headers, logs):
    repo_dir = OUT_DIR / f"{owner}__{repo}"
    repo_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Scraping: {owner}/{repo}")
    print(f"  {desc}")

    try:
        all_files = get_tree(owner, repo, headers)
    except Exception as e:
        print(f"  [ERROR] Failed to get tree: {e}")
        return

    mx3_files = [(p, b) for p, b in all_files if p.lower().endswith(".mx3")]
    print(f"  Found {len(mx3_files)} .mx3 files out of {len(all_files)} total files")

    for path, branch in mx3_files:
        print(f"  Downloading: {path} ...", end=" ")
        try:
            meta = download_file(owner, repo, path, branch, repo_dir, headers)
            if meta:
                print(f"OK ({meta['lines']} lines)")
                logs.append(meta)
            else:
                print("FAIL (non-200)")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)  # be polite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    token = args.token
    headers = get_headers(token)
    if token:
        print(f"Using GitHub API with authentication token.")
    else:
        print(f"WARNING: No GitHub token. Rate limit is 60 req/hour. Set GITHUB_TOKEN env var.")

    logs = []
    for owner, repo, desc in REPOS:
        scrape_repo(owner, repo, desc, headers, logs)
        # Check rate limit
        if not token and len(logs) > 50:
            print("\nApproaching unauthenticated rate limit, sleeping 60s...")
            time.sleep(60)

    # Save log
    LOG_FILE.write_text(json.dumps(logs, indent=2))
    print(f"\n{'='*60}")
    print(f"Total .mx3 files downloaded: {len(logs)}")
    print(f"Log saved: {LOG_FILE}")

    # Summary per repo
    from collections import defaultdict
    per_repo = defaultdict(int)
    for entry in logs:
        per_repo[f"{entry['owner']}/{entry['repo']}"] += 1
    print("\nPer-repo breakdown:")
    for k, v in sorted(per_repo.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
