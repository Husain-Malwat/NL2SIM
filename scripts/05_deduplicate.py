#!/usr/bin/env python3

import hashlib
import csv
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
OUT_DIR   = ROOT / "data" / "raw" / "all_deduped"
LOG_FILE  = ROOT / "data" / "collection_log.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("official_tests",    RAW / "official_tests"),
    ("official_examples", RAW / "official_examples"),
    ("workshop",          RAW / "workshop"),
    ("community",         RAW / "community"),
    ("github_search",     RAW / "github_search"),
    ("papers",            RAW / "papers"),
    ("forum",             RAW / "forum"),
]

# API identifier regex (case-insensitive) — all known top-level identifiers
ALL_API_IDS = re.compile(
    r'\b(setgridsize|setcellsize|setmesh|setpbc|edgesmooth'
    r'|rect|circle|cylinder|ellipse|ellipsoid|cuboid|sphere|superball|layers|layer|cell'
    r'|xrange|yrange|zrange|roughness|imageshape'
    r'|defregion|defregioncell|redefregion'
    r'|uniform|vortex|twodomain|randommag|vortexwall|blochskyrmion|neelskyrmion'
    r'|conicalMag|helicalmag|hopfionrlattack'
    r'|msat|aex|alpha|ku1|ku2|kc1|kc2|kc3|anisu|anisc1|anisc2'
    r'|dind|dbulk|froznspins|temp|gammall'
    r'|b_ext|j|pol|xi|lambda|epsilonprime|fixedlayer|freebodythickness'
    r'|save|saveas|autosave|snapshot|snapshotas|autosnapshot'
    r'|tableadd|tableaddvar|tablesave|tableautosave'
    r'|run|steps|runwhile|relax|minimize|setsolver'
    r'|maxdt|mindt|fixdt|maxerr|headroom'
    r'|mfmlift|mfmdipole'
    r'|crop|cropx|cropy|cropz|croplane|cropregion'
    r'|shift|shiftgeom|shiftm|shiftregions'
    r'|ext_makegrains|ext_make3dgrains|ext_scaleexchange|ext_interexchange'
    r'|ext_bubblepos|ext_dwpos|ext_dwxpos|ext_dwtilt|ext_corepos'
    r'|ext_topologicalcharge|ext_centerwall|ext_centerbubble'
    r'|addfieldterm|addedensterm|removecustomfields'
    r'|const|constvector|mul|madd|add|div|dot|cross|normalized|shifted|masked|mulmv'
    r'|sum|sumvector|runningaverage'
    r'|b_demag|b_exch|b_anis|b_eff|b_therm|b_custom'
    r'|e_total|e_demag|e_exch|e_anis|e_zeeman|e_therm|e_dmi|e_custom'
    r'|torque|lltorque|sttorque|geom|regions'
    r'|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh'
    r'|exp|log|log2|log10|sqrt|cbrt|pow|abs|floor|ceil|round'
    r'|max|min|mod|erf|erfc|gamma|rand|randnorm|randexp'
    r'|vector|index2coord|loadfile|exit|print|sprint|sprintf)\b',
    re.IGNORECASE,
)

CATEGORIES = {
    "std_problem":   re.compile(r'std_problem|standardproblem', re.I),
    "skyrmion":      re.compile(r'skyrmion|skyrm', re.I),
    "domain_wall":   re.compile(r'domain.wall|dw|racetrack', re.I),
    "vortex":        re.compile(r'vortex', re.I),
    "hysteresis":    re.compile(r'hysteresis|hyst', re.I),
    "stt":           re.compile(r'slonczewski|zhang.li|spin.transfer|stt', re.I),
    "spin_waves":    re.compile(r'spin.wave|magnon|dispersion', re.I),
    "mfm":           re.compile(r'\bmfm\b', re.I),
    "voronoi":       re.compile(r'voronoi|grain', re.I),
    "pma":           re.compile(r'\bpma\b|perp.*aniso|out.of.plane', re.I),
    "geometry":      re.compile(r'geometry|shape', re.I),
    "relax":         re.compile(r'\brelax\b|\bminimize\b', re.I),
    "oscillation":   re.compile(r'oscill|precession|ferromag.*resonance|fmr', re.I),
    "hopfion":       re.compile(r'hopfion', re.I),
    "benchmark":     re.compile(r'bench|benchmark', re.I),
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def classify(content: str, filename: str) -> str:
    combined = f"{filename} {content[:500]}"
    for cat, pat in CATEGORIES.items():
        if pat.search(combined):
            return cat
    return "general"


def complexity(content: str) -> str:
    lines = len([l for l in content.splitlines() if l.strip() and not l.strip().startswith("//")])
    api_calls = len(ALL_API_IDS.findall(content))
    region_count = len(set(re.findall(r'DefRegion\s*\(\s*(\d+)', content, re.I)))
    if lines < 30 and region_count <= 1:
        return "trivial"
    elif lines < 80 and region_count <= 2:
        return "simple"
    elif lines < 200 or region_count <= 5:
        return "medium"
    else:
        return "complex"


def extract_api_calls(content: str) -> str:
    found = set(m.lower() for m in ALL_API_IDS.findall(content))
    return "|".join(sorted(found))


def main():
    seen_hashes: dict[str, dict] = {}  # hash → first entry
    rows = []
    counter = 0

    for source_type, source_dir in SOURCES:
        if not source_dir.exists():
            print(f"  [SKIP] {source_dir} does not exist")
            continue

        # find all .mx3 files, including subdirectories for community
        mx3_files = sorted(source_dir.rglob("*.mx3"))
        print(f"  {source_type}: {len(mx3_files)} .mx3 files found")

        for fpath in mx3_files:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"    [ERR] {fpath}: {e}")
                continue

            h = sha256(content)
            subdir = str(fpath.parent.relative_to(source_dir)) if fpath.parent != source_dir else ""

            is_dup = h in seen_hashes
            if not is_dup:
                # Save canonical copy
                saved_name = f"{source_type}_{counter:04d}_{fpath.name}"
                out_path = OUT_DIR / saved_name
                try:
                    shutil.copy2(fpath, out_path)
                except Exception as e:
                    print(f"    [ERR] copy {fpath}: {e}")
                    saved_name = ""

                seen_hashes[h] = {
                    "id": counter,
                    "saved_filename": saved_name,
                }
                counter += 1

            api_calls = extract_api_calls(content)
            lines = content.count("\n") + 1
            size  = len(content.encode())
            cat   = classify(content, fpath.name)
            cpx   = complexity(content)

            has_relax    = bool(re.search(r'\brelax\s*\(', content, re.I))
            has_minimize = bool(re.search(r'\bminimize\s*\(', content, re.I))
            has_run      = bool(re.search(r'\brun\s*\(', content, re.I))
            has_regions  = bool(re.search(r'\bDefRegion\s*\(', content, re.I))
            has_dmi      = bool(re.search(r'\b(Dind|Dbulk)\s*[=.]', content, re.I))
            has_stt      = bool(re.search(r'\b(Pol|FixedLayer|Lambda|EpsilonPrime|Slonczewski|ZhangLi)\b', content, re.I))

            rows.append({
                "id":               seen_hashes[h]["id"],
                "content_hash":     h,
                "is_duplicate":     is_dup,
                "source_type":      source_type,
                "source_subdir":    subdir,
                "original_filename": fpath.name,
                "saved_filename":   seen_hashes[h]["saved_filename"],
                "num_lines":        lines,
                "size_bytes":       size,
                "api_calls_used":   api_calls,
                "physics_category": cat,
                "complexity":       cpx,
                "has_relax":        has_relax,
                "has_minimize":     has_minimize,
                "has_run":          has_run,
                "has_regions":      has_regions,
                "has_dmi":          has_dmi,
                "has_stt":          has_stt,
                "parse_status":     "ok",
            })

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    unique = sum(1 for r in rows if not r["is_duplicate"])
    dups   = len(rows) - unique
    print(f"\nTotal scripts processed : {len(rows)}")
    print(f"Unique scripts          : {unique}")
    print(f"Duplicates removed      : {dups}")
    print(f"Collection log          : {LOG_FILE}")
    print(f"Deduped copies          : {OUT_DIR}")

    # Category breakdown
    cats = defaultdict(int)
    for r in rows:
        if not r["is_duplicate"]:
            cats[r["physics_category"]] += 1
    print("\nPhysics category breakdown (unique scripts):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {count}")


if __name__ == "__main__":
    main()
