#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone

API_URL = "https://mumax.github.io/api.html"
OUT_DIR = Path(__file__).parent.parent / "data" / "reference"
OUT_FILE = OUT_DIR / "api_scraped.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "mumax3-dataset-builder/1.0 (academic research)",
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_api_page(url: str) -> str:
    """Fetch raw HTML from the API page."""
    print(f"Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"  Got {len(resp.text)} bytes, status {resp.status_code}")
    return resp.text


def parse_api_entries(html: str) -> list[dict]:
    """
    Parse the API page HTML to extract all API entries.

    The page has a flat structure of <a> anchored entries:
        <a id="Name">NAME_WITH_SIGNATURE</a>
        <p>Description text</p>
        <p>methods: ...</p>
        <p>examples: ...</p>
    """
    try:
        from bs4 import BeautifulSoup
        return _parse_with_bs4(html)
    except ImportError:
        print("  beautifulsoup4 not installed, falling back to regex parser")
        return _parse_with_regex(html)


def _parse_with_bs4(html: str) -> list[dict]:
    """Parse using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    entries = []

    # Each API entry is an <a> tag with an id attribute (anchor)
    # followed by description/methods/examples in subsequent elements
    anchors = soup.find_all("a", id=True)

    for anchor in anchors:
        entry_id = anchor.get("id", "")
        if not entry_id or entry_id in ("top", "basics", "advanced"):
            continue

        # The anchor text contains the name + optional signature
        raw_text = anchor.get_text(strip=True)
        if not raw_text:
            continue

        # Parse signature: "Name(params) ReturnType" or just "Name"
        name, signature, return_type = _parse_signature(raw_text)
        if not name:
            continue

        # Collect subsequent siblings for description, methods, examples
        description = ""
        methods = []
        example_refs = []
        unit = ""

        sibling = anchor.find_next_sibling()
        collected = 0
        while sibling and collected < 5:
            if sibling.name == "a" and sibling.get("id"):
                break  # next API entry

            text = sibling.get_text(strip=True)

            if text.startswith("methods:"):
                methods = _parse_methods(text)
            elif text.startswith("examples:"):
                example_refs = _parse_example_refs(text, sibling)
            elif not description and text and not text.startswith("↑"):
                description = text
                # Extract unit from description like "(J/m)" or "(T)"
                unit_match = re.search(r'\(([A-Za-z/²³·µμ]+(?:\d*))\)\s*$', description)
                if unit_match:
                    unit = unit_match.group(1)

            sibling = sibling.find_next_sibling()
            collected += 1

        entry = {
            "name": name,
            "signature": signature,
            "return_type": return_type,
            "description": description,
            "methods": methods,
            "examples": example_refs,
        }
        if unit:
            entry["unit"] = unit

        # Classify the entry type
        entry["entry_type"] = _classify_entry(name, signature, return_type, methods, description)

        entries.append(entry)

    return entries


def _parse_signature(raw: str) -> tuple[str, str, str]:
    """Parse 'Name(params) ReturnType' into (name, full_sig, return_type)."""
    raw = raw.strip()

    # Function with params: "Name(Type, Type) ReturnType"
    func_match = re.match(r'^(\w+)\s*(\([^)]*\))\s*(.*?)$', raw)
    if func_match:
        name = func_match.group(1)
        params = func_match.group(2)
        ret = func_match.group(3).strip()
        return name, f"{name}{params}", ret

    # Simple name (variable/constant)
    name_match = re.match(r'^(\w+)$', raw)
    if name_match:
        return name_match.group(1), name_match.group(1), ""

    # Fallback: take first word
    parts = raw.split()
    if parts:
        return parts[0], raw, ""

    return "", "", ""


def _parse_methods(text: str) -> list[str]:
    """Parse 'methods: Average( )  Comp( int )  ...' into list of method signatures."""
    methods_str = text.replace("methods:", "").strip()
    # Split on multiple spaces (entries are separated by whitespace)
    raw_methods = re.findall(r'(\w+\s*\([^)]*\))', methods_str)
    return [m.strip() for m in raw_methods]


def _parse_example_refs(text: str, element) -> list[int]:
    """Parse example references like [[1]] [[3]] [[5]] into list of ints."""
    refs = re.findall(r'\[(\d+)\]', text)
    return [int(r) for r in refs]


def _classify_entry(name: str, sig: str, ret: str, methods: list, desc: str) -> str:
    """Classify the API entry type."""
    ret_lower = ret.lower() if ret else ""
    desc_lower = desc.lower() if desc else ""

    if ret_lower == "shape" or "shape" in ret_lower:
        return "shape"
    if ret_lower == "config" or "config" in ret_lower:
        return "config"
    if "quantity" in ret_lower:
        return "custom_quantity"
    if "(" in sig and ret:
        return "function"
    if any("Set(" in m or "SetRegion(" in m for m in methods):
        if any("Comp(" in m for m in methods):
            return "vector_parameter"
        return "scalar_parameter"
    if any("EvalTo(" in m for m in methods):
        if any("Get(" in m for m in methods):
            return "scalar_output"
        if any("Comp(" in m for m in methods):
            return "vector_field"
        return "quantity"
    if "(" in sig:
        return "function"
    if desc_lower and any(w in desc_lower for w in ("returns", "function")):
        return "math_function"

    return "constant" if not methods else "variable"


def _parse_with_regex(html: str) -> list[dict]:
    """Fallback regex-based parser."""
    entries = []

    # Find anchored entries: <a id="...">text</a>
    pattern = re.compile(
        r'<a\s+id="([^"]+)"[^>]*>(.*?)</a>\s*'
        r'(?:<p>(.*?)</p>)?',
        re.S
    )

    for m in pattern.finditer(html):
        entry_id = m.group(1)
        raw_text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        desc = re.sub(r'<[^>]+>', '', m.group(3) or "").strip()

        if not raw_text or entry_id in ("top", "basics", "advanced"):
            continue

        name, sig, ret = _parse_signature(raw_text)
        if not name:
            continue

        entries.append({
            "name": name,
            "signature": sig,
            "return_type": ret,
            "description": desc,
            "methods": [],
            "examples": [],
            "entry_type": _classify_entry(name, sig, ret, [], desc),
        })

    return entries


def build_section_index(entries: list[dict]) -> dict:
    """Group entries by section for easier consumption."""
    sections = {
        "mesh": [],
        "shapes": [],
        "configs": [],
        "material_parameters": [],
        "excitation": [],
        "output_quantities": [],
        "output_scheduling": [],
        "running": [],
        "spin_currents": [],
        "mfm": [],
        "slicing": [],
        "moving_window": [],
        "extensions": [],
        "custom_quantities": [],
        "math": [],
        "misc": [],
    }

    section_keywords = {
        "mesh": {"SetGridSize", "SetCellSize", "SetMesh", "SetPBC", "EdgeSmooth",
                 "SetGeom", "geom", "EnableDemag", "DemagAccuracy", "OpenBC"},
        "shapes": {"Rect", "Circle", "Cylinder", "Ellipse", "Ellipsoid", "Cuboid",
                   "Sphere", "Superball", "Cone", "Cell", "XRange", "YRange", "ZRange",
                   "Layer", "Layers", "Square", "Triangle", "Universe", "ImageShape",
                   "GrainRoughness"},
        "configs": {"Uniform", "Vortex", "Antivortex", "TwoDomain", "RandomMag",
                    "RandomMagSeed", "VortexWall", "BlochSkyrmion", "NeelSkyrmion",
                    "Conical", "Helical", "HopfionCompactSupport"},
        "material_parameters": {"Msat", "Aex", "alpha", "Ku1", "Ku2", "Kc1", "Kc2",
                                "Kc3", "anisU", "anisC1", "anisC2", "Dind", "Dbulk",
                                "GammaLL", "FrozenSpins", "Temp", "B1", "B2",
                                "NoDemagSpins", "frozenspins"},
        "excitation": {"B_ext"},
        "output_quantities": {"m", "m_full", "B_demag", "B_exch", "B_anis", "B_ext",
                              "B_eff", "B_therm", "B_mel", "B_custom",
                              "E_total", "E_demag", "E_exch", "E_anis", "E_Zeeman",
                              "E_therm", "E_mel", "E_custom",
                              "Edens_total", "Edens_demag", "Edens_exch", "Edens_anis",
                              "Edens_Zeeman", "Edens_therm", "Edens_mel", "Edens_custom",
                              "torque", "LLtorque", "STTorque", "maxTorque",
                              "MaxAngle", "spinAngle", "dt", "LastErr", "PeakErr",
                              "NEval", "geom", "regions",
                              "ExchCoupling", "DindCoupling"},
        "output_scheduling": {"Save", "SaveAs", "AutoSave", "Snapshot", "SnapshotAs",
                              "AutoSnapshot", "TableAdd", "TableAddVar", "TableSave",
                              "TableAutoSave", "TablePrint", "Flush", "Fprintln",
                              "OutputFormat", "FilenameFormat", "SnapshotFormat",
                              "OVF1_TEXT", "OVF1_BINARY", "OVF2_TEXT", "OVF2_BINARY", "DUMP"},
        "running": {"Run", "Steps", "RunWhile", "Relax", "Minimize", "SetSolver",
                    "MaxDt", "MinDt", "FixDt", "MaxErr", "Headroom",
                    "MinimizerStop", "MinimizerSamples", "RelaxTorqueThreshold",
                    "DoPrecess", "step", "t", "ClearPostSteps"},
        "spin_currents": {"J", "Pol", "xi", "Lambda", "EpsilonPrime", "FixedLayer",
                          "FixedLayerPosition", "FreeLayerThickness",
                          "FIXEDLAYER_TOP", "FIXEDLAYER_BOTTOM",
                          "DisableSlonczewskiTorque", "DisableZhangLiTorque",
                          "EdgeCarryShift"},
        "mfm": {"MFM", "MFMLift", "MFMDipole"},
        "slicing": {"Crop", "CropX", "CropY", "CropZ", "CropLayer", "CropRegion"},
        "moving_window": {"Shift", "ShiftGeom", "ShiftM", "ShiftRegions",
                          "ShiftMagL", "ShiftMagR", "ShiftMagU", "ShiftMagD",
                          "TotalShift"},
        "extensions": set(),  # matched by prefix
        "custom_quantities": {"Const", "ConstVector", "Mul", "Madd", "Add", "Div",
                              "Dot", "Cross", "Normalized", "Shifted", "Masked",
                              "MulMV", "Sum", "SumVector", "RunningAverage",
                              "AddFieldTerm", "AddEdensTerm",
                              "RemoveCustomFields", "RemoveCustomEnergies"},
    }

    math_names = {"abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
                  "cbrt", "ceil", "cos", "cosh", "erf", "erfc", "exp", "exp2", "expm1",
                  "floor", "gamma", "hypot", "ilogb", "j0", "j1", "jn",
                  "ldexp", "log", "log10", "log1p", "log2", "logb",
                  "max", "min", "mod", "pow", "pow10", "remainder",
                  "sin", "sinc", "sinh", "sqrt", "tan", "tanh", "trunc",
                  "y0", "y1", "yn",
                  "rand", "randExp", "randInt", "randNorm", "randSeed",
                  "norm", "heaviside", "Sign",
                  "pi", "Mu0", "inf", "true", "false"}

    for entry in entries:
        name = entry["name"]
        placed = False

        # Check extensions first (by prefix)
        if name.startswith("ext_"):
            sections["extensions"].append(entry)
            placed = True
            continue

        # Check math
        if name in math_names or name.lower() in math_names:
            sections["math"].append(entry)
            placed = True
            continue

        # Check keyword sections
        for section_name, keywords in section_keywords.items():
            if name in keywords:
                sections[section_name].append(entry)
                placed = True
                break

        if not placed:
            # Strain tensor components
            if name.startswith("e") and len(name) == 3 and name[1:] in ("xx", "xy", "xz", "yy", "yz", "zz"):
                sections["material_parameters"].append(entry)
            elif name.startswith("F_"):
                sections["output_quantities"].append(entry)
            else:
                sections["misc"].append(entry)

    return sections


def main():
    print("=" * 60)
    print("Official MuMax3 API Scraper")
    print("=" * 60)

    html = fetch_api_page(API_URL)
    entries = parse_api_entries(html)
    print(f"\nExtracted {len(entries)} API entries\n")

    if not entries:
        print("ERROR: No API entries found. Page structure may have changed.")
        return

    # Classify by type
    type_counts = {}
    for e in entries:
        t = e["entry_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Entry types:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Build section index
    sections = build_section_index(entries)
    print(f"\nSections:")
    for name, items in sections.items():
        if items:
            print(f"  {name}: {len(items)} entries")

    # Write output
    output = {
        "source_url": API_URL,
        "scrape_time": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "type_counts": type_counts,
        "entries": entries,
        "sections": {k: v for k, v in sections.items() if v},
    }

    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nOutput: {OUT_FILE}")
    print(f"Total: {len(entries)} API entries scraped")


if __name__ == "__main__":
    main()
