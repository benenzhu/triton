#!/usr/bin/env python3
"""Annotate Triton MLIR pipeline dump with inline Python source comments.

Usage:
    python annotate_pipeline.py pipeline.txt -o annotated.txt
    python annotate_pipeline.py pipeline.txt  # writes to pipeline.annotated.txt
"""

import argparse
import re
from pathlib import Path

# loc definition patterns
RE_LOC_DEF_FILE = re.compile(r'^#(\w+)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)')
RE_LOC_DEF_NAMED = re.compile(r'^#(\w+)\s*=\s*loc\("[^"]+"\(#(\w+)\)\)')
# loc reference in IR body
RE_LOC_REF = re.compile(r'loc\(#(\w+)\)')
# inline loc like loc("file":line:col)
RE_LOC_INLINE = re.compile(r'loc\("([^"]+)":(\d+):(\d+)\)')
# pass header
RE_PASS_HEADER = re.compile(r'^// -----// (IR Dump .+) //-----')


def load_source_cache(file_path):
    """Load and cache source file lines."""
    try:
        lines = Path(file_path).read_text().splitlines()
        return lines
    except (FileNotFoundError, PermissionError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Annotate MLIR pipeline with Python source")
    parser.add_argument("input", type=Path, help="pipeline.txt from triton_wrapper")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output file")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(".annotated.txt")

    lines = args.input.read_text().splitlines()

    # --- Pass 1: collect all loc definitions ---
    raw_locs = {}  # id -> {file, line} or {parent: id}
    for line in lines:
        m = RE_LOC_DEF_FILE.match(line)
        if m:
            raw_locs[m.group(1)] = {"file": m.group(2), "line": int(m.group(3))}
            continue
        m = RE_LOC_DEF_NAMED.match(line)
        if m:
            raw_locs[m.group(1)] = {"parent": m.group(2)}

    # --- Resolve named refs ---
    resolved = {}

    def resolve(loc_id):
        if loc_id in resolved:
            return resolved[loc_id]
        entry = raw_locs.get(loc_id)
        if entry is None:
            return None
        if "file" in entry:
            resolved[loc_id] = entry
            return entry
        parent = resolve(entry["parent"])
        if parent:
            resolved[loc_id] = parent
        return parent

    for loc_id in raw_locs:
        resolve(loc_id)

    # --- Source file cache ---
    source_cache = {}

    def get_source_line(file_path, line_no):
        if file_path not in source_cache:
            source_cache[file_path] = load_source_cache(file_path)
        src = source_cache[file_path]
        if src is None or line_no < 1 or line_no > len(src):
            return None
        return src[line_no - 1]

    # --- Pass 2: annotate ---
    out = []
    prev_source_tag = None

    for line in lines:
        # Pass headers: reset context, keep as-is
        if RE_PASS_HEADER.match(line):
            prev_source_tag = None
            out.append(line)
            continue

        # Skip loc definition lines
        if RE_LOC_DEF_FILE.match(line) or RE_LOC_DEF_NAMED.match(line):
            out.append(line)
            continue

        # Try to find source mapping
        source_info = None
        m = RE_LOC_REF.search(line)
        if m:
            loc = resolved.get(m.group(1))
            if loc:
                source_info = (loc["file"], loc["line"])
        else:
            m = RE_LOC_INLINE.search(line)
            if m:
                source_info = (m.group(1), int(m.group(2)))

        # Insert source comment if it changed
        if source_info and source_info != prev_source_tag:
            file_path, line_no = source_info
            src_text = get_source_line(file_path, line_no)
            short_file = Path(file_path).name
            if src_text is not None:
                out.append(f"  // ---- {short_file}:{line_no}  {src_text.strip()}")
            else:
                out.append(f"  // ---- {short_file}:{line_no}  <source unavailable>")
            prev_source_tag = source_info

        out.append(line)

    args.output.write_text("\n".join(out) + "\n")
    print(f"Written to {args.output}")
    print(f"  {len(lines)} lines in -> {len(out)} lines out")
    print(f"  {len(resolved)} locs resolved, {len(source_cache)} source files loaded")


if __name__ == "__main__":
    main()
