# -*- coding: utf-8 -*-
"""
verify_notebooks.py - runs the four notebooks the way Colab does (a fresh Python kernel, top to
bottom), saves their outputs into the notebook files, and checks them:

  - no cell raises an error, and nothing is printed to stderr (warnings would confuse students);
  - every number in a `# CHECK` comment appears in that cell's output;
  - the notebooks contain no em or en dashes.

The first cell of notebooks 1 to 3 reads data/ when it exists next to the working folder, so the
notebooks run from the course folder against the local data. Notebook 4 names the GitHub address
directly; it runs from a temporary copy that points at the local folder, and its outputs are
copied back into the original.

    python3 scripts/verify_notebooks.py            # all four
    python3 scripts/verify_notebooks.py 2 3        # some of them
"""
import copy
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_notebooks import NB_NAMES, RAW  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def output_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        elif "data" in o:
            d = o["data"]
            parts.append(d.get("text/plain", ""))
            parts.append(re.sub(r"<[^>]+>", " ", d.get("text/html", "")))
        elif o.get("output_type") == "error":
            parts.append("ERROR " + o.get("ename", "") + ": " + o.get("evalue", ""))
    return "\n".join(p if isinstance(p, str) else "".join(p) for p in parts)


def check_cell(cell):
    """Numbers and dates of the # CHECK lines that are missing from the output."""
    out = output_text(cell)
    flat = out.replace(",", "")
    missing = []
    for line in cell.source.splitlines():
        if "# CHECK" not in line:
            continue
        text = line.split("# CHECK", 1)[1]
        text = re.sub(r"for the course data:?", "", text)
        for d in DATE.findall(text):
            if d not in out:
                missing.append(d)
        text = DATE.sub(" ", text)
        text = re.sub(r"\b\d{4}-\d{2}\b", " ", text)               # year spans such as 2023-24
        for n in NUM.findall(text):
            n = n.replace(",", "").rstrip(".")
            if not n or n in ("-",):
                continue
            core = n.lstrip("-")
            if re.fullmatch(r"20[2-3]\d", core):                      # a year named in words ("WAPE on 2025")
                continue
            if core in ("0", "1", "2", "3", "4", "5") and "." not in n:   # small counts in words ("5 weeks") are context
                continue
            if n not in flat and core not in flat:
                missing.append(n)
    return missing


def run(n):
    path = ROOT / "notebooks" / NB_NAMES[n]
    nb = nbformat.read(path, as_version=4)
    runnable = copy.deepcopy(nb)
    for c in runnable.cells:
        if c.cell_type == "code":
            c.source = c.source.replace(RAW, "data")
    t0 = time.time()
    client = NotebookClient(runnable, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    secs = time.time() - t0
    problems = []
    for i, (orig, done) in enumerate(zip(nb.cells, runnable.cells)):
        if orig.cell_type != "code":
            continue
        orig.outputs = done.outputs
        orig.execution_count = done.execution_count
        for o in done.outputs:
            if o.get("output_type") == "error":
                problems.append(f"cell {i}: error {o.get('ename')}")
            if o.get("output_type") == "stream" and o.get("name") == "stderr":
                problems.append(f"cell {i}: stderr: {o.get('text', '')[:160]!r}")
        for m in check_cell(done):
            problems.append(f"cell {i}: CHECK value {m} not in the output")
    text = "\n".join(c.source for c in nb.cells)
    if chr(0x2014) in text or chr(0x2013) in text:
        problems.append("an em or en dash in the notebook")
    nbformat.write(nb, path)
    return secs, problems


if __name__ == "__main__":
    which = [int(a) for a in sys.argv[1:]] or sorted(n for n in NB_NAMES if (ROOT / "notebooks" / NB_NAMES[n]).exists())
    bad = 0
    for n in which:
        secs, problems = run(n)
        print(f"notebook {n} ({NB_NAMES[n]}): ran in {secs:.0f} s, {len(problems)} problem(s)")
        for p in problems:
            print("   ", p)
        bad += len(problems)
    print("ALL NOTEBOOK CHECKS PASSED" if not bad else f"{bad} PROBLEM(S)")
    sys.exit(1 if bad else 0)
