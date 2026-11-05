"""Check regenerated data against the committed copy (run after generate_data.py).

CSVs and config must match byte for byte. Golden values must match within each
case's own tolerance: BLAS builds differ in the last digits, so a byte diff of
golden.json would fail on harmless rounding.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    exact = subprocess.run(
        ["git", "diff", "--exit-code", "--stat", "--", ".", ":!golden/golden.json"], cwd=HERE
    )
    committed = subprocess.run(
        ["git", "show", "HEAD:data/golden/golden.json"],
        cwd=HERE, check=True, capture_output=True, text=True,
    ).stdout
    old = {c["name"]: c for c in json.loads(committed)["cases"]}
    new = {c["name"]: c for c in json.loads((HERE / "golden" / "golden.json").read_text())["cases"]}
    bad = sorted(set(old) ^ set(new))
    for name in set(old) & set(new):
        a, b = old[name], new[name]
        if a["inputs"] != b["inputs"] or abs(a["expect"]["value"] - b["expect"]["value"]) > a["tol"]:
            bad.append(name)
    print(f"golden cases compared: {len(new)}, outside tolerance or changed: {len(bad)}")
    for name in bad[:20]:
        print(f"  {name}")
    return 1 if bad or exact.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
