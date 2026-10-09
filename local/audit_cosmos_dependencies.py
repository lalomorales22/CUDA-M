#!/usr/bin/env python3
"""Check a pip dry-run report against OSV before installing its exact versions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--allow-advisory", action="append", default=[])
    parser.add_argument("--reason")
    args = parser.parse_args()
    packages = [item["metadata"] for item in json.loads(args.report.read_text())["install"]]
    payload = {"queries": [{"package": {"name": p["name"], "ecosystem": "PyPI"},
                            "version": p["version"]} for p in packages]}
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(packages):
        raise RuntimeError("incomplete OSV response; installation refused")
    findings = [{"package": p["name"], "version": p["version"], "vulns": r["vulns"]}
                for p, r in zip(packages, results) if r.get("vulns")]
    if args.allow_advisory and not args.reason:
        raise RuntimeError("a reviewed advisory exception requires a reason")
    unresolved = [f for f in findings if any(v["id"] not in args.allow_advisory for v in f["vulns"])]
    output = {"packages": [{"name": p["name"], "version": p["version"]} for p in packages],
              "findings": findings, "unresolved": unresolved,
              "reviewed_exceptions": args.allow_advisory, "reason": args.reason,
              "source": "https://api.osv.dev/v1/querybatch"}
    args.report.with_suffix(".security.json").write_text(json.dumps(output, indent=2) + "\n")
    if unresolved:
        print(json.dumps(unresolved, indent=2))
        raise RuntimeError("known vulnerabilities found; installation refused")
    print(f"OSV: {len(packages)} versions checked, {len(findings)} findings, "
          f"{len(args.allow_advisory)} reviewed exceptions, no unresolved findings", flush=True)
    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                        *[p["name"] + "==" + p["version"] for p in packages]], check=True)


if __name__ == "__main__":
    main()
