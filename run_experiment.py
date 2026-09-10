#!/usr/bin/env python3
"""Run one experiment from a JSON/YAML config.

    python run_experiment.py --config configs/killtest.json --device cuda:0
    python run_experiment.py --json '{"name":"x","steps":100}' --device cuda:1
"""
import argparse
import json
from pathlib import Path

from crynexa.train import RunConfig, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--json")
    ap.add_argument("--device")
    ap.add_argument("--name")
    ap.add_argument("--out")
    a = ap.parse_args()

    d = {}
    if a.config:
        p = Path(a.config)
        d = (json.loads(p.read_text()) if p.suffix == ".json"
             else __import__("yaml").safe_load(p.read_text()))
    if a.json:
        d.update(json.loads(a.json))
    if a.device:
        d["device"] = a.device
    if a.name:
        d["name"] = a.name

    cfg = RunConfig(**d)
    res = run(cfg, out_dir=a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
