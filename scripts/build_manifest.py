"""生成指定目标域和重复编号的实验 manifest。"""
import argparse
import json
from pathlib import Path

import yaml

from data.load import load_scans
from tasks.build import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["T1", "T2"], default="T1")
    parser.add_argument("--target", default="B0F1")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/v1.yaml").read_text())
    frame, audit = load_scans(config["data_path"], config)
    manifest = build_manifest(frame, audit, config, args.target, args.task, args.repeat)
    directory = Path("outputs/manifests")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.task}_{args.target}_e{args.repeat}.json"
    with path.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, separators=(",", ":"))
    path.with_suffix(".yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"manifest": str(path), **audit,
                      "sources": [d["domain"] for d in manifest["sources"]],
                      "support_scans": len(manifest["target_support"]),
                      "query_scans": len(manifest["target_query"])}, indent=2))


if __name__ == "__main__":
    main()
