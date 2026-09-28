"""读取 manifest，运行五种方法并保存权重、预测和指标。"""
import argparse
import json
import os
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from data.load import load_scans, query_features, support_data
from data.preprocess import Preprocessor
from evaluation.metrics import evaluate, predict_checkpoints, uncovered_ap_fraction
from models.wknn import predict_wknn
from models.femloc import shared_module
from training.adapt import adapt, prepare_target
from training.baselines import adapt_mlp, make_mlp, train_source_mlp, train_t0_mlp
from training.source import meta_train


def save_run(name, run, features, frame, output, config, common):
    rows = []
    for step, prediction in predict_checkpoints(run, features, config).items():
        stats, table = evaluate(frame, features["row_ids"], prediction)
        table.to_csv(output / f"{name}_step{step}.csv", index=False)
        rows.append({**common, "method": name, "step": step, **stats,
                     "adapt_seconds": run["seconds"],
                     "uncovered_ap_fraction": uncovered_ap_fraction(features, run["preprocessor"])})
    torch.save({"snapshots": run["snapshots"], "preprocessor": run["preprocessor"].state(),
                "support_row_ids": run["support_row_ids"], "query_row_ids": features["row_ids"],
                "compact": run["compact"], "config": config}, output / f"{name}.pt")
    (output / f"{name}_training.json").write_text(json.dumps(run["losses"]))
    print(f"{name}: position mean={rows[-1]['position_mean']:.4f}, steps={rows[-1]['step']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", choices=["phase1", "full"], default="phase1")
    parser.add_argument("--source-checkpoint", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.manifest.with_suffix(".yaml").read_text())
    manifest = json.loads(args.manifest.read_text())
    torch.set_num_threads(config["torch_threads"])
    output = Path("outputs") / args.phase / args.manifest.stem
    output.mkdir(parents=True, exist_ok=False)
    rounds = 1 if args.phase == "phase1" else config["meta"]["rounds"]
    common = {"phase": args.phase, "task": manifest["task"], "target": manifest["target"],
              "repeat": manifest["repeat"], "source_rounds": rounds,
              "source_reused_from": str(args.source_checkpoint) if args.source_checkpoint else None}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "run.json").write_text(json.dumps({**common, "manifest": str(args.manifest.resolve()),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": config["device"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(config["device"])}, indent=2))
    frame, audit = load_scans(config["data_path"], config)
    if audit != manifest["audit"]:
        raise ValueError("Training CSV does not match the saved manifest")
    support = support_data(frame, manifest["target_support"])
    features = query_features(frame, manifest["target_query"])
    domains = [item["domain"] for item in manifest["sources"]]
    prep = Preprocessor.fit(support, config)
    prediction = predict_wknn(support, features, prep, config)
    stats, table = evaluate(frame, features["row_ids"], prediction)
    table.to_csv(output / "WKNN.csv", index=False)
    rows = [{**common, "method": "WKNN", "step": 0, **stats,
             "uncovered_ap_fraction": uncovered_ap_fraction(features, prep)}]
    (output / "WKNN_inputs.json").write_text(json.dumps({"support_row_ids": support["row_ids"],
                                                         "query_row_ids": features["row_ids"]}))
    print(f"WKNN: position mean={stats['position_mean']:.4f}", flush=True)
    scratch = adapt_mlp(None, support, domains, manifest["target"], manifest["repeat"], config)
    rows += save_run("MLP-Scratch", scratch, features, frame, output, config, common)
    if args.source_checkpoint:
        source = torch.load(args.source_checkpoint, map_location=config["device"], weights_only=True)
        parent = json.loads(Path(source["manifest"]).read_text())
        if (source["config"] != config or source["source_rounds"] != rounds
                or source["repeat"] != manifest["repeat"] or parent["sources"] != manifest["sources"]
                or parent["audit"] != manifest["audit"]):
            raise ValueError("Source checkpoint must have the identical source data and configuration")
        source_mlp = make_mlp(config, domains, manifest["repeat"], domains[0])
        source_mlp.load_state_dict(source["mlp"])
        shared = shared_module(config, domains, manifest["repeat"])
        initial = deepcopy(shared)
        shared.load_state_dict(source["shared"])
        initial.load_state_dict(source["initial"])
        ft_log, meta_log = [], []
        print(f"Source models loaded: {args.source_checkpoint}", flush=True)
    else:
        source_mlp, ft_log = train_source_mlp(frame, manifest, config, rounds)
    ft = adapt_mlp(source_mlp, support, domains, manifest["target"], manifest["repeat"], config)
    rows += save_run("MLP-FT", ft, features, frame, output, config, common)
    if args.source_checkpoint is None:
        shared, initial, clients, meta_log = meta_train(frame, manifest, config, rounds)
    torch.save({"shared": shared.state_dict(), "initial": initial.state_dict(),
                "mlp": source_mlp.state_dict(), "source_domains": domains,
                "manifest": str(args.manifest.resolve()), "repeat": manifest["repeat"],
                "source_rounds": rounds, "config": config}, output / "source_shared.pt")
    (output / "source_training.json").write_text(json.dumps({"meta": meta_log, "mlp_ft": ft_log}))
    private = prepare_target(support, manifest["target"], manifest["repeat"], config)
    for name, shared_model in [("FeMLoc-RI", initial), ("FeMLoc", shared)]:
        run = adapt(shared_model, support, private, config)
        rows += save_run(name, run, features, frame, output, config, common)
    if args.phase == "phase1":
        t0 = train_t0_mlp(frame, manifest, config)
        rows += save_run("T0-MLP", t0, features, frame, output, config, {**common, "task": "T0"})
        data = support_data(frame, manifest["target_candidate"])
        prediction = predict_wknn(data, features, Preprocessor.fit(data, config), config)
        stats, table = evaluate(frame, features["row_ids"], prediction)
        table.to_csv(output / "T0-WKNN.csv", index=False)
        rows.append({**common, "task": "T0", "method": "T0-WKNN", "step": 0, **stats})
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "completed.json").write_text(json.dumps({**common, "status": "completed", "audit": audit}, indent=2))
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
