"""汇总已完成实验，并在全部重复齐备后计算建筑等权的总体结果。"""
import json
from pathlib import Path

import pandas as pd
import yaml


def main():
    config = yaml.safe_load(Path("configs/v1.yaml").read_text())
    records = []
    for complete in sorted(Path("outputs/full").glob("*/completed.json")):
        status = json.loads(complete.read_text())
        frame = pd.read_csv(complete.parent / "metrics.csv")
        frame = frame[(frame.method == "WKNN") | (frame.step == config["adapt"]["steps"])].copy()
        frame["building"] = int(status["target"].split("F")[0][1:])
        records.append(frame)
    table = pd.concat(records, ignore_index=True)
    directory = Path("outputs/summary")
    directory.mkdir(exist_ok=True)
    table.to_csv(directory / "completed_runs.csv", index=False)
    target = table.groupby(["task", "target", "method"]).position_mean.agg(["mean", "std", "count"])
    target.rename(columns={"count": "n_repeats"}).to_csv(directory / "target_summary.csv")
    progress = table[["task", "target", "repeat"]].drop_duplicates().groupby("task").size().to_dict()
    print(json.dumps({"completed": progress, "expected_per_task": 13 * config["task"]["repeats"]}))
    counts = table.groupby(["task", "repeat", "method"]).target.nunique()
    complete = (len(counts) == 2 * config["task"]["repeats"] * 5 and counts.eq(13).all())
    if complete:
        buildings = table.groupby(["task", "repeat", "method", "building"]).position_mean.mean()
        repeats = buildings.groupby(["task", "repeat", "method"]).mean()
        repeats.groupby(["task", "method"]).agg(["mean", "std"]).to_csv(directory / "task_summary.csv")
        pivot = table.pivot(index=["task", "target", "repeat"], columns="method", values="position_mean")
        differences = {name: pivot.FeMLoc - pivot[name] for name in
                       ["WKNN", "MLP-Scratch", "MLP-FT", "FeMLoc-RI"]}
        pd.DataFrame(differences).to_csv(directory / "paired_differences.csv")


if __name__ == "__main__":
    main()
