"""按任务协议生成并保存训练、适应和评价所用的行号。"""
import math

import numpy as np

from tasks.stable import model_seed, stable_order


def position_index(frame, domain):
    selected = frame[frame.domain == domain]
    return {p: {g: v.index.tolist() for g, v in rows.groupby("group", sort=True)}
            for p, rows in selected.groupby("position", sort=True)}


def partition(index, config, domain):
    """将域内位置分为候选集和 query 集。"""
    positions = stable_order(index, config, "position_split", domain, -1)
    nq = math.ceil(config["task"]["query_fraction"] * len(positions))
    return positions[nq:], positions[:nq]


def all_rows(index, positions):
    return sorted(row for p in positions for rows in index[p].values() for row in rows)


def sample_episode(index, positions, count, config, domain, counter, purpose):
    """选择位置、采集组及扫描，生成一次 episode。"""
    if len(positions) < count:
        raise ValueError(f"{domain}: not enough positions for the fixed V1 episode")
    selected = stable_order(positions, config, purpose + "_position", domain, counter)[:count]
    result = []
    for p in selected:
        groups = stable_order(index[p], config, purpose + "_group", domain, counter)
        for g in groups[:config["task"]["scans_per_position"]]:
            result.append(stable_order(index[p][g], config, purpose + "_row", domain, counter)[0])
    return result


def uniform_batches(index, positions, config, domain, repeat, purpose, steps, size):
    """生成位置、采集组、扫描三级均匀抽样的 batch 行号。"""
    rng = np.random.Generator(np.random.PCG64(model_seed(config, purpose, domain, repeat, purpose)))
    positions = sorted(positions)
    groups = {p: sorted(index[p]) for p in positions}
    result = []
    for _ in range(steps):
        batch = []
        for _ in range(size):
            p = positions[int(rng.integers(len(positions)))]
            g = groups[p][int(rng.integers(len(groups[p])))]
            rows = index[p][g]
            batch.append(rows[int(rng.integers(len(rows)))])
        result.append(batch)
    return result


def build_manifest(frame, audit, config, target, task, repeat):
    if config["profile"] != "N":
        raise NotImplementedError("This first-stage implementation supports natural profile N only")
    domains = frame[["domain", "BUILDINGID", "FLOOR"]].drop_duplicates().sort_values(
        ["BUILDINGID", "FLOOR"])
    building = int(domains.loc[domains.domain == target, "BUILDINGID"].iloc[0])
    if task == "T1":
        source = domains[(domains.BUILDINGID == building) & (domains.domain != target)].domain.tolist()
    elif task == "T2":
        source = domains[domains.BUILDINGID != building].domain.tolist()
    else:
        raise ValueError("This cross-space manifest supports T1/T2 only; T0 uses its target pools")
    index = position_index(frame, target)
    pc, pq = partition(index, config, target)
    support = sample_episode(index, pc, config["task"]["k"], config, target, repeat, "support")
    manifest = {"protocol": config["protocol"], "profile": config["profile"],
                "task": task, "target": target, "target_building": building,
                "repeat": repeat, "k": config["task"]["k"],
                "r": config["task"]["scans_per_position"], "audit": audit,
                "target_support": support, "target_query": all_rows(index, pq),
                "target_candidate": all_rows(index, pc), "sources": []}
    manifest["t0_batches"] = uniform_batches(
        index, pc, config, target, repeat, "t0_sampling",
        config["baseline"]["t0_steps"], config["baseline"]["t0_batch_size"])
    for domain in source:
        index = position_index(frame, domain)
        pc, pq = partition(index, config, domain)
        item = {"domain": domain, "fit_rows": all_rows(index, pc),
                "query_rows": all_rows(index, pq), "rounds": []}
        item["ae_batches"] = uniform_batches(
            index, pc, config, domain, repeat, "ae_sampling",
            config["ae"]["steps"], config["ae"]["source_batch_size"])
        for round_id in range(config["meta"]["rounds"]):
            counter = [repeat, round_id]
            item["rounds"].append({
                "support": sample_episode(index, pc, config["task"]["k"], config, domain,
                                          counter, "source_support"),
                "query": sample_episode(index, pq, config["task"]["source_query_positions"],
                                        config, domain, counter, "source_query")})
        manifest["sources"].append(item)
    return manifest
