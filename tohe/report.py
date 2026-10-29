import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import read, rows, write


def report(run_dir, dataset_path):
    run_dir = Path(run_dir)
    data = rows(run_dir / "predictions.jsonl")
    if not data:
        raise ValueError("No real LLM predictions; dry-run prompts are not results")
    manifest = read(run_dir / "manifest.json")
    dataset = read(dataset_path)
    from .io import digest
    if digest(dataset) != manifest["dataset_hash"]:
        raise ValueError("Report dataset does not match evaluated dataset")
    expected = {q["id"] for q in dataset["questions"] if q["split"] == manifest["split"]}
    groups = defaultdict(list)
    for row in data:
        groups[(row["condition"], row["seed"], "all", "all")].append(row)
        if row["strata"].get("answer_empty") is not None:
            label = f"{row['strata']['task']}/empty={row['strata']['answer_empty']}"
            groups[(row["condition"], row["seed"], "task_answer_empty", label)].append(row)
        for name in ("task", "node_count", "answer_size", "k", "shortest_path", "answer_empty", "has_multi_parent", "bfs_dfs_distinct"):
            value = row["strata"].get(name)
            if value is not None:
                groups[(row["condition"], row["seed"], name, str(value))].append(row)
    table = []
    for (condition, seed, stratum, value), group in sorted(groups.items(), key=lambda p: str(p[0])):
        token_counts = [r["usage"]["prompt_tokens"] for r in group
                        if r.get("usage") and "prompt_tokens" in r["usage"]]
        f1s = [r["set_f1"] for r in group if r["set_f1"] is not None]
        table.append({"condition": condition, "seed": seed, "stratum": stratum, "value": value,
                      "n": len(group), "accuracy": sum(r["correct"] for r in group) / len(group),
                      "invalid_rate": sum(not r["valid"] for r in group) / len(group),
                      "unknown_id_rate": sum(r.get("error_type") == "unknown_node_id" for r in group) / len(group),
                      "answer_type_error_rate": sum(r.get("error_type") == "answer_type" for r in group) / len(group),
                      "parse_error_rate": sum(r.get("error_type") == "invalid_json_or_missing_answer" for r in group) / len(group),
                      "set_f1": float(np.mean(f1s)) if f1s else None,
                      "mean_prompt_tokens": float(np.mean(token_counts)) if token_counts else None,
                      "token_count_coverage": len(token_counts) / len(group)})
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    baseline = {r["question_id"]: r for r in data if r["condition"] == "direct_id"}
    comparisons = []
    rng = np.random.default_rng(2026)
    for condition, seed in sorted({(r["condition"], r["seed"]) for r in data}, key=str):
        group = [r for r in data if r["condition"] == condition and r["seed"] == seed]
        complete = {r["question_id"] for r in group} == expected and set(baseline) == expected
        by_graph = defaultdict(list)
        for row in group:
            base = baseline.get(row["question_id"])
            if base:
                by_graph[row["context_id"]].append(float(row["correct"]) - float(base["correct"]))
        deltas = np.array([np.mean(v) for v in by_graph.values()])
        interval = None
        if complete and len(deltas) > 1:
            bootstrap = rng.choice(deltas, (2000, len(deltas)), replace=True).mean(1)
            interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        comparisons.append({"condition": condition, "seed": seed, "complete": complete,
                            "paired_graphs": len(deltas),
                            "accuracy_delta_vs_direct_id": float(deltas.mean()) if len(deltas) else None,
                            "graph_bootstrap_95_interval": interval})
    selected = [q for q in dataset["questions"] if q["split"] == manifest["split"]]
    constant_correct = sum(q["gold"] is False if isinstance(q["gold"], bool) else q["gold"] == [] for q in selected)
    result = {"comparisons": comparisons,
              "fixed_response_baseline": {"policy": "false for reachability; [] for set tasks",
                                           "correct": constant_correct, "n": len(selected),
                                           "accuracy": constant_correct / len(selected)},
              "limitations": [
        "Intervals conditional on the observed evidence collection; contexts can share entities.",
        "Seed-specific results are reported separately, never counted as independent questions.",
        "No confidence interval is reported for incomplete runs."]}
    write(run_dir / "paired_comparisons.json", result)
    return result
