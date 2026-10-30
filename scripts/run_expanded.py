"""Persistent expanded study; run dev first and keep test as a separate phase."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from tohe.io import read, write, digest

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/expanded-qwen2.5-7b-cuda.json")
parser.add_argument("--storage-root", default="/home/work/pilot_experiment1")
parser.add_argument("--root", required=True)
parser.add_argument("--source-root", required=True)
parser.add_argument("--tokenizer-python", default=".venv-vllm/bin/python")
parser.add_argument("--phase", choices=["prepare", "dev", "test"], default="dev")
parser.add_argument("--context-limit", type=int, default=16384)
args = parser.parse_args()
cfg = read(args.config)
root, storage, source = (Path(p).resolve() for p in (args.root, args.storage_root, args.source_root))
if not os.path.ismount(storage) or not root.is_relative_to(storage) or root == storage:
    raise ValueError("Experiment ROOT must be a child of the verified mounted storage root")
root.mkdir(parents=True, exist_ok=True)
config_path = root / "experiment-config.json"
if config_path.exists() and read(config_path) != cfg:
    raise ValueError("Existing expanded experiment has a different configuration")
write(config_path, cfg)
run = Path(__file__).with_name("run_experiment.py")


def stage(name, *extra):
    print("STAGE", name, extra, flush=True)
    subprocess.run([sys.executable, str(run), "--stage", name, "--config", str(config_path),
                    "--root", str(root), *extra], check=True)


try:
    write(root / "job-status.json", {"phase": args.phase, "status": "running", "pid": os.getpid()})
    resources = {"disk_free_GiB": shutil.disk_usage(storage).free / 2**30}
    for name in ("memory.current", "memory.max"):
        p = Path("/sys/fs/cgroup") / name
        if p.exists():
            resources[name] = p.read_text().strip()
    resources["gpu"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"]).decode()
    write(root / "resources-at-start.json", resources)
    graph = read(source / "data/graph.json")
    target = root / "data/graph.json"
    if target.exists() and read(target) != graph:
        raise ValueError("Existing source graph differs")
    write(target, graph)
    if not (root / "data/specs.json").exists():
        stage("specs", "--exclude-specs", str(source / "data/specs.json"))
    dataset = read(root / "data/specs.json")
    expected = (cfg["qa"]["dev_graphs"] + cfg["qa"]["test_graphs"]) * 6
    assert len(dataset["questions"]) == expected == 1800
    old = {q["semantic_key"] for q in read(source / "data/specs.json")["questions"]}
    assert not old & {q["semantic_key"] for q in dataset["questions"]}
    assert dataset["metadata"]["bfs_dfs_distinct_contexts"] == 300
    assert dataset["metadata"]["multi_parent_contexts"] >= 30
    write(root / "data/source-exclusion.json", {"source_specs_hash": digest(read(source / "data/specs.json")),
                                               "excluded_questions": len(old)})
    if args.phase != "prepare":
        if args.phase == "dev":
            stage("train")
            stage("generate", "--max-calls", "10800")
        qa = read(root / "data/qa.audited-v3.json")
        if qa["metadata"]["generation_status"] != "complete":
            raise ValueError("Question generation is incomplete; resume dev phase before evaluating")
        endpoint = os.environ[cfg["llm"]["endpoint_env"]]
        model_url = endpoint.rsplit("/chat/completions", 1)[0] + "/models"
        with urllib.request.urlopen(model_url, timeout=30) as response:
            models = json.load(response)["data"]
        model = next(x for x in models if x["id"] == cfg["llm"]["model"])
        if model.get("max_model_len", 0) < args.context_limit:
            raise ValueError(f"Raw coordinates require a server context limit of {args.context_limit}")
        subprocess.run([args.tokenizer_python, str(run.with_name("check_token_budget.py")),
                        "--config", str(config_path), "--qa", str(root / "data/qa.audited-v3.json"),
                        "--checkpoints", str(root / "checkpoints"),
                        "--output", str(root / "token-budget.json"), "--limit", str(args.context_limit)], check=True)
        limit = str(sum(q["split"] == args.phase for q in qa["questions"]) * len(cfg["conditions"]))
        stage("evaluate", "--split", args.phase, "--max-calls", limit)
        stage("report", "--split", args.phase)
    write(root / "job-status.json", {"phase": args.phase, "status": "complete", "pid": os.getpid()})
except Exception as exc:
    write(root / "job-status.json", {"phase": args.phase, "status": "failed", "error": str(exc), "pid": os.getpid()})
    raise
