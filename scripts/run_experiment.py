"""Server entry point. Stages are explicit; only generate/evaluate call an LLM."""
import argparse
import os
import re
import urllib.request
from pathlib import Path
from tohe.io import read, write

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True, choices=["prepare", "train", "specs", "generate", "evaluate", "report"])
parser.add_argument("--config", default="configs/gpt4o-mini.json")
parser.add_argument("--root", default="runs/evidence-local")
parser.add_argument("--split", choices=["dev", "test"], default="dev")
parser.add_argument("--max-calls", type=int, default=0, help="Required positive limit for LLM stages; excludes transient HTTP retries")
args = parser.parse_args()
cfg, root = read(args.config), Path(args.root)
graph_path, qa_path = root / "data/graph.json", root / "data/qa.json"
specs_path, checkpoints = root / "data/specs.json", root / "checkpoints"
model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", cfg["llm"].get("model") or "unset")
run_dir = root / "evaluation" / model_slug / args.split

if args.stage == "prepare":
    from tohe.data import prepare
    revision = "dff101b5089a635a751aa45e3a6cb21670a3588e"
    url = f"https://raw.githubusercontent.com/facebookresearch/poincare-embeddings/{revision}/wordnet/mammals_filter.txt"
    target = root / "data/mammals_filter.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    if graph_path.exists():
        raise FileExistsError("Graph already prepared; use existing artifact or a new --root")
    graph = prepare(graph_path, target, download=True, nltk_data=str(root / "data/nltk"))
    write(root / "data/source.json", {"filter_url": url, "upstream_revision": revision,
                                      "graph_hash": graph["graph_hash"]})
    print(f"Prepared {len(graph['nodes'])} nodes and {len(graph['closure'])} training pairs")
elif args.stage == "train":
    from tohe.evidence import train_evidence
    if not specs_path.exists():
        raise FileNotFoundError("Run --stage specs first: evidence graphs must be fixed before training")
    train_evidence(read(specs_path), cfg, checkpoints)
elif args.stage == "specs":
    from tohe.qa import make_specs
    if specs_path.exists():
        raise FileExistsError("Specs already exist; keep one frozen question set for both evaluators")
    result = make_specs(read(graph_path), cfg, specs_path)
    print(result["metadata"])
elif args.stage == "generate":
    from tohe.pipeline import generate_questions
    if args.max_calls <= 0:
        parser.error("Set --max-calls for actual LLM generation")
    result = generate_questions(read(specs_path), cfg, qa_path, args.max_calls)
    print(result["metadata"])
elif args.stage == "evaluate":
    from tohe.pipeline import evaluate
    if args.max_calls <= 0:
        parser.error("Set --max-calls for actual LLM evaluation")
    print(evaluate(read(qa_path), cfg, run_dir, checkpoints, args.split, args.max_calls))
else:
    from tohe.report import report
    print(report(run_dir, qa_path))
