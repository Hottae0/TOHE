import argparse
import json
from pathlib import Path
from .io import read, write


def main():
    parser = argparse.ArgumentParser(description="TOHE WordNet experiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--output", required=True)
    data.add_argument("--filter", required=True)
    data.add_argument("--reference-csv")
    data.add_argument("--download", action="store_true")
    data.add_argument("--nltk-data")
    fixture = sub.add_parser("synthetic")
    fixture.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--graph", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--condition", choices=["euclidean", "poincare", "v1", "v2"], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", required=True)
    metrics = sub.add_parser("embedding-metrics")
    metrics.add_argument("--graph", required=True)
    metrics.add_argument("--checkpoint", required=True)
    metrics.add_argument("--output", required=True)
    specs = sub.add_parser("qa-specs")
    specs.add_argument("--graph", required=True)
    specs.add_argument("--config", required=True)
    specs.add_argument("--output", required=True)
    generate = sub.add_parser("generate-qa")
    generate.add_argument("--specs", required=True)
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-calls", type=int, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--qa", required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoints", required=True)
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--split", choices=["dev", "test"], required=True)
    evaluate.add_argument("--max-calls", type=int, required=True)
    evaluate.add_argument("--dry-run", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--qa", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        result = prepare(args.output, args.filter, args.reference_csv, args.download, args.nltk_data)
        print(json.dumps({"nodes": len(result["nodes"]), "direct_edges": len(result["edges"]),
                          "training_pairs": len(result["closure"]), "metadata": result["metadata"]}, indent=2))
    elif args.command == "synthetic":
        from .data import synthetic
        synthetic(args.output)
    elif args.command == "train":
        from .model import train
        train(read(args.graph), read(args.config), args.condition, args.seed, args.output)
    elif args.command == "embedding-metrics":
        import torch
        from .model import embedding_metrics
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        graph = read(args.graph)
        if checkpoint["graph_hash"] != graph["graph_hash"]:
            raise ValueError("Wrong graph for checkpoint")
        result = embedding_metrics(graph, checkpoint)
        write(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "qa-specs":
        from .qa import make_specs
        result = make_specs(read(args.graph), read(args.config), args.output)
        print(json.dumps(result["metadata"], indent=2))
    elif args.command == "generate-qa":
        from .pipeline import generate_questions
        result = generate_questions(read(args.specs), read(args.config), args.output, args.max_calls)
        print(json.dumps(result["metadata"], indent=2))
    elif args.command == "evaluate":
        from .pipeline import evaluate
        result = evaluate(read(args.qa), read(args.config), args.run_dir, args.checkpoints,
                          args.split, args.max_calls, args.dry_run)
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .report import report
        print(json.dumps(report(args.run_dir, args.qa), indent=2))


if __name__ == "__main__":
    main()
