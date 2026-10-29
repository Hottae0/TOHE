"""Build training graphs using only the facts actually visible to the LLM."""
import re
from pathlib import Path
from .graph import closure
from .io import digest, read, write

SCOPE = "evidence_subgraph_only"
VARIANTS = ("euclidean", "poincare", "v1", "v2")


def local_graph(context):
    aliases = context["nodes"]
    mapping = context["original_ids"]
    if set(aliases) != set(mapping) or len(set(mapping.values())) != len(aliases):
        raise ValueError("Evidence node mapping must be a bijection")
    nodes = sorted(mapping.values())
    if any(u not in mapping or v not in mapping for u, v in context["edges"]):
        raise ValueError("Evidence edge references an invisible node")
    edges = sorted({(mapping[u], mapping[v]) for u, v in context["edges"]})
    # Never read the global graph's closure or cached evidence_hash here.
    graph = {"nodes": nodes, "edges": edges, "closure": closure(nodes, edges),
             "metadata": {"scope": SCOPE}}
    graph["graph_hash"] = digest(graph)
    return graph


def evidence_graphs(dataset, split=None):
    graphs, signatures, splits = {}, {}, {}
    for item in dataset["questions"]:
        context_id = item["context_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", context_id):
            raise ValueError("Unsafe context ID")
        context = item["graph"]
        signature = digest({k: context[k] for k in ("nodes", "edges", "original_ids")})
        if context_id in signatures:
            if signatures[context_id] != signature or splits[context_id] != item["split"]:
                raise ValueError("One context ID must refer to one graph, mapping and split")
        else:
            signatures[context_id] = signature
            splits[context_id] = item["split"]
            graphs[context_id] = local_graph(context)
    return {k: v for k, v in graphs.items() if split is None or splits[k] == split}


def checkpoint_path(root, context_id, variant, seed):
    return Path(root) / context_id / f"{variant}-{seed}.pt"


def check_checkpoint(checkpoint, graph, config, variant, seed):
    if checkpoint.get("embedding_scope") != SCOPE:
        raise ValueError("Global/legacy checkpoint rejected: train evidence-local embeddings first")
    if checkpoint["graph_hash"] != graph["graph_hash"] or checkpoint["nodes"] != graph["nodes"]:
        raise ValueError("Checkpoint does not match the exact visible evidence graph")
    if checkpoint["config"]["training"] != config["training"] or checkpoint["config"]["serializer"] != config["serializer"]:
        raise ValueError("Checkpoint training/serializer config mismatch")
    if checkpoint["variant"] != variant or checkpoint["seed"] != seed:
        raise ValueError("Mislabeled checkpoint")


def train_evidence(dataset, config, root):
    import torch
    from .model import train, embedding_metrics
    if config.get("embedding_scope") != SCOPE:
        raise ValueError("Set embedding_scope=evidence_subgraph_only; use the updated configs")
    graphs = evidence_graphs(dataset)
    results = {}
    for context_id, graph in graphs.items():
        evidence_path = Path(root) / context_id / "evidence.json"
        if evidence_path.exists() and read(evidence_path)["graph_hash"] != graph["graph_hash"]:
            raise ValueError("Existing evidence differs; use a new experiment root")
        write(evidence_path, graph)
        for variant in VARIANTS:
            for seed in config["training"]["seeds"]:
                path = checkpoint_path(root, context_id, variant, seed)
                if path.exists():
                    ckpt = torch.load(path, map_location="cpu", weights_only=True)
                    check_checkpoint(ckpt, graph, config, variant, seed)
                else:
                    print(f"Evidence {context_id}: {len(graph['nodes'])} nodes, {variant}, seed={seed}", flush=True)
                    ckpt = train(graph, config, variant, seed, path)
                metrics = embedding_metrics(graph, ckpt)
                write(path.with_suffix(".metrics.json"), metrics)
                results[f"{context_id}/{variant}/{seed}"] = metrics
    write(Path(root) / "evidence-summary.json", results)
    return results
