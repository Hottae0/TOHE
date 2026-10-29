import random
from collections import deque
from .graph import adjacency, answer, path_length, traversal
from .io import digest, write


TASKS = {
    "reachability": "Whether a directed path of at least one edge exists from source to target.",
    "common_parents": "The complete intersection of the two nodes' direct outgoing is_a neighbors.",
    "exact_k": "All nodes reachable by a directed path of exactly k edges; this does NOT mean shortest-path distance k."
}


def semantic_key(graph, spec):
    """Independent of anonymous IDs, wording, answer, and evidence context."""
    original = graph["original_ids"]
    if spec["type"] == "reachability":
        key = (spec["type"], original[spec["source"]], original[spec["target"]])
    elif spec["type"] == "common_parents":
        key = (spec["type"], *sorted([original[spec["left"]], original[spec["right"]]]))
    else:
        key = (spec["type"], original[spec["source"]], spec["k"])
    return digest(key)


def _context(graph, size, rng, anchor=None):
    neighbors = adjacency(graph["nodes"], graph["edges"])
    for u, v in graph["edges"]:
        neighbors[v].add(u)
    initial = [rng.choice(graph["nodes"])] if anchor is None else [anchor] + sorted(
        v for u, v in graph["edges"] if u == anchor)
    seen, queue = set(), deque(initial)
    while queue and len(seen) < size:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        others = sorted(neighbors[node] - seen)
        rng.shuffle(others)
        queue.extend(others)
    if len(seen) != size:
        return None
    originals = sorted(seen)
    rng.shuffle(originals)
    mapping = {node: f"N{i:03d}" for i, node in enumerate(originals)}
    edges = sorted((mapping[u], mapping[v]) for u, v in graph["edges"] if u in seen and v in seen)
    return {"nodes": sorted(mapping.values()), "edges": edges,
            "original_ids": {alias: node for node, alias in mapping.items()},
            "evidence_hash": digest(sorted((u, v) for u, v in graph["edges"] if u in seen and v in seen))}


def _specs(graph, rng, context_index, used):
    nodes = graph["nodes"]
    pairs = [(u, v) for u in nodes for v in nodes if u != v]
    rng.shuffle(pairs)
    output = []
    for truth in (True, False):
        found = None
        for u, v in pairs:
            spec = {"type": "reachability", "source": u, "target": v}
            if answer(graph, spec) == truth and semantic_key(graph, spec) not in used:
                found = spec
                break
        if found is None:
            return None
        output.append(found)
    for nonempty in (True, False):
        found = None
        for u, v in pairs:
            spec = {"type": "common_parents", "left": min(u, v), "right": max(u, v)}
            if bool(answer(graph, spec)) == nonempty and semantic_key(graph, spec) not in used:
                found = spec
                break
        if found is None:
            return None
        output.append(found)
    for k in (2, 3):
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        nonempty = (context_index + k) % 2 == 0
        found = next(({"type": "exact_k", "source": u, "k": k} for u in shuffled
                      if bool(answer(graph, {"type": "exact_k", "source": u, "k": k})) == nonempty
                      and semantic_key(graph, {"type": "exact_k", "source": u, "k": k}) not in used), None)
        # Empty exact-k queries can exhaust their unique source-node pool in a
        # small taxonomy. Keep k and semantic uniqueness; report answer strata.
        if found is None:
            found = next(({"type": "exact_k", "source": u, "k": k} for u in shuffled
                          if semantic_key(graph, {"type": "exact_k", "source": u, "k": k}) not in used), None)
        if found is None:
            return None
        output.append(found)
    return output


def make_specs(graph, config, output, exclude_dataset=None):
    cfg = config["qa"]
    if cfg["questions_per_graph"] != 6:
        raise ValueError("The frozen task design uses six questions per evidence graph")
    if max(cfg["nodes_per_graph"]) > len(graph["nodes"]):
        raise ValueError("Requested evidence graph exceeds source graph size")
    rng = random.Random(config["seed"])
    excluded_questions = (exclude_dataset or {}).get("questions", [])
    excluded_keys = {q["semantic_key"] for q in excluded_questions}
    excluded_evidence = {q["graph"]["evidence_hash"] for q in excluded_questions}
    dataset, seen, used, attempts = [], set(excluded_evidence), set(excluded_keys), 0
    total = cfg["dev_graphs"] + cfg["test_graphs"]
    split_labels = ["dev"] * cfg["dev_graphs"] + ["test"] * cfg["test_graphs"]
    rng.shuffle(split_labels)
    direction = cfg.get("serialization_traversal", "outgoing")
    if direction not in ("outgoing", "weak"):
        raise ValueError("Unknown serialization traversal direction")
    quota = cfg.get("multi_parent_graphs", 0)
    if not 0 <= quota <= total:
        raise ValueError("multi_parent_graphs must be between zero and total contexts")
    source_adj = adjacency(graph["nodes"], graph["edges"])
    anchors = sorted(n for n in graph["nodes"] if len(source_adj[n]) > 1)
    if quota and not anchors:
        raise ValueError("Source has no multi-parent nodes; cannot fill structural quota")
    # Spread diagnostic contexts over both splits and graph sizes.
    diagnostic_sizes = cfg.get("multi_parent_nodes_per_graph", cfg["nodes_per_graph"])
    eligible_indices = [i for i in range(total)
                        if cfg["nodes_per_graph"][i % len(cfg["nodes_per_graph"])] in diagnostic_sizes]
    if quota > len(eligible_indices):
        raise ValueError("Insufficient contexts in the selected diagnostic graph sizes")
    diagnostic_indices = set(rng.sample(eligible_indices, quota))
    contexts = 0
    last_progress = 0
    while contexts < total:
        attempts += 1
        if attempts - last_progress > 2000:
            raise RuntimeError(f"Cannot fill quotas without duplicate questions after {contexts} contexts. Do not relax deduplication silently; revise sample quotas.")
        if attempts > total * 500:
            raise RuntimeError("Cannot satisfy unique-context/task quotas; reduce counts or inspect graph")
        size = cfg["nodes_per_graph"][contexts % len(cfg["nodes_per_graph"])]
        context = _context(graph, size, rng, rng.choice(anchors) if contexts in diagnostic_indices else None)
        if context is None or context["evidence_hash"] in seen:
            continue
        if cfg.get("require_distinct_orders") and traversal(context, "bfs", direction) == traversal(context, "dfs", direction):
            continue
        specs = _specs(context, rng, contexts, used)
        if specs is None:
            continue
        seen.add(context["evidence_hash"])
        context_adj = adjacency(context["nodes"], context["edges"])
        has_multi_parent = any(len(v) > 1 for v in context_adj.values())
        if contexts in diagnostic_indices and not has_multi_parent:
            raise ValueError("Anchored context did not retain both parent edges")
        order_distinct = traversal(context, "bfs", direction) != traversal(context, "dfs", direction)
        context_id = f"g{contexts:04d}"
        for number, spec in enumerate(specs):
            gold = answer(context, spec)
            item = {"id": f"{context_id}-q{number}", "context_id": context_id,
                    "split": split_labels[contexts], "graph": context, "spec": spec,
                    "semantic_key": semantic_key(context, spec),
                    "task_definition": TASKS[spec["type"]], "gold": gold,
                    "question": None, "question_origin": "awaiting_llm_generation",
                    "serialization_traversal": direction,
                    "strata": {"task": spec["type"], "node_count": size,
                               "answer_empty": None if isinstance(gold, bool) else len(gold) == 0,
                               "has_multi_parent": has_multi_parent,
                               "bfs_dfs_distinct": order_distinct,
                               "answer_size": int(gold) if isinstance(gold, bool) else len(gold),
                               "k": spec.get("k"),
                               "shortest_path": path_length(context, spec["source"], spec["target"]) if spec["type"] == "reachability" else None}}
            dataset.append(item)
            used.add(item["semantic_key"])
        contexts += 1
        last_progress = attempts
        if contexts % 25 == 0:
            print(f"Prepared {contexts}/{total} unique contexts ({len(dataset)} semantic questions)", flush=True)
    dev_nodes = {v for q in dataset if q["split"] == "dev" for v in q["graph"]["original_ids"].values()}
    test_nodes = {v for q in dataset if q["split"] == "test" for v in q["graph"]["original_ids"].values()}
    payload = {"graph_hash": graph["graph_hash"], "config_hash": digest(config),
               "metadata": {"scope": "evidence_subgraph_only",
                            "questions": len(dataset), "contexts": contexts,
                            "semantic_duplicates": len(dataset) - len({q["semantic_key"] for q in dataset}),
                            "excluded_prior_questions": len(excluded_keys),
                            "source_multi_parent_nodes": len(anchors),
                            "multi_parent_contexts": len({q["context_id"] for q in dataset if q["strata"]["has_multi_parent"]}),
                            "bfs_dfs_distinct_contexts": len({q["context_id"] for q in dataset if q["strata"]["bfs_dfs_distinct"]}),
                            "serialization_traversal": direction,
                            "exact_k_balance": "prefer_alternating_empty_nonempty_then_any_unique_source",
                            "shared_dev_test_nodes": len(dev_nodes & test_nodes),
                            "split_unit": "unique_evidence_graph_not_disjoint_entities",
                            "gold_source": "graph_algorithm", "generation_status": "not_started"},
               "questions": dataset}
    write(output, payload)
    return payload
