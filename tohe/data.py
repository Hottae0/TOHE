"""WordNet preparation; direct edges and training closure are distinct artifacts."""
import csv
import re
from pathlib import Path
from .graph import closure
from .io import digest, write


def prepare(output, filter_path=None, reference_csv=None, download=False, nltk_data=None):
    import nltk
    if nltk_data:
        nltk.data.path.insert(0, str(nltk_data))
    if download:
        if not nltk.download("wordnet", download_dir=nltk_data, quiet=True):
            raise RuntimeError("WordNet download failed")
    from nltk.corpus import wordnet as wn
    root = wn.synset("mammal.n.01")
    selected, queue = {root.name(): root}, [root]
    while queue:
        synset = queue.pop()
        for child in synset.hyponyms() + synset.instance_hyponyms():
            if child.name() not in selected:
                selected[child.name()] = child
                queue.append(child)
    edges = sorted({(name, p.name()) for name, synset in selected.items()
                    for p in synset.hypernyms() + synset.instance_hypernyms()
                    if p.name() in selected})
    nodes = sorted(selected)
    complete = closure(nodes, edges)
    if not filter_path:
        raise ValueError("Supply the upstream mammals_filter.txt; unfiltered data is not the agreed protocol")
    patterns = [line.strip() for line in Path(filter_path).read_text().splitlines() if line.strip()]
    pattern = re.compile("(" + "|".join(patterns) + ")") if patterns else None
    kept = [pair for pair in complete if not pattern or not pattern.match(" ".join(pair))]
    retained = {v for pair in kept for v in pair}
    nodes = sorted(retained)
    edges = [pair for pair in edges if set(pair) <= retained]
    # The official filter applies to TRAINING PAIRS, not direct-edge evidence.
    # Preserve original direct edges; do not invent shortcut/direct relations.
    direct_closure = set(closure(nodes, edges))
    matched = None
    if reference_csv:
        with open(reference_csv, newline="", encoding="utf-8") as stream:
            reference = {(r["id1"], r["id2"]) for r in csv.DictReader(stream)}
        matched = set(kept) == reference
        if not matched:
            raise ValueError(f"Reference mismatch: only local={len(set(kept)-reference)}, only reference={len(reference-set(kept))}")
    graph = {"nodes": nodes, "edges": edges, "closure": kept,
             "metadata": {"source": "NLTK WordNet", "wordnet_version": wn.get_version(),
                          "root": root.name(), "filter_sha256": digest(patterns),
                          "reference_verified": matched, "scope": "global_mammals_reconstruction",
                          "training_pairs_without_retained_direct_path": len(set(kept) - direct_closure),
                          "direct_reachability_pairs_excluded_by_training_filter": len(direct_closure - set(kept))}}
    graph["graph_hash"] = digest(graph)
    write(output, graph)
    return graph


def synthetic(output):
    nodes = [f"s{i}" for i in range(12)]
    edges = [(f"s{i}", f"s{(i-1)//2}") for i in range(1, 12)] + [("s7", "s4"), ("s9", "s3")]
    graph = {"nodes": nodes, "edges": edges, "closure": closure(nodes, edges),
             "metadata": {"source": "synthetic", "scope": "smoke_only"}}
    graph["graph_hash"] = digest(graph)
    write(output, graph)
    return graph
