"""A separately versioned hierarchy diagnostic on frozen evidence graphs."""
import copy
import itertools
import json
import random
import re
from .graph import adjacency, reachable, path_length, traversal
from .io import digest

VERSION = "hierarchy-qa-v1"
TASKS = ("ancestor_choice", "taxonomic_relation", "lowest_common_ancestors")
LABELS = ("supertype", "subtype", "unrelated")
DEFINITIONS = {
    "ancestor_choice": "Select candidate nodes reachable from source by one or more outgoing is_a edges. Exactly one of the four candidates qualifies. Return its ID in a one-element JSON array.",
    "taxonomic_relation": "Classify LEFT relative to RIGHT: subtype if LEFT reaches RIGHT by a nonempty outgoing is_a path; supertype if RIGHT reaches LEFT; unrelated if neither reaches the other. Return exactly one lowercase label string: supertype, subtype, unrelated.",
    "lowest_common_ancestors": "Use reflexive ancestors: a node is its own ancestor and also has all nodes reachable by outgoing is_a paths as ancestors. Return every common ancestor of LEFT and RIGHT that is not a strict ancestor of another common ancestor. Multiple answers are allowed in a DAG. Return a sorted array of IDs, or [] when there is none.",
}


def index_graph(graph):
    adj = adjacency(graph["nodes"], graph["edges"])
    ancestors = {n: reachable(adj, n) for n in graph["nodes"]}
    if any(n in ancestors[n] for n in ancestors):
        raise ValueError("Hierarchy QA requires a DAG")
    return adj, ancestors


def gold_answer(graph, spec):
    _, anc = index_graph(graph)
    task = spec["type"]
    if task == "ancestor_choice":
        return sorted(set(spec["candidates"]) & anc[spec["source"]])
    left, right = spec["left"], spec["right"]
    if task == "taxonomic_relation":
        if right in anc[left]:
            return "subtype"
        if left in anc[right]:
            return "supertype"
        return "unrelated"
    if task == "lowest_common_ancestors":
        common = (anc[left] | {left}) & (anc[right] | {right})
        return sorted(a for a in common if not any(a in anc[b] for b in common - {a}))
    raise ValueError(task)


def canonical_question(spec):
    task = spec["type"]
    if task == "ancestor_choice":
        return (f"Which one of the candidate nodes {', '.join(spec['candidates'])} is a direct or "
                f"indirect supertype of {spec['source']}? Follow one or more outgoing is_a edges "
                "from the source. Return the qualifying candidate ID in a one-element array.")
    left, right = spec["left"], spec["right"]
    if task == "taxonomic_relation":
        return (f"What is the taxonomic relationship of {left} to {right}: supertype, subtype, or "
                "unrelated? A subtype reaches its supertype along one or more outgoing is_a edges. "
                "Use unrelated when neither node reaches the other. Return a label string.")
    return (f"List all lowest common ancestors of {left} and {right}. Include each queried node "
            "in its own ancestor set. Follow outgoing is_a edges toward other ancestors. Keep "
            "only common ancestors that are not strict ancestors of another common ancestor. "
            "Allow multiple answers in a DAG; return a sorted ID array, or [] if none exist.")


def make_specs(source):
    """Two questions per task per original-dev graph; never select by model accuracy."""
    contexts = {}
    previous_pairs = {}
    for q in source["questions"]:
        if q["split"] != "dev":
            continue
        contexts.setdefault(q["context_id"], q)
        s = q["spec"]
        fields = ("source", "target") if s["type"] == "reachability" else ("left", "right")
        if all(f in s for f in fields):
            previous_pairs.setdefault(q["context_id"], set()).add(frozenset(s[f] for f in fields))
    output = {"metadata": {"protocol": VERSION, "selection": "All original-dev contexts from the frozen interim evaluation; no score-based selection", "question_mode": VERSION, "generation_status": "partial", "gold_source": "graph_algorithm", "scope": "evidence_subgraph_only", "source_hash": digest(source)}, "questions": []}
    relation_index = 0
    for cid, template in sorted(contexts.items()):
        graph = template["graph"]
        nodes = sorted(graph["nodes"])
        adj, anc = index_graph(graph)
        rng = random.Random(int(digest({"version": VERSION, "context": cid})[:16], 16))
        forbidden = previous_pairs.get(cid, set())
        specs, used_sources = [], set()
        for j in range(2):
            sources = [n for n in nodes if anc[n] and len(set(nodes) - anc[n] - {n}) >= 3 and n not in used_sources]
            rng.shuffle(sources)
            if not sources:
                raise ValueError(f"Insufficient four-way ancestor choices in {cid}")
            source_node = sources[0]
            used_sources.add(source_node)
            indirect = sorted(anc[source_node] - adj[source_node])
            positive = rng.choice(indirect or sorted(anc[source_node]))
            negatives = set(nodes) - anc[source_node] - {source_node}
            reverse = sorted(n for n in negatives if source_node in anc[n])
            unrelated = sorted(n for n in negatives if source_node not in anc[n])
            rng.shuffle(reverse); rng.shuffle(unrelated)
            distractors = reverse[:1] + unrelated[:2]
            remaining = sorted(negatives - set(distractors)); rng.shuffle(remaining)
            distractors = (distractors + remaining)[:3]
            rng.shuffle(distractors)
            candidates = list(distractors)
            candidates.insert((len(output["questions"]) // 6 * 2 + j) % 4, positive)
            specs.append({"type": TASKS[0], "source": source_node, "candidates": candidates})
        pairs = list(itertools.permutations(nodes, 2)); rng.shuffle(pairs)
        used_pairs = set()
        for j in range(2):
            label = LABELS[relation_index % 3]; relation_index += 1
            pair = next(((a, b) for a, b in pairs if frozenset((a, b)) not in forbidden | used_pairs
                         and ("subtype" if b in anc[a] else "supertype" if a in anc[b] else "unrelated") == label), None)
            if pair is None:
                raise ValueError(f"No unused {label} pair in {cid}")
            used_pairs.add(frozenset(pair))
            specs.append({"type": TASKS[1], "left": pair[0], "right": pair[1]})
        lca_pairs = list(itertools.combinations(nodes, 2)); rng.shuffle(lca_pairs)
        # Prefer incomparable queries with nonempty shared ancestors, without imposing a tree.
        lca_pairs.sort(key=lambda p: (p[1] in anc[p[0]] or p[0] in anc[p[1]],
                       not bool((anc[p[0]] | {p[0]}) & (anc[p[1]] | {p[1]}))))
        selected_pairs = []
        for a, b in lca_pairs:
            if frozenset((a, b)) in forbidden | used_pairs:
                continue
            selected_pairs.append((a, b)); used_pairs.add(frozenset((a, b)))
            if len(selected_pairs) == 2:
                break
        if len(selected_pairs) != 2:
            raise ValueError(f"Insufficient distinct LCA pairs in {cid}")
        specs.extend({"type": TASKS[2], "left": a, "right": b} for a, b in selected_pairs)
        for j, spec in enumerate(specs):
            gold = gold_answer(graph, spec)
            if spec["type"] == TASKS[0] and len(gold) != 1:
                raise AssertionError("Candidate task must have exactly one valid option")
            item = {k: copy.deepcopy(template[k]) for k in ("graph", "context_id", "split", "serialization_traversal") if k in template}
            item.update(id=f"h-{cid}-q{j}", spec=spec, gold=gold, question=None,
                        semantic_key=digest({"graph": graph, "spec": spec}),
                        task_definition=DEFINITIONS[spec["type"]], question_origin="pending")
            item["strata"] = {"task": spec["type"], "node_count": len(nodes),
                "answer_size": len(gold) if isinstance(gold, list) else None,
                "answer_empty": not gold if isinstance(gold, list) else None,
                "relation_label": gold if isinstance(gold, str) else None,
                "has_multi_parent": any(len(x) > 1 for x in adj.values())}
            if spec["type"] == TASKS[0]:
                item["strata"]["ancestor_hops"] = path_length(graph, spec["source"], gold[0])
            output["questions"].append(item)
    from collections import Counter
    n = len(contexts)
    output["metadata"].update(contexts=n, questions=len(output["questions"]),
        task_counts=dict(Counter(q["spec"]["type"] for q in output["questions"])),
        relation_label_counts=dict(Counter(q["gold"] for q in output["questions"] if isinstance(q["gold"], str))),
        semantic_duplicates=len(output["questions"]) - len({q["semantic_key"] for q in output["questions"]}))
    assert len(output["questions"]) == n * 6 and output["metadata"]["semantic_duplicates"] == 0
    return output


def wording_error(text, spec):
    if not isinstance(text, str) or not text.strip():
        return "A nonempty English question is required"
    expected = set(spec.get("candidates", [])) | {spec[k] for k in ("source", "left", "right") if k in spec}
    if set(re.findall(r"\bN\d+\b", text)) != expected:
        return "Mention every supplied node ID exactly, with no additional node IDs"
    if spec["type"] != "taxonomic_relation" and re.search(r"how many|count the|number of", text, re.I):
        return "Ask for node IDs, not a count"
    if spec["type"] == "lowest_common_ancestors":
        if not re.search(r"own ancestor|reflexive|zero[- ]length|zero or more", text, re.I):
            return "Explicitly state: Treat each queried node as its own ancestor."
        if not re.search(r"lowest|minimal|not (?:a )?strict ancestor", text, re.I):
            return "Request minimal common ancestors, excluding strict ancestors of another common ancestor"
    return None


def review_schema():
    branches = []
    for task, fields in ((TASKS[0], ("source", "candidates")), (TASKS[1], ("left", "right")), (TASKS[2], ("left", "right"))):
        props = {"type": {"type": "string", "enum": [task]}}
        props.update({f: ({"type": "array", "items": {"type": "string"}} if f == "candidates" else {"type": "string"}) for f in fields})
        branches.append({"type": "object", "properties": props, "required": list(props), "additionalProperties": False})
    props = {"interpreted_spec": {"anyOf": branches + [{"type": "null"}]},
        "answer_kind": {"type": "string", "enum": ["node_set", "relation_label", "count", "unspecified"]},
        "edge_direction": {"type": "string", "enum": ["outgoing_to_supertypes", "incoming", "unspecified"]},
        "semantics": {"type": "string", "enum": ["strict_candidate_ancestor", "left_relative_to_right", "reflexive_minimal_common_ancestors", "unspecified"]},
        "ambiguous": {"type": "boolean"}}
    return {"type": "json_schema", "json_schema": {"name": "hierarchy_question_review", "schema": {
        "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}


def review_error(review, spec):
    semantics = dict(zip(TASKS, ("strict_candidate_ancestor", "left_relative_to_right", "reflexive_minimal_common_ancestors")))
    expected = dict(interpreted_spec=spec, answer_kind="relation_label" if spec["type"] == TASKS[1] else "node_set",
                    edge_direction="outgoing_to_supertypes", semantics=semantics[spec["type"]], ambiguous=False)
    if not isinstance(review, dict) or set(review) != set(expected):
        return "Return exactly the five generic review fields"
    actual = copy.deepcopy(review)
    if isinstance(actual.get("interpreted_spec"), dict) and spec["type"] == TASKS[0]:
        candidates = actual["interpreted_spec"].get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(set(candidates)):
            return "Candidates must be distinct node IDs"
        actual["interpreted_spec"]["candidates"] = sorted(candidates)
        expected["interpreted_spec"] = dict(spec, candidates=sorted(spec["candidates"]))
    for key, value in expected.items():
        if digest(actual.get(key)) != digest(value):
            return f"Semantic review mismatch: {key}"
    return None


def review_messages(question, feedback=""):
    # Demonstrations are unrelated to the target. The reviewer never sees target gold/spec.
    examples = [
        {"type": TASKS[0], "source": "N101", "candidates": ["N102", "N103", "N104", "N105"]},
        {"type": TASKS[1], "left": "N106", "right": "N107"},
        {"type": TASKS[2], "left": "N108", "right": "N109"}]
    semantics = ("strict_candidate_ancestor", "left_relative_to_right", "reflexive_minimal_common_ancestors")
    messages = [{"role": "system", "content": "Interpret the question; never solve it. Extract the task and exact node IDs. Distinguish the relation of LEFT to RIGHT from its reverse. Lowest common ancestors must use reflexive ancestor sets and allow multiple minimal elements in a DAG. Mark missing semantics ambiguous. Return the five review fields. " + feedback}]
    for spec, sem in zip(examples, semantics):
        messages.extend([{"role": "user", "content": canonical_question(spec)}, {"role": "assistant", "content": json.dumps(dict(
            interpreted_spec=spec, answer_kind="relation_label" if spec["type"] == TASKS[1] else "node_set",
            edge_direction="outgoing_to_supertypes", semantics=sem, ambiguous=False))}])
    return messages + [{"role": "user", "content": question}]


def serialize(item, condition, values=None):
    from .serialize import METADATA, RAW_METADATA
    graph, spec = item["graph"], item["spec"]
    direct, raw = condition.startswith("direct_"), condition.startswith("raw_")
    order = condition.removeprefix("direct_") if direct else "id"
    if order not in ("id", "bfs", "dfs") or (not direct and values is None):
        raise ValueError("Invalid condition or missing embedding metadata")
    ordered = traversal(graph, order, item.get("serialization_traversal", "outgoing"))
    adj = adjacency(graph["nodes"], graph["edges"])
    common = ("You are given a directed graph and optional embedding metadata.\n"
        "A --is_a--> B means A is a direct subtype of B. Multiple direct supertypes are allowed.\n"
        "Answer using only the supplied graph. The formal task specification defines the semantics.\n"
        'Return only a JSON object with one key, "answer".\n'
        "For ancestor_choice and lowest_common_ancestors, return a sorted array of unique valid node IDs.\n"
        "For taxonomic_relation, return exactly one lowercase string: supertype, subtype, unrelated.\n")
    metadata = "" if direct else RAW_METADATA if raw else METADATA
    if raw:
        metadata += "Coordinate space: Euclidean.\n" if condition == "raw_euclidean" else "Coordinate space: unit Poincare ball, curvature -1.\n"
    lines = [common + metadata, "Nodes:"]
    for node in ordered:
        if direct:
            lines.append(node)
        elif raw:
            lines.append(f"{node}: coordinates=" + json.dumps(values[graph["original_ids"][node]], allow_nan=False))
        else:
            radial, sector = values[graph["original_ids"][node]]
            lines.append(f"{node}: radial_bin={radial}, direction_group=g{sector}")
    lines.append("\nDirect edges:")
    lines.extend(f"{node} --is_a--> {parent}" for node in ordered for parent in sorted(adj[node]))
    lines.extend(["\nQuestion:", item["question"] or canonical_question(spec), "\nFormal task specification:",
                  json.dumps(spec, sort_keys=True), "\nTask-specific answer rules:", DEFINITIONS[spec["type"]]])
    return "\n".join(lines)


def score(prediction, item):
    from .pipeline import score_prediction
    spec = item["spec"]
    if spec["type"] == TASKS[1]:
        valid = type(prediction) is str and prediction in LABELS
        return dict(valid=valid, correct=valid and prediction == item["gold"], set_f1=None,
                    error_type=None if valid else "answer_type_or_label")
    result = score_prediction(prediction, item["gold"], item["graph"]["nodes"])
    if result["valid"] and (len(prediction) != len(set(prediction)) or prediction != sorted(prediction)):
        result.update(valid=False, correct=False, error_type="unsorted_or_duplicate_ids")
    if spec["type"] == TASKS[0] and result["valid"]:
        if len(prediction) != 1 or not set(prediction) <= set(spec["candidates"]):
            result.update(valid=False, correct=False, error_type="candidate_constraint")
    return result
