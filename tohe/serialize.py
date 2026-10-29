import json
from .graph import adjacency, traversal


COMMON = """You are given a directed graph and optional embedding metadata.

Rules:
- A --is_a--> B means that A is a direct subtype of B.
- A node may have multiple direct supertypes.
- Determine the answer from the explicitly provided graph.
- The formal task specification defines the exact question semantics.
- Return only a JSON object with one key, \"answer\".
- For reachability, answer must be a JSON boolean.
- For set-valued tasks, answer must be a sorted array of unique node IDs.
"""

PROMPT_VERSION = "prompt-v2"


def answer_rules(spec):
    if spec["type"] == "reachability":
        return 'Task output: {"answer": true} or {"answer": false}. Follow only outgoing is_a edges.'
    rules = ('Task output: {"answer": ["NODE_ID", ...]}. Return {"answer": []} if no node qualifies. '
             'Return a sorted array of unique IDs from the Nodes list, not a count or boolean. '
             'Copy IDs exactly, including leading zeros. Do not invent or abbreviate IDs.')
    if spec["type"] == "exact_k":
        rules += (f' Follow exactly {spec["k"]} outgoing is_a edges from {spec["source"]}. '
                  'This is path length, not shortest-path distance. Return only endpoints after '
                  'exactly that many edges; an intermediate node qualifies only if it is also '
                  'an endpoint of a path of the required length.')
    else:
        rules += ' Return the intersection of the two direct outgoing neighbor sets only.'
    return rules

METADATA = """
Embedding metadata rules:
- Embeddings are trained only on this evidence graph. Group IDs are local to it.
- A smaller radial_bin indicates a smaller distance from the embedding origin.
  It does not necessarily indicate graph depth.
- Nodes in the same direction_group have similar embedding directions.
- Group identifiers are categorical. Numerical differences do not represent distances.
- Metadata is approximate and does not establish an edge or a directed path.
- Explicit graph edges are authoritative.
"""


def serialize(item, condition, codes=None):
    graph = item["graph"]
    direct = condition.startswith("direct_")
    order = condition.removeprefix("direct_") if direct else "id"
    if order not in ("id", "bfs", "dfs"):
        raise ValueError(order)
    if not direct and codes is None:
        raise ValueError("Embedding condition requires codes")
    ordered = traversal(graph, order)
    adj = adjacency(graph["nodes"], graph["edges"])
    lines = [COMMON + ("" if direct else METADATA), "Nodes:"]
    for node in ordered:
        if direct:
            lines.append(node)
        else:
            radial, sector = codes[graph["original_ids"][node]]
            lines.append(f"{node}: radial_bin={radial}, direction_group=g{sector}")
    lines.append("\nDirect edges:")
    for node in ordered:
        for parent in sorted(adj[node]):
            lines.append(f"{node} --is_a--> {parent}")
    lines += ["\nQuestion:", item["question"] or "[LLM question generation pending]",
              "\nFormal task specification:", json.dumps(item["spec"], sort_keys=True),
              item["task_definition"], "\nTask-specific answer rules:", answer_rules(item["spec"])]
    return "\n".join(lines)
