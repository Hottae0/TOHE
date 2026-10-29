"""Deterministic English questions for a controlled diagnostic, without an LLM."""
import copy
from pathlib import Path
from .io import digest, read, write

CANONICAL_VERSION = "canonical-v1"


def canonical_question(spec):
    if spec["type"] == "reachability":
        return (f"Does a directed path of at least one is_a edge exist from {spec['source']} "
                f"to {spec['target']}? Follow only the provided edge direction. Return a boolean.")
    if spec["type"] == "common_parents":
        return (f"List all nodes P for which both direct edges {spec['left']} --is_a--> P "
                f"and {spec['right']} --is_a--> P are present. Use direct edges only, "
                "not more distant ancestors. Return node IDs, not their count.")
    if spec["type"] == "exact_k":
        return (f"List all nodes reachable from {spec['source']} by following exactly "
                f"{spec['k']} directed is_a edges. Include a node if such a path exists, "
                "even if a shorter path also exists. Return only qualifying endpoints, "
                "not intermediate nodes or a count.")
    raise ValueError(f"Unknown task: {spec['type']}")


def canonical_questions(dataset, output):
    result = copy.deepcopy(dataset)
    result["source_hash"] = digest(dataset)
    result["metadata"].update(question_mode=CANONICAL_VERSION, generation_status="complete")
    for item in result["questions"]:
        item.update(question=canonical_question(item["spec"]),
                    question_origin="canonical_template",
                    generation={"method": "deterministic_template", "version": CANONICAL_VERSION},
                    semantic_review="canonical_by_construction")
    output = Path(output)
    if output.exists() and read(output) != result:
        raise ValueError("Canonical question file differs; use a new output path")
    write(output, result)
    return result
