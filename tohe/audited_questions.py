"""Generate wording, then reconstruct its semantics in a separate blind call."""
import copy
import json
import os
import re
from pathlib import Path
from .io import append, digest, read, write
from .llm import chat, parse_json
from .questions import canonical_question

VERSION = "audited-v3"


def review_response_format():
    """Generic shape only: no expected IDs, k, task or answer kind is forced."""
    branches = []
    for task, fields in (("reachability", ("source", "target")),
                         ("common_parents", ("left", "right")),
                         ("exact_k", ("source", "k"))):
        properties = {"type": {"type": "string", "enum": [task]}}
        properties.update({f: {"type": "integer" if f == "k" else "string"} for f in fields})
        branches.append({"type": "object", "properties": properties,
                         "required": list(properties), "additionalProperties": False})
    properties = {
        "interpreted_spec": {"anyOf": branches + [{"type": "null"}]},
        "answer_kind": {"type": "string", "enum": ["boolean", "node_set", "count", "unspecified"]},
        "edge_direction": {"type": "string", "enum": ["outgoing", "incoming", "unspecified"]},
        "path_semantics": {"type": "string", "enum": ["nonempty_path", "direct_neighbors",
                              "exact_length", "shortest_distance", "unspecified"]},
        "ambiguous": {"type": "boolean"}}
    return {"type": "json_schema", "json_schema": {"name": "graph_question_review",
            "schema": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}


def review_format_error(value):
    schema = review_response_format()["json_schema"]["schema"]
    if not isinstance(value, dict) or set(value) != set(schema["properties"]):
        return "Review must contain exactly the five required fields"
    spec = value["interpreted_spec"]
    if spec is not None:
        if not isinstance(spec, dict):
            return "interpreted_spec must be a JSON object, not explanatory text"
        branches = schema["properties"]["interpreted_spec"]["anyOf"][:-1]
        branch = next((b for b in branches if spec.get("type") in b["properties"]["type"]["enum"]), None)
        if branch is None or set(spec) != set(branch["required"]):
            return "interpreted_spec has invalid task keys"
        for key in spec:
            expected_type = int if key == "k" else str
            if type(spec[key]) is not expected_type:
                return f"interpreted_spec.{key} has an invalid JSON type"
    for key in ("answer_kind", "edge_direction", "path_semantics"):
        if value[key] not in schema["properties"][key]["enum"]:
            return f"Invalid enum value for {key}"
    if type(value["ambiguous"]) is not bool:
        return "ambiguous must be a JSON boolean"
    return None


def wording_error(question, spec):
    if not isinstance(question, str) or not question.strip():
        return "question must be nonempty English text"
    ids = {v for k, v in spec.items() if k in ("source", "target", "left", "right")}
    if set(re.findall(r"\bN\d+\b", question)) != ids:
        return "Mention all task node IDs exactly, including leading zeros; no other node IDs"
    if spec["type"] != "reachability" and re.search(r"how many|what is the number|count the", question, re.I):
        return "Ask for a list of node IDs, never the number of nodes"
    if spec["type"] == "exact_k" and re.search(r"edges away|distance", question, re.I):
        return "Describe following exactly k directed edges, not distance or edges away"
    return None


def review_messages(question, feedback=""):
    # Generic demonstrations use unrelated IDs, not the expected spec or gold.
    instruction = ("Interpret the question, do not solve it. A task asking for the intersection "
        "of TWO direct neighbor sets is common_parents. A yes/no path-existence task is "
        "reachability. A fixed-length endpoint list is exact_k. P denotes a bound output "
        "variable, NEVER a literal node ID. Return interpreted_spec as an object, answer_kind, "
        "edge_direction, path_semantics, ambiguous. Mark ambiguous wording instead of "
        "silently assuming missing semantics. For count questions use answer_kind=count; "
        "for shortest-distance questions use path_semantics=shortest_distance. " + feedback)
    examples = [
        ("Does a directed path of at least one is_a edge exist from N101 to N102? Return a boolean.",
         dict(type="reachability", source="N101", target="N102"), "boolean", "nonempty_path"),
        ("List all node IDs P for which both direct edges N103 --is_a--> P and "
         "N104 --is_a--> P are present. Use direct edges only.",
         dict(type="common_parents", left="N103", right="N104"), "node_set", "direct_neighbors"),
        ("List all nodes reachable from N105 by following exactly 2 directed is_a edges, "
         "even if a shorter path exists.",
         dict(type="exact_k", source="N105", k=2), "node_set", "exact_length")]
    messages = [{"role": "system", "content": instruction}]
    for wording, spec, kind, semantics in examples:
        messages.extend([{"role": "user", "content": wording}, {"role": "assistant", "content": json.dumps(
            dict(interpreted_spec=spec, answer_kind=kind, edge_direction="outgoing",
                 path_semantics=semantics, ambiguous=False))}])
    messages.append({"role": "user", "content": question})
    return messages


def review_context_error(value, question):
    """Check the reviewer against the question itself, without a target spec."""
    spec = value.get("interpreted_spec")
    if not isinstance(spec, dict):
        return None
    ids = set(re.findall(r"\bN\d+\b", question))
    for field in ("source", "target", "left", "right"):
        if field in spec and spec[field] not in ids:
            return "Reviewer invented a literal node ID; output variables such as P are not node IDs"
    allowed = {"reachability": {"nonempty_path", "unspecified"},
               "common_parents": {"direct_neighbors", "unspecified"},
               "exact_k": {"exact_length", "shortest_distance", "unspecified"}}
    if value["path_semantics"] not in allowed[spec["type"]]:
        return "Reviewer task type contradicts its path semantics; interpret the entire question jointly"
    return None


def review_error(value, spec):
    if not isinstance(value, dict):
        return "Reviewer did not return an object"
    expected_kind = "boolean" if spec["type"] == "reachability" else "node_set"
    semantics = {"reachability": "nonempty_path", "common_parents": "direct_neighbors",
                 "exact_k": "exact_length"}[spec["type"]]
    expected = dict(interpreted_spec=spec, answer_kind=expected_kind,
                    edge_direction="outgoing", path_semantics=semantics, ambiguous=False)
    for key, target in expected.items():
        if digest(value.get(key)) != digest(target):
            return f"Semantic review mismatch in {key}: {value.get(key)!r}; required {target!r}"
    return None


def generate_audited_questions(dataset, config, output, max_calls):
    if max_calls <= 0:
        raise ValueError("Set a positive request limit (includes generation and review)")
    output = Path(output)
    llm = config["generation_llm"]
    fingerprint = digest({"version": VERSION, "config": llm,
                          "endpoint": os.environ.get(llm["endpoint_env"])})
    current = read(output) if output.exists() else copy.deepcopy(dataset)
    if output.exists() and (current.get("source_hash") != digest(dataset)
                            or current.get("generation_config_hash") != fingerprint):
        raise ValueError("Question cache differs; use a fresh audited output file")
    current.update(source_hash=digest(dataset), generation_config_hash=fingerprint)
    current["metadata"].update(question_mode=VERSION, generation_status="partial")
    failures = output.with_suffix(".generation-failures.jsonl")
    calls = 0
    for item in current["questions"]:
        if item.get("semantic_review") == "model_verified_v3":
            continue
        feedback = ""
        for attempt in range(1, 4):
            if calls >= max_calls:
                write(output, current)
                return current
            candidate = item.get("generation_candidate")
            error, review, format_failure = None, None, False
            if candidate is None:
                instruction = {"instruction": "Write one unambiguous English graph question. "
                    "Do NOT solve it. Preserve all IDs and the task semantics. "
                    "For set tasks explicitly ask to list node IDs, NEVER how many. "
                    "Specify outgoing edge direction. For exact_k describe paths of exactly "
                    "k edges, not shortest distance. Mention that a shorter path does not "
                    "exclude a qualifying endpoint. Return JSON with question and an exact copy of spec.",
                    "spec": item["spec"], "semantic_reference": canonical_question(item["spec"]),
                    "previous_failure": feedback}
                response = chat(llm, [{"role": "user", "content": json.dumps(instruction)}])
                calls += 1
                candidate = {"response": response}
                try:
                    value = parse_json(response["text"])
                    if not isinstance(value, dict) or digest(value.get("spec")) != digest(item["spec"]):
                        error = "Return an exact copy of the spec"
                    else:
                        error = wording_error(value.get("question"), item["spec"])
                        candidate["question"] = value.get("question")
                except (ValueError, TypeError) as exc:
                    error = f"Invalid generator JSON: {exc}"
            if error is None:
                item["generation_candidate"] = candidate
                write(output, current)
                if calls >= max_calls:
                    return current  # Persist the candidate; review it on the next run.
                reviewer_cfg = dict(llm, temperature=0.0, max_output_tokens=512,
                                    response_format=review_response_format())
                review = chat(reviewer_cfg, review_messages(candidate["question"], candidate.get("review_feedback", "")))
                calls += 1
                try:
                    interpreted = parse_json(review["text"])
                    error = review_format_error(interpreted) or review_context_error(interpreted, candidate["question"])
                    format_failure = error is not None
                    if error is None:
                        error = review_error(interpreted, item["spec"])
                except (ValueError, TypeError) as exc:
                    error = f"Invalid reviewer JSON: {exc}"
                    format_failure = True
            if error is None:
                item.pop("generation_candidate", None)
                item.update(question=candidate["question"].strip(), question_origin="llm_generated",
                            semantic_review="model_verified_v3",
                            generation={"model": llm["model"], **candidate["response"], "prompt_version": VERSION},
                            semantic_audit={"model": llm["model"], **review, "version": VERSION,
                                            "review_protocol": "balanced-v2",
                                            "question_hash": digest(candidate["question"].strip())})
                write(output, current)
                print(f"Generated and audited {item['id']} ({calls}/{max_calls} requests)", flush=True)
                break
            if format_failure:
                candidate["review_feedback"] = "Correct your previous review format error: " + error
                item["generation_candidate"] = candidate
            else:
                item.pop("generation_candidate", None)
            append(failures, {"question_id": item["id"], "attempt": attempt, "error": error,
                              "error_stage": "review_format" if format_failure else "wording_or_semantics",
                              "expected_spec": item["spec"], "candidate": candidate, "review": review})
            feedback = error
            write(output, current)
            print(f"Rejected {item['id']}: {error}", flush=True)
            if attempt == 3:
                raise ValueError(f"Question {item['id']} failed 3 attempts; inspect {failures}")
    current["metadata"]["generation_status"] = "complete"
    write(output, current)
    return current
