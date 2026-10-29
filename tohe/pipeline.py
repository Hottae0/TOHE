import copy
import json
import os
from pathlib import Path
from .io import append, digest, read, rows, write
from .llm import chat, parse_json
from .serialize import serialize, PROMPT_VERSION


def generate_questions(dataset, config, output, max_calls):
    if max_calls <= 0:
        raise ValueError("max_calls must be positive")
    output = Path(output)
    source_hash = digest(dataset)
    generation_hash = digest({"config": config["generation_llm"],
                              "endpoint": os.environ.get(config["generation_llm"]["endpoint_env"])})
    if output.exists():
        current = read(output)
        if current.get("source_hash") != source_hash or current.get("generation_config_hash") != generation_hash:
            raise ValueError("Existing QA output belongs to different inputs/model; choose a new path")
    else:
        current = copy.deepcopy(dataset)
        current.update(source_hash=source_hash, generation_config_hash=generation_hash)
    calls = 0
    for item in current["questions"]:
        if item["question_origin"] == "llm_generated":
            continue
        if calls >= max_calls:
            break
        instruction = {
            "instruction": "Write one clear English graph question for this exact task. Do not solve it. Do not change IDs, direction, quantifiers, or k. Return JSON with question and an exact copy of spec.",
            "spec": item["spec"], "definition": item["task_definition"]}
        instruction["response_example"] = {"question": "<one English question>", "spec": item["spec"]}
        messages = [{"role": "user", "content": json.dumps(instruction)}]
        failure_path = output.with_suffix(".generation-failures.jsonl")
        for attempt in range(1, 4):
            if calls >= max_calls:
                write(output, current)
                return current
            response = chat(config["generation_llm"], messages)
            calls += 1
            value, error = None, None
            try:
                value = parse_json(response["text"])
                if not isinstance(value, dict):
                    error = "Response must be a JSON object"
                elif digest(value.get("spec")) != digest(item["spec"]):
                    error = "spec must exactly match the supplied keys, values and JSON types"
                elif not isinstance(value.get("question"), str) or not value["question"].strip():
                    error = "question must be a nonempty string"
            except (ValueError, TypeError) as exc:
                error = f"Invalid JSON: {exc}"
            if error is None:
                break
            append(failure_path, {"question_id": item["id"], "attempt": attempt,
                                  "expected_spec": item["spec"],
                                  "received_spec": value.get("spec") if isinstance(value, dict) else None,
                                  "error": error, "response": response,
                                  "generation_config_hash": generation_hash})
            print(f"Question {item['id']} attempt {attempt}/3 rejected: {error}. "
                  f"Details: {failure_path}", flush=True)
            if attempt == 3:
                write(output, current)
                raise ValueError(f"Question {item['id']} failed validation after 3 attempts. "
                                 f"Expected spec: {json.dumps(item['spec'])}. "
                                 f"Full responses: {failure_path}")
            messages.extend([
                {"role": "assistant", "content": response["text"]},
                {"role": "user", "content": f"Validation failed: {error}. Rewrite the response. "
                 "Return only a JSON object with question and spec. Copy this spec exactly, "
                 "including every key and no additional spec fields: " + json.dumps(item["spec"])}])
        # Structural validation is not proof of natural-language equivalence.
        # The formal spec is included at evaluation; audit wording separately.
        item.update(question=value["question"].strip(), question_origin="llm_generated",
                    generation={"model": config["generation_llm"]["model"], **response,
                                "prompt_version": "exact-spec-repair-v2", "attempts": attempt},
                    semantic_review="pending")
        current["metadata"]["generation_status"] = "partial"
        write(output, current)
        print(f"Generated {item['id']} ({calls}/{max_calls} requests this run)", flush=True)
    if all(q["question_origin"] == "llm_generated" for q in current["questions"]):
        current["metadata"]["generation_status"] = "complete"
    write(output, current)
    return current


def score_prediction(prediction, gold, nodes=None):
    if isinstance(gold, bool):
        valid = isinstance(prediction, bool)
        return {"valid": valid, "correct": valid and prediction == gold, "set_f1": None,
                "error_type": None if valid else "answer_type"}
    valid = isinstance(prediction, list) and all(isinstance(v, str) for v in prediction)
    if not valid:
        return {"valid": False, "correct": False, "set_f1": 0.0, "error_type": "answer_type"}
    if nodes is not None and set(prediction) - set(nodes):
        return {"valid": False, "correct": False, "set_f1": 0.0, "error_type": "unknown_node_id"}
    actual, expected = set(prediction), set(gold)
    if not actual and not expected:
        f1 = 1.0
    else:
        f1 = 2 * len(actual & expected) / (len(actual) + len(expected))
    return {"valid": True, "correct": actual == expected, "set_f1": f1, "error_type": None}


def evaluate(dataset, config, run_dir, checkpoint_dir, split, max_calls, dry_run=False):
    import torch
    import hashlib
    from .evidence import SCOPE, evidence_graphs, checkpoint_path, check_checkpoint
    if config.get("embedding_scope") != SCOPE:
        raise ValueError("Evaluation requires the evidence-local protocol config")
    from .questions import CANONICAL_VERSION, canonical_question
    graphs = evidence_graphs(dataset, split)
    # Preflight before spending API calls. Load each evidence/method/seed once,
    # then reuse its codes across all questions sharing that context.
    cached = {}
    for condition in config["conditions"]:
        if condition.startswith("direct_"):
            continue
        for seed in config["training"]["seeds"]:
            for context_id, graph in graphs.items():
                path = checkpoint_path(checkpoint_dir, context_id, condition, seed)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                check_checkpoint(checkpoint, graph, config, condition, seed)
                codes = dict(zip(checkpoint["nodes"], checkpoint["codes"].tolist()))
                cached[(context_id, condition, seed)] = (codes, hashlib.sha256(path.read_bytes()).hexdigest())
    from .audited_questions import review_error, wording_error
    for item in dataset["questions"]:
        if item["split"] != split or dry_run:
            continue
        if item["question_origin"] == "canonical_template":
            if (dataset["metadata"].get("question_mode") != CANONICAL_VERSION
                    or item["question"] != canonical_question(item["spec"])):
                raise ValueError("Canonical question does not match its specification")
        elif item["question_origin"] == "llm_generated":
            if item.get("semantic_review") != "model_verified_v3":
                raise ValueError("LLM questions require audited-v3 semantic review; regenerate questions")
            try:
                error = wording_error(item["question"], item["spec"]) or review_error(
                    parse_json(item["semantic_audit"]["text"]), item["spec"])
                if item["semantic_audit"].get("question_hash") != digest(item["question"]):
                    error = "Question text changed after semantic review"
            except (ValueError, KeyError, TypeError):
                error = "Missing or invalid semantic audit"
            if error:
                raise ValueError(f"Question {item['id']} failed semantic validation: {error}")
        else:
            raise ValueError("Generate questions first; unsupported question origin")
    run_dir = Path(run_dir)
    manifest = {"dataset_hash": digest(dataset), "config_hash": digest(config), "split": split,
                "dry_run": dry_run, "conditions": config["conditions"],
                "prompt_version": PROMPT_VERSION,
                "endpoint_hash": digest(os.environ.get(config["llm"]["endpoint_env"])),
                "embedding_scope": SCOPE,
                "evidence_graph_hashes": {key: graph["graph_hash"] for key, graph in graphs.items()}}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Run directory contains incompatible inputs; use a new directory")
    write(manifest_path, manifest)
    records_path = run_dir / ("prompts.jsonl" if dry_run else "predictions.jsonl")
    previous = rows(records_path)
    completed = {row["key"]: row for row in previous}
    calls = 0
    for condition in config["conditions"]:
        seeds = [None] if condition.startswith("direct_") else config["training"]["seeds"]
        for seed in seeds:
            for item in dataset["questions"]:
                if item["split"] != split:
                    continue
                codes, ckpt_hash = (None, None) if seed is None else cached[(item["context_id"], condition, seed)]
                prompt = serialize(item, condition, codes)
                key = f"{condition}/{seed}/{item['id']}"
                prompt_hash = digest(prompt)
                if key in completed:
                    old = completed[key]
                    if old["prompt_hash"] != prompt_hash or old.get("checkpoint_hash") != ckpt_hash:
                        raise ValueError("Cached prediction has changed prompt/checkpoint")
                    continue
                if calls >= max_calls:
                    return {"new_records": calls, "stopped_at_limit": True}
                base = {"key": key, "condition": condition, "seed": seed, "question_id": item["id"],
                        "context_id": item["context_id"], "strata": item["strata"],
                        "evidence_graph_hash": graphs[item["context_id"]]["graph_hash"],
                        "prompt_hash": prompt_hash, "checkpoint_hash": ckpt_hash,
                        "prompt": prompt, "prompt_characters": len(prompt)}
                if dry_run:
                    append(records_path, {**base, "status": "prompt_only_no_llm_call"})
                else:
                    response = chat(config["llm"], [{"role": "user", "content": prompt}])
                    try:
                        predicted = parse_json(response["text"])["answer"]
                    except (ValueError, KeyError, TypeError):
                        predicted = None
                        parse_error = True
                    else:
                        parse_error = False
                    scores = score_prediction(predicted, item["gold"], item["graph"]["nodes"])
                    if parse_error:
                        scores["error_type"] = "invalid_json_or_missing_answer"
                    append(records_path, {**base, **response, "prediction": predicted,
                                          "gold": item["gold"], **scores})
                calls += 1
    return {"new_records": calls, "stopped_at_limit": False}
