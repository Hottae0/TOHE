import copy
import os
from pathlib import Path
from .io import append, digest, read, rows, write
from .llm import chat, parse_json
from .serialize import serialize


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
        response = chat(config["generation_llm"], [{"role": "user", "content": __import__("json").dumps(instruction)}])
        calls += 1
        value = parse_json(response["text"])
        if value.get("spec") != item["spec"] or not isinstance(value.get("question"), str) or not value["question"].strip():
            raise ValueError(f"Invalid LLM question for {item['id']}; raw output: {response['text'][:200]}")
        # Structural validation is not proof of natural-language equivalence.
        # The formal spec is included at evaluation; audit wording separately.
        item.update(question=value["question"].strip(), question_origin="llm_generated",
                    generation={"model": config["generation_llm"]["model"], **response},
                    semantic_review="pending")
        current["metadata"]["generation_status"] = "partial"
        write(output, current)
    if all(q["question_origin"] == "llm_generated" for q in current["questions"]):
        current["metadata"]["generation_status"] = "complete"
    write(output, current)
    return current


def score_prediction(prediction, gold):
    if isinstance(gold, bool):
        valid = isinstance(prediction, bool)
        return {"valid": valid, "correct": valid and prediction == gold, "set_f1": None}
    valid = isinstance(prediction, list) and all(isinstance(v, str) for v in prediction)
    if not valid:
        return {"valid": False, "correct": False, "set_f1": 0.0}
    actual, expected = set(prediction), set(gold)
    if not actual and not expected:
        f1 = 1.0
    else:
        f1 = 2 * len(actual & expected) / (len(actual) + len(expected))
    return {"valid": True, "correct": actual == expected, "set_f1": f1}


def evaluate(dataset, config, run_dir, checkpoint_dir, split, max_calls, dry_run=False):
    import torch
    run_dir = Path(run_dir)
    manifest = {"dataset_hash": digest(dataset), "config_hash": digest(config), "split": split,
                "dry_run": dry_run, "conditions": config["conditions"],
                "endpoint_hash": digest(os.environ.get(config["llm"]["endpoint_env"])),
                "embedding_scope": "global_wordnet_graph_not_only_local_evidence"}
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
            codes, ckpt_hash = None, None
            if seed is not None:
                path = Path(checkpoint_dir) / f"{condition}-{seed}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if checkpoint["graph_hash"] != dataset["graph_hash"]:
                    raise ValueError("Checkpoint and QA source graph differ")
                if checkpoint["config"]["serializer"] != config["serializer"] or checkpoint["config"]["training"] != config["training"]:
                    raise ValueError("Checkpoint training/serializer config mismatch")
                if checkpoint["variant"] != condition or checkpoint["seed"] != seed:
                    raise ValueError("Mislabeled checkpoint")
                import hashlib
                ckpt_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                codes = {n: c for n, c in zip(checkpoint["nodes"], checkpoint["codes"].tolist())}
            for item in dataset["questions"]:
                if item["split"] != split:
                    continue
                if not dry_run and item["question_origin"] != "llm_generated":
                    raise ValueError("Generate real LLM questions first; template/mock questions are not research QA")
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
                        "prompt_hash": prompt_hash, "checkpoint_hash": ckpt_hash,
                        "prompt": prompt, "prompt_characters": len(prompt)}
                if dry_run:
                    append(records_path, {**base, "status": "prompt_only_no_llm_call"})
                else:
                    response = chat(config["llm"], [{"role": "user", "content": prompt}])
                    try:
                        predicted = parse_json(response["text"])["answer"]
                    except (ValueError, KeyError):
                        predicted = None
                    append(records_path, {**base, **response, "prediction": predicted,
                                          "gold": item["gold"], **score_prediction(predicted, item["gold"])})
                calls += 1
    return {"new_records": calls, "stopped_at_limit": False}
