"""Live integration checks: three correct tasks and three intentionally wrong meanings."""
import argparse
from tohe.io import read
from tohe.llm import chat, parse_json
from tohe.questions import canonical_question
from tohe.audited_questions import review_messages, review_response_format, review_format_error, review_context_error, review_error

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/small-qwen2.5-7b-cuda.json")
args = parser.parse_args()
cfg = dict(read(args.config)["generation_llm"], temperature=0.0,
           max_output_tokens=512, response_format=review_response_format())
reach = dict(type="reachability", source="N001", target="N019")
parents = dict(type="common_parents", left="N003", right="N016")
exact = dict(type="exact_k", source="N038", k=3)
cases = [
    ("reachability", canonical_question(reach), reach, True),
    ("common_parents_regression", "List all node IDs P for which both direct edges N003 --is_a--> P "
     "and N016 --is_a--> P are present. Use direct edges only, not more distant ancestors.", parents, True),
    ("exact_k", canonical_question(exact), exact, True),
    ("reject_count", "How many nodes are reachable from N038 by following exactly 3 directed is_a edges?", exact, False),
    ("reject_shortest_distance", "List all nodes whose shortest directed is_a path from N038 has length 3.", exact, False),
    ("reject_reversed_query", canonical_question(dict(type="reachability", source="N019", target="N001")), reach, False)]
failed = []
for name, question, spec, expected in cases:
    response = chat(cfg, review_messages(question))
    value = parse_json(response["text"])
    structure_error = review_format_error(value) or review_context_error(value, question)
    semantic_error = None if structure_error else review_error(value, spec)
    # Malformed or internally inconsistent review never counts as a successful rejection.
    passed = structure_error is None and ((semantic_error is None) == expected)
    print(name, "PASS" if passed else "FAIL", value, flush=True)
    if not passed:
        failed.append(name)
if failed:
    raise SystemExit("Reviewer integration failed: " + ", ".join(failed))
print("LIVE REVIEWER CHECKS: 6/6 PASS", flush=True)
