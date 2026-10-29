# Audited question regeneration

The previous wording-only generator accepted “How many nodes ...?” for a node-set task. Exact spec echo did not validate the meaning of the English question. The new default retains LLM generation and adds validation rather than changing the graph tasks or training.

## Changes

1. The generation prompt explicitly requires outgoing direction, node sets rather than counts, and exact path length rather than shortest distance. A deterministic English reference is supplied to the writer; it contains no gold or graph edges.
2. Local rules reject missing/altered task IDs, count questions for set tasks, and ambiguous distance wording for exact-k.
3. A separate Qwen call receives **only the question and the interpretation rubric**, without the expected spec, graph, gold or generation conversation. It reconstructs the task, answer kind, direction and path semantics and flags ambiguity. Code compares this interpretation to the frozen spec.
4. Failed questions are regenerated with feedback, up to three attempts per invocation. No invalid question is accepted as a fallback. Full failures and successful audit responses are saved. A request limit includes both generation and review calls; an unaudited candidate is persisted and reviewed when resumed.
5. All seven conditions receive the same updated task-specific output rules. Unknown node IDs are recorded separately from answer-type and parse errors. IDs are not automatically corrected. Invalid-ID set answers receive F1=0 under the new scoring protocol; old and new invalid/F1 rates are not directly comparable.

The reviewer is a **separate call to the same model**, not an independent model or proof of semantic correctness. Human sample review remains necessary. The question is never selected based on whether the QA evaluator answers it correctly. Gold and embeddings are unchanged. A canonical-template diagnostic remains available with --question-mode canonical and is honestly labeled as deterministic, not LLM-generated.

### Structured reviewer response repair

The reviewer sometimes returned a prose string in interpreted_spec instead of an object. Reviewer calls now send a generic response_format=json_schema using [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/). The schema permits all three tasks, count answers, reversed/unspecified direction, shortest-distance semantics, ambiguity and unknown tasks. It does not contain the expected task, IDs, k or gold and cannot force a passing verdict. A local shape validator distinguishes reviewer-format errors from actual semantic mismatches. Format failures preserve the candidate and retry its review with feedback; semantic failures regenerate the question. These requests still share the bounded request budget.

Existing audited-v3 caches and accepted questions can resume; new successful reviews record review_protocol=structured-v1. The generic review shape changed without changing the writer or graph semantics. No checkpoint regeneration or file deletion is needed. If the server rejects response_format with HTTP 400, inspect vLLM version/support rather than silently disabling validation. Only review calls use constrained output; QA evaluator decoding remains the same across all conditions.

### Common-parent reviewer regression

The live reviewer incorrectly interpreted the bound variable P as a literal reachability target, despite a correct two-parent intersection question. The review prompt now includes balanced demonstrations of all three operations instead of a reachability-only example and explicitly distinguishes bound output variables. Reviewer-invented IDs and contradictions between task type and path semantics are diagnosed against the question itself and retry the review, preserving valid wording. New reviews record review_protocol=balanced-v2; existing accepted questions remain resumable. Run `python scripts/check_reviewer.py --config "$CONFIG"` for six live positive/negative integration checks before a long generation run. A format error cannot count as a successful negative control.

## Existing server run

Keep the same ROOT and config so the evidence and checkpoints are reused. Do not run prepare, specs or train again. The Qwen server can remain running.

```bash
git pull --ff-only origin codex/wordnet-pilot
export CONFIG=configs/small-qwen2.5-7b-cuda.json
export ROOT=runs/small-local-300ep-cuda
export TOHE_QWEN_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions

python scripts/run_experiment.py --stage generate --config "$CONFIG" --root "$ROOT" --max-calls 720
```

Small requires at least 240 logical calls: 120 generation + 120 review. The 720 budget covers up to three full attempts per question; a question rejected three times stops the run for inspection. Repeat the command only if the status is partial from reaching the request budget; persistent rejection should be inspected in the failure log.

Questions are saved as **data/qa.audited-v3.json**. Old data/qa.json, evaluation outputs, source specs and checkpoints remain untouched. Do not delete them to resume. Confirm generation_status=complete and sample the English questions with their reconstructed interpretation before evaluation:

```bash
python - <<'PY'
import os
from pathlib import Path
from tohe.io import read
qa = read(Path(os.environ['ROOT']) / 'data/qa.audited-v3.json')
print(qa['metadata'])
assert qa['metadata']['generation_status'] == 'complete'
assert all(q.get('semantic_review') == 'model_verified_v3' for q in qa['questions'])
for task in ('reachability', 'common_parents', 'exact_k'):
    for q in [q for q in qa['questions'] if q['split'] == 'dev' and q['spec']['type'] == task][:3]:
        print(q['id'], q['spec'], q['question'], q['semantic_audit']['text'], sep='\n')
PY
```

After sample review:

```bash
python scripts/run_experiment.py --stage evaluate --config "$CONFIG" --root "$ROOT" --split dev --max-calls 210
python scripts/run_experiment.py --stage report --config "$CONFIG" --root "$ROOT" --split dev
```

New results are in **evaluation/Qwen_Qwen2.5-7B-Instruct/prompt-v2/audited-v3/dev/**. The summary separates unknown_id_rate, answer_type_error_rate and parse_error_rate. Old cached predictions are not reused. Compare raw predictions and question meaning as well as aggregate scores; no improvement is guaranteed. This is a revised development protocol, not an untouched test result.

For the full profile, generation requires at least 3,600 requests (1,800 writer + 1,800 reviewer); the three-attempt upper budget is 10,800. Evaluation counts are unchanged. Use a separate full ROOT.
