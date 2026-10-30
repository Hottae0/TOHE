# Hierarchy QA diagnostic v1

This exploratory development evaluation reuses all 33 original-dev contexts and
the exact completed 300-epoch, seed-11 evidence-local checkpoints from the interim
experiment. It does not select graphs by model accuracy, change embedding losses,
retrain embeddings, or inspect test QA results.

The frozen design contains two questions per context per task, 198 questions and
2,178 evaluated responses across the original 11 conditions. Question generation
and blind semantic review require additional local Qwen requests, normally 396
before any rejected wording is regenerated. Every rejection is retained.

## Definitions fixed before evaluation

- **Ancestor choice (66):** exactly one of four candidates is a strict ancestor
  reached by outgoing is_a edges. Prefer an indirect positive when available;
  distractors include a reverse-relation node and unrelated nodes when available.
  Correct-option positions are balanced 17/17/16/16. Candidate sets are not shown
  to the blind reviewer outside the question itself.
- **Taxonomic relation (66):** classify LEFT relative to RIGHT as supertype,
  subtype, or unrelated. Gold labels are balanced 22/22/22. Exclude unordered
  node pairs previously queried by reachability/common-parents in the same graph.
- **Lowest common ancestors (66):** intersect reflexive ancestor sets, then retain
  all minimal elements in the subtype partial order. A queried node can itself be
  the answer. Multiple answers and empty answers are allowed. Prefer incomparable
  query nodes with shared ancestors. Do not assume a tree or a unique LCA.

English wording is generated from the frozen specification and checked in an
independent Qwen call. The reviewer receives generic demonstrations and the
question only, without its expected specification, graph, or gold answer. Its
generic structured-output schema does not force the expected IDs or task.
All formal answers are computed by graph algorithms. Failed wording is retried
at most six times per question, then the run stops with inspectable diagnostics.

All conditions contain the complete original directed edges. Embedding arms add
the original codes or full learned coordinates. No answer, shortest-path list,
ancestor list, or candidate similarity score is exposed as embedding metadata.
Only the task instructions change from the preceding QA protocol, so previous
and new accuracies must not be pooled as one benchmark.

## Validation and interpretation

The pipeline checks every graph/checkpoint identity and full training completion,
semantics of generated QA, Qwen model/context availability, and all 2,178 actual
chat-template token budgets including response allowance before evaluation.
It writes summaries by task, graph size, answer size/emptiness, relation label,
multi-parent structure, and positive ancestor hop count. Paired intervals resample
the 33 evidence contexts. Completeness, duplicate keys, and gold consistency are
verified before reporting success.

`embedding_diagnostics.json` additionally reports existing observed-graph MAP and
hard-code triplet performance, plus native coordinate/code four-candidate nearest
neighbor accuracy. Ties receive fractional credit. These distances are symmetric
and are not directional entailment decoders; their scores are reconstruction
diagnostics, not held-out link prediction results. Random candidate choice has
25% expected accuracy. Constant relation-label plus empty-set baselines are also
reported. No development outcome establishes final test performance.

Validation: all 30 local tests passed, including DAG multi-answer
LCA semantics, reflexive ancestors, relation direction, balanced/unique sampling,
strict answer constraints, preservation of every directed fact across all 11
conditions, and a mocked evaluation/resume/report round trip. The live server
preflight passed on the actual 33 graphs and 132 completed checkpoints, with
198 questions, 66 per task, 22 per relation label, and zero semantic duplicates.
Live wording inspection added a deterministic requirement that LCA questions
explicitly state the reflexive convention; previously accepted wording is
revalidated and any repaired question retains its prior audit in the failure log.
The graph specification and learned embeddings are unchanged by wording repairs.

## Server execution

Use a new output root, the preserved code snapshot containing these added files,
the interim source root, and the existing checkpoint root:

```bash
python -u scripts/run_hierarchy_diagnostic.py \
  --source-root /home/work/pilot_experiment1/TOHE-expanded-1800-raw/interim-completed-dev \
  --checkpoint-root /home/work/pilot_experiment1/TOHE-expanded-1800-raw/checkpoints \
  --config /home/work/pilot_experiment1/TOHE-expanded-1800-raw/experiment-config.json \
  --root /home/work/pilot_experiment1/TOHE-expanded-1800-raw/hierarchy-diagnostic-v1
```

`--stage prepare` performs the frozen-specification and checkpoint preflight without
LLM requests. The default `all` stage generates/audits QA, checks token lengths,
evaluates all conditions, and writes `evaluation/summary.csv`,
`evaluation/paired_comparisons.json`, `evaluation/verification.json`, and
`evaluation/embedding_diagnostics.json`. The code is deterministic before the
explicit LLM wording step, and compatible cached questions/predictions are reused.
