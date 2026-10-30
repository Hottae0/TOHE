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

Validation: all 31 local tests passed, including DAG multi-answer
LCA semantics, reflexive ancestors, relation direction, balanced/unique sampling,
strict answer constraints, preservation of every directed fact across all 11
conditions, and a mocked evaluation/resume/report round trip. The live server
preflight passed on the actual 33 graphs and 132 completed checkpoints, with
198 questions, 66 per task, 22 per relation label, and zero semantic duplicates.
Live wording inspection added a deterministic requirement that LCA questions
explicitly state the reflexive convention; previously accepted wording is
revalidated and any repaired question retains its prior audit in the failure log.
The graph specification and learned embeddings are unchanged by wording repairs.

## Verified exploratory results (2026-10-30)

All 2,178 responses completed. Local reanalysis verified all 78 exported file
hashes, recomputed graph gold and scoring, and found no missing, unexpected or
duplicate predictions in either the new diagnostic or previous interim run.

| Condition | Strict accuracy | Content-only sensitivity |
|---|---:|---:|
| direct_id | 46.46% | 46.46% |
| direct_bfs | 42.42% | 42.93% |
| direct_dfs | 43.94% | 44.44% |
| euclidean | 30.81% | 42.93% |
| poincare | 31.31% | 44.95% |
| v1 | 27.27% | 39.90% |
| v2 | 26.26% | 36.87% |
| raw_euclidean | 20.20% | 37.88% |
| raw_poincare | 23.23% | 40.40% |
| raw_v1 | 24.75% | 39.39% |
| raw_v2 | 24.24% | 38.38% |

The original strict results are preserved. The last task instruction requests a
label string/array while the common instruction requests an answer object; many
embedding-arm relation responses used an unwrapped valid label. This exposes an
output-format confound. Content-only sensitivity unwraps valid JSON values and
ignores set ordering/duplicates, without ID repair, synonym mapping, explanatory
answer extraction, gold-dependent parsing or extra LLM calls. It is post hoc,
not a replacement preregistered metric. V1/V2 still do not outperform direct_id.

Native four-candidate coordinate retrieval was 75.76% for Euclidean and 100% for
Poincare/V1/V2 on the same 66 observed-graph questions. Native code retrieval was
59.85%, 74.24%, 56.82%, 65.15%, respectively, with fractional tie credit. These
are reconstruction diagnostics. They suggest information loss in the code
mapping; they do not establish held-out QA generalization or a causal source of
LLM errors. The input does not provide the code-direction prototypes or full
code-distance table used by the native decoder. Raw vectors also increase prompt
length. Format consistency and decodable metadata should be controlled before
claiming a reasoning advantage. All 66 LCA questions happened to have one answer;
only four of the 33 contexts have multiple parents, limiting DAG coverage.

The exported archive is `TOHE-hierarchy-results-20261030.tar.gz`, SHA-256
`5855d0071726378bc255c0d22a06cc9e57cf2125543cbd3013bc7cb967821424`.

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
