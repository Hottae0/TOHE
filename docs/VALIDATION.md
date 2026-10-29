# Implementation validation — 2026-10-29

These checks validate execution and data handling. They are not final scientific results.

## Current evidence-local revision

- 12 protocol tests passed locally. Added checks for excluding paths through invisible nodes, rejecting conflicting context IDs, reusing the same evidence embeddings across questions regardless of gold, masking no-negative sources, and rejecting global/wrong-evidence checkpoints before any API call.
- Real WordNet small profile: 20 evidence subgraphs (20 or 40 nodes each), 120 unique semantic questions, 30 dev / 90 test, 84 shared original nodes across the splits.
- Ran 80 independent local fits (20 contexts × 4 methods × 1 seed), each for two validation epochs. Every checkpoint contains only the corresponding evidence nodes.
- Completed 630 prompt-only test evaluations, with zero LLM calls. This is a pipeline check, not a full small-study performance result. The small config now uses 300 epochs / 20 burn-in; that longer study has not been executed.
- Server entry point now defaults to `runs/evidence-local`; use a fresh `runs/small-local-300ep` for the small profile. Old global checkpoints and earlier 30-epoch small checkpoints are incompatible.

## Earlier global-protocol validation (historical, not current QA training)

- Windows, Python 3.12, PyTorch 2.15.0, NLTK 3.10.4, WordNet 3.0.
- Independently ran the upstream `wordnet/transitive_closure.py` at commit `dff101b5089a635a751aa45e3a6cb21670a3588e`.
- Compared our training relation set against the resulting upstream mammals CSV: **exact set equality**.
- Retained 1,180 synsets, 1,180 direct edges, and 6,540 filtered training closure pairs. In this extraction, closure of retained direct edges also equals the training relation set.
- Trained all four embedding conditions for **two validation epochs**, one seed, batch size 128 and 10 negatives. All losses/embeddings were finite. This intentionally shortened validation is separate from the checked-in 300-epoch main configs.
- Built **1,800 semantic question specifications**, 600 for each of reachability, common direct parents and exact-k. Development 300, test 1,500.
- Semantic duplicate queries: **0** across the entire set, including dev/test. Identical evidence contexts: **0**. Shared original nodes between dev/test: **874** in this run. No entity-disjoint claim.
- Generated **10,500 dry-run prompts**: 1,500 questions × 7 conditions, with one embedding seed in this validation. No API calls were made.
- Tests cover multiple parents, exact-length paths versus shortest paths, cycles, all-edge preservation in every serializer, no gold in evaluator prompts, semantic dedup across aliases, code probabilities and finite gradients, four trainers, dry-run resume, question generation resume, strict answer typing, epoch checkpoint resume, and prediction/report integration using mocks.

## Not completed / deliberately deferred

- Full 300-epoch × 3-seed study.
- Actual LLM-generated English questions, semantic wording audit, or LLM QA scores.
- Qwen/vLLM live integration (no server configured locally).
- Linux/Docker execution on the user's server; a CI workflow is supplied but its success must be checked separately.
- Original paper's numerical ranking/MAP reproduction; only its **data preprocessing** was verified exactly. Our optimizer/sampling/evaluator details are documented separately.
- Loss-component ablations, Euclidean TOHE counterparts, shuffled codes, alternate serializers, held-out link prediction, external QA datasets.

## How to repeat validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_wordnet.py runs/evidence-local/data/graph.json runs/local-validation
```

The latter writes a separate short-run config, embeddings, metrics, task specs and prompt-only records. It does not call a language model. Do not insert its numerical scores into a main experiment table.

## Qwen-only configuration revision

Question generation and evaluation both use Qwen2.5-7B-Instruct through the same local endpoint. Full/small CPU and CUDA profiles are provided. GPT profiles were removed. Existing generated-question caches must not be resumed with a changed generator; use a fresh ROOT. Same-model generation/evaluation bias remains. GPU training and live vLLM were not executed locally.

## Question-generation validation repair

14 tests pass locally, including malformed JSON and changed-spec correction, full rejection logging, bounded request budgets, repeated failure without accepting invalid questions, and resume. Invalid responses now trigger up to three attempts with exact-spec feedback, each counted against max-calls. Existing completed questions are retained; newly generated questions record prompt_version=exact-spec-repair-v2. The user's Qwen rejection has not been reproduced live locally; full server responses are needed to establish the exact mismatch.

## Audited wording and task-specific prompt revision

18 local protocol tests pass, including count-versus-set rejection despite exact spec echo, blind semantic reconstruction rejecting shortest-distance interpretation, candidate persistence across request budgets, fail-closed review, canonical provenance, checkpoint reuse, and unknown-node ID classification. Main server and CLI now use audited-v3 questions; the legacy echo-only function remains for historical tests. New prompt-v2/audited-v3 outputs do not overwrite previous experiments. No live GPU/Qwen validation of this revision was run locally. See QA_REPAIR.md for migration and same-model reviewer limitations.
