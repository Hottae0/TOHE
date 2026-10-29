# Expanded WordNet study with raw coordinates

The expanded profile is configs/expanded-qwen2.5-7b-cuda.json: 300 evidence graphs, 1,800 QA (600/task), dev 300/test 1,500, sizes 20/40/80, seed 11, 300 epochs/20 burn-in. Four embedding methods are fitted per context: 1,200 checkpoints. Eleven evaluation conditions comprise the original seven plus raw_euclidean, raw_poincare, raw_v1 and raw_v2. Raw arms reuse the same fitted tensors, serialize every coordinate using Python's round-trip JSON float representation without rounding/bucketing, and disclose the coordinate geometry. They are text inputs, not soft-token projections.

Expected evaluation calls: dev 3,300; test 16,500. Question writing and blind model review require at least 3,600 calls; retries share the request limit. Automated same-model review is not an independent guarantee of semantic equivalence.

## Frozen data and structural diagnostics

Old small-study semantic queries and identical evidence graphs are excluded explicitly using --exclude-specs. Deduplication remains based on original synset IDs and operation, not wording. Prior source specs hash and the exclusion size are saved. Test entities can still overlap development entities; no entity-disjoint claim is made.

The mammals source has only ONE multi-parent node (elephant.n.01). Thirty preselected 80-node contexts are anchored around that node and both direct parents to ensure the diagnostic relationship survives induced subgraph sampling. The quota is restricted to size 80 because smaller neighborhoods exhausted unique exact-k source queries in preflight. The remaining contexts are sampled normally and can also include that node. This is a repeated structural diagnostic around one source motif, not broad independent DAG coverage, and size/structure are partly confounded. Do not claim general multi-parent graph performance from it. Report actual counts, shared entities and answer sizes.

BFS/DFS serialization orders use weak adjacency for visiting nodes; every rendered is_a edge retains its original direction. This changes ordering only, not the graph, QA algorithms or embedding loss. The expanded profile rejects samples with identical BFS/DFS visiting orders. The old outgoing-order profiles keep their original behavior. All conditions within the expanded run share facts, questions, graph direction and output rules.

Local full-scale preflight produced 1,800 distinct queries, excluded all 120 prior queries, found 46 multi-parent contexts, and obtained different BFS/DFS orders for all 300 contexts. The exact output is deterministic for the frozen source/config. Empty and nonempty answers, multi-parent presence, and distinct orders are reported as strata.

## Storage and resource requirements

Taeyeong is mounted at **/home/work/pilot_experiment1**, not /home/work/Taeyeong. Use a new child directory there. Do not modify the old study. The launcher verifies the mount before writing. Available filesystem space is not the same as the user's storage quota.

The GPU holds Qwen once; it does not hold 1,200 models concurrently. Embeddings are fitted one context at a time and saved to disk. Peak device memory is therefore governed mainly by a single <=80-node fit, not the total number of contexts. A two-epoch four-method memory probe is provided; its checkpoints are diagnostic and never reused as 300-epoch study fits.

Raw float prompts are longer. Set vLLM --max-model-len 16384 and measure actual input+output lengths with the same local Qwen tokenizer and chat template. The launcher checks the server's advertised context limit and checks all QA prompts before evaluation; it does not silently truncate them. Only the existing .venv-vllm environment needs transformers for this check; the training environment does not gain a new dependency.

## Persistent launcher

Use a code snapshot under the mounted storage and keep the active Python environment. Example paths (verify mount and prior backup first):

```bash
export TOHE_QWEN_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
python scripts/run_expanded.py \
  --root /home/work/pilot_experiment1/TOHE-expanded-1800-raw \
  --source-root /home/work/pilot_experiment1/TOHE-results-20261029-214115/runs/small-local-300ep-cuda \
  --tokenizer-python /home/work/TOHE/.venv-vllm/bin/python \
  --phase dev
```

The dev phase prepares/copies the source, freezes specs with exclusions, trains, generates/audits, checks token limits, evaluates dev, and writes the report. It does NOT evaluate test. Resume with the same config/source/root after an interruption; existing artifacts are checked. job-status.json and resources-at-start.json are saved to the mounted root. Review development outputs and freeze the protocol before invoking --phase test (16,500 requests).

No graph facts are invented to meet the multi-parent quota. Existing data, checkpoints, API keys and model weights are not copied into public sources. A one-seed result does not establish seed robustness, and added coordinate text does not guarantee better QA.

Live 80-node probe: all four trainers completed two validation epochs with <=24 MiB PyTorch reserved memory; overall GPU free memory remained about 11.6 GiB alongside the running Qwen server. With full unrounded float text, the longest measured probe budget was 11,489 tokens including the 256-token response budget. This exceeds 4,096/8,192, so the study uses a 16,384-token server and checks every actual post-training prompt again before evaluation. These measurements describe one evidence graph; they are resource checks, not accuracy results.
