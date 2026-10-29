# Comparison with prior work and next evaluations

No loss-component ablation code is included in this version. The following are research gaps and future decisions, not completed experiments or silently enabled conditions.

## Existing evaluation choices

| Source | What it measures | Fit to TOHE |
|---|---|---|
| [Poincare Embeddings](https://arxiv.org/abs/1705.08039) | WordNet reconstruction and held-out relation ranking | Current structural baseline; no LLM QA in the original experiment |
| [GRBench / Graph-CoT](https://arxiv.org/html/2404.07103v3) | 1,740 questions over 10 domain graphs | Useful precedent: manual task templates, GPT-4 wording, algorithmic answers. Different graphs/tasks, not directly comparable scores |
| [GraphDO](https://aclanthology.org/2025.acl-long.321/) | Four description orders, six graph tasks, six LLMs | Supports including order controls; broader structural diversity than mammals |
| [KQA Pro](https://aclanthology.org/2022.acl-long.422/) | Around 120K existing QA with KoPL/SPARQL | Strong later external QA evaluation. Requires multi-relation, attribute, qualifier and program-aware graph handling; not a drop-in mammals test |
| [HyperLex](https://aclanthology.org/J17-4004/) | Human graded lexical entailment for 2,616 pairs | Existing evaluation without new QA generation; measures semantic entailment rather than our explicit graph-context QA |

The examined sources did not establish a ready-made WordNet mammals QA set matching our three exact tasks. We therefore retain generated, algorithm-grounded WordNet questions for the main controlled study. KQA Pro is the preferred later external QA test, rather than fabricating WordNet compatibility. Published question counts guide scale, not statistical equivalence.

## Material gaps, in priority order

1. **No empirical QA result yet.** Model generation and evaluation need configured endpoints. Preparing task specifications is not LLM question generation. Synthetic or two-epoch smoke runs cannot support superiority claims.
2. **Novelty isolation.** Seven conditions test the combined method. Euclidean with corresponding text-oriented losses is still needed to isolate hyperbolic geometry. Shuffled-code and order-only/combined checks are also deferred.
3. **Global graph versus local evidence.** The embeddings encode global mammals relationships, while Direct Text receives local facts. Clearly report this transductive scope; add evidence-local fitting or equivalent global knowledge controls before claiming equal information.
4. **Graph diversity.** Preserving multiple parents does not make mammals representative of dense, cyclic or heterogeneous graphs. In the validated extraction, 1,180 nodes have 1,180 direct edges, so it is very sparse. Add a broader WordNet domain or a suitable external benchmark before general-graph claims.
5. **Question dependence.** Semantic queries and identical evidence graphs are deduplicated globally; individual nodes overlap across dev/test. Do not describe the split as entity-disjoint. Bootstrap intervals are conditional diagnostics.
6. **Language generation.** LLM paraphrases may change meaning despite echoing a correct spec. Audit samples by task. Formal specs help control the graph task but bypass some language interpretation. A future natural-language-only comparison should be separate. Using GPT to generate and evaluate can produce style affinity; Qwen supplies a different evaluator, not a proof of neutrality.
7. **Renderer bottleneck.** v2 optimizes code-prototype distances, but categorical IDs reveal only membership/order information. QA gains are a hypothesis. Do not interpret improved hidden geometry as demonstrated LLM understanding.
8. **Hyperparameter fairness.** Current weights/radius/temperature are starting settings. Use equal development-set search budgets. Never choose weights or winner after inspecting test scores. Baseline optimizer details differ from the original code; run the original baseline/evaluator for paper-level numeric reproduction.
9. **Cost and tokens.** Metadata adds tokens and inference work. Current comparison preserves all facts with differing actual input length. Equal-budget compression and retrieval experiments are future work.
10. **Metric breadth.** This version provides reconstruction, hard-code diagnostics, QA exact match/F1, per-stratum reports and intervals. It does not implement held-out link prediction, human QA review outcomes, or significance claims across independent graphs/domains.

## Reporting language

Use: “We use LLM-assisted English question generation from graph-defined task specifications, with algorithmically computed ground truth.”

Avoid: “The LLM generated the ground-truth answers”; “all questions/entities are independent”; “we reproduce the original Poincare results”; “TOHE improves QA” before full validated runs.

The study can proceed without any additional benchmark code being added now. Add new evaluations only after reviewing the first complete results and these limitations.
