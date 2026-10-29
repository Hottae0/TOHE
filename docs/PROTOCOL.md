# Frozen protocol and implementation decisions

## Scope fixed in the conversation

Seven main conditions only; evidence-local reconstruction on WordNet mammals subgraphs; graph → text → frozen LLM; English prompts; original direct facts preserved; fixed serializer; no learned quantizer or LLM tuning. Loss-component ablation code was explicitly deferred by the user.

The original conversation specified objective families, not complete executable equations. The choices below make them executable. They are **implementation choices requiring development-set validation**, not equations reproduced from a published TOHE paper.

## Source protocol

[Poincare Embeddings](https://arxiv.org/abs/1705.08039) studies WordNet noun transitive closure using reconstruction and held-out links. We use its mammals preprocessing as the source graph, then create fixed evidence subgraphs and fit each separately. This local QA protocol is not the original full-mammals reconstruction experiment, WN18RR, or inductive new-entity inference.

Reference repository: [facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings), commit `dff101b5089a635a751aa45e3a6cb21670a3588e`.

Its mammals script specifies d=5, lr=0.3, 300 epochs, 50 negatives, 20 burn-in epochs, batch size 10. These populate our initial config. This repository uses synchronous dense Riemannian SGD rather than the upstream asynchronous sparse implementation. It is an independent implementation, **not a numerically exact reproduction**. The source data was separately compared with the upstream preprocessing output; see VALIDATION.md.

Direct edges are child → immediate hypernym, including instance hypernyms. All applicable parents in each evidence are preserved. The upstream preprocessing defines the source graph. Actual QA embedding training recomputes the closure from the visible evidence edges only. It never slices the global closure: an A → X → B path contributes nothing when X is outside evidence. Global pretrained embeddings are never used for initialization.

Run prepare → specs → train → generate → evaluate → report. A checkpoint is stored at `<context_id>/<method>-<seed>.pt` and reused for the six questions of that evidence. It contains only evidence nodes. Graph hashes are recomputed from visible nodes/edges, including isolated nodes, rather than trusting dataset hash labels. Checkpoints from the old global protocol are rejected. Every embedding checkpoint required by a split is verified before any evaluation API calls. Small runs fit 20 × 4 × 1 = 80 local models; full runs fit 300 × 4 × 3 = 3,600.

## Embeddings and fixed text codes

Curvature is fixed at -1. Poincare points remain inside the unit ball. Euclidean points are unconstrained. With rho the manifold distance to the origin and u=z/||z||:

- Fixed radial centers: c_k=(k+1/2) R/K, k=0,...,K-1.
- Fixed directions: normalized Gaussian vectors drawn once with prototype seed 2026, shared across methods.
- Hard code: nearest radial center and direction of maximum cosine similarity.
- R=6, K=8, J=8 in the initial config. Radii outside the center range map to the nearest endpoint bin.
- No learnable centers, boundaries, codebook, decoder, adapters, or vocabulary additions.

Soft surrogates used only to train embeddings:

    p_r(k|v) = softmax_k(-((rho_v-c_k)/(R/K))^2 / tau)
    p_s(j|v) = softmax_j(<u_v,a_j> / tau)
    q_v(k,j) = p_r(k|v) p_s(j|v)

Hard argmax/nearest assignments are used at inference and for code metrics. tau=0.25 is fixed initially. Exact zero vectors use normalization epsilons; no synthetic root or root-depth labels are introduced.

## Common graph loss

For observed local child/ancestor pair (u,v), sample negatives w only from that evidence's nodes for which (u,w) is not a local observed training pair, excluding self. Use sampled cross entropy over the positive and negative candidates with score -d(u,v). Poincare uses hyperbolic distance; Euclidean uses squared Euclidean distance as in the original baseline family. All conditions use the same observed pairs, seed-specific sampling procedure, dimension, epoch budget and batch size.

If a source has no valid local negatives, omit that source's pairs from the contrastive graph loss but keep their applicable auxiliary terms. Self indices used for tensor padding have masked scores and are rejected as incomparable negatives. No external nodes or false negatives are substituted. Record `sampling.no_negative_pairs` in each checkpoint. If every pair is ineligible, base models have zero graph gradient; such graph metrics are trivial and should not support a performance claim. Graphs with no observed relations fail explicitly.

This is a conventional sampled-softmax implementation. Upstream negative distributions, self candidates, score normalization, parallel update ordering and ranking ties can differ. Consequently local MAP/Rank should not be placed beside published table numbers as an exact reproduction without separately running the official evaluator.

## v1

    L_v1 = L_graph + lambda_r L_radial + lambda_s L_sector + lambda_c L_collision

- Radial: mean relu(m_r + rho_parent - rho_child) on observed closure pairs. This is an explicit hierarchy bias for WordNet, not an assumption that radius is literal graph depth.
- Sector: mean -log(sum_j p_s(j|child) p_s(j|ancestor)). Comparable nodes are encouraged toward compatible directions; no unique branch labels are used. Multiple parents can create conflicting preferences; this is a real limitation to measure.
- Collision: mean sum_(k,j) q_u(k,j) q_w(k,j) on incomparable node pairs. It penalizes joint radial/directional collisions, not merely radial collisions. All distinct nodes retain their IDs regardless of collision.

## v2

    L_v2 = L_graph + lambda_r L_radial + lambda_q L_code_structure + lambda_h L_hard

Prototype x_(k,j)=tanh(c_k/2) a_j. Define D_ab=d_H(x_a,x_b)/R (zero diagonal). For soft codes, d_bar(u,v)=q_u^T D q_v.

- Code structure: mean relu(m_q + d_bar(child,ancestor) - d_bar(child,incomparable)).
- Hard: mean [H(p_r)+H(p_s)] / log(KJ) on nodes involved in the minibatch.
- Radial: same as v1.

v2's expected soft distance is a training surrogate, not necessarily a true metric; in particular a diffuse code can have positive self-distance. Low entropy can collapse codes. This implementation does not claim either problem is solved. The final QA only sees hard code memberships and explicit facts; it does **not** receive the full prototype distance table. A failure to transfer code-distance preservation into QA is informative.

Initial weights are 0.1 for radial/sector/collision/structure and 0.01 for hard; margins 0.1 and 0.2. They are unvalidated starting values. There is no staged loss schedule or loss-ablation runner. All active terms are used from the start, with the common learning-rate burn-in.

## QA construction and deduplication

Select connected evidence neighborhoods by weak/undirected traversal solely for sampling, then retain every original directed edge among selected nodes. This does not change the relation direction. Sizes cycle through 20, 40, 80. Six questions per context:

1. Directed reachability, one true and one false.
2. Common direct parents, one nonempty and one empty intersection.
3. Exact 2-hop and exact 3-hop sets. Prefer alternating empty/nonempty answers but fall back to another unique source when the preferred category is exhausted. Report actual strata; do not claim exact balance.

Deduplication uses original synset IDs and operation: reachability(source,target), common_parents(sorted pair), exact_k(source,k). It ignores wording, aliases, answer and context, conservatively rejecting even cases whose local answer would differ. Identical evidence graphs are also excluded globally. This does not forbid reuse of individual entities or isomorphic but differently labeled graphs.

LLM-assisted generation: code chooses the semantic query; an LLM writes English and echoes its specification; code checks that specification. Algorithmic gold is never given to the question writer or evaluator prompt. Formal task semantics are included in every evaluator prompt to remove ambiguity. This means the main test is graph reasoning with explicit task semantics, not natural-language parsing alone. Semantic equivalence of the generated wording still needs a human sample audit.

Questions are generated once and shared across all conditions, seeds and evaluator models. Partial generation and evaluation are cached; prompt/config/data/checkpoint mismatches cause errors. Never change a dataset while resuming a run.

## Metrics and interpretation

- Mean filtered rank: candidate positives other than target excluded, self excluded, midpoint ties.
- MAP: all other nodes ranked with deterministic ID tie-breaks; sources without positives excluded.
- Hard-code order accuracy: observed ancestor vs sampled incomparable, ties count 0.5; fixed evaluator seed.
- QA: exact boolean or set equality; invalid JSON/type = wrong. Set F1 for set answers only.
- Actual prompt tokens from endpoint usage; no token parity claim. All facts remain present.
- Per-seed reports and graph-cluster paired bootstrap versus Direct ID. Context overlap limits interpretation of confidence intervals; they do not establish unseen-entity generalization.

Every main condition now starts from the same visible evidence. Embedding fitting remains transductive within each observed subgraph, and its cost must be counted. No graph weights are transferred from development to test evidence; only development-selected hyperparameters and the fixed serializer are shared. Different contexts can still share entities/structure, so entity-disjoint generalization is not established.
