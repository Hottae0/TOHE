"""Fixed serialization-aware losses. No learned codebook or LLM adapter."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from .io import digest, write


def distance(x, y, hyperbolic=True):
    sq = (x - y).square().sum(-1)
    if not hyperbolic:
        return sq  # squared Euclidean score as in the Poincare baseline
    denominator = ((1 - x.square().sum(-1)) * (1 - y.square().sum(-1))).clamp_min(1e-12)
    return torch.acosh((1 + 2 * sq / denominator).clamp_min(1 + 1e-12))


class FixedSerializerGeometry:
    def __init__(self, config, dimension, dtype=torch.float64, device="cpu", hyperbolic=True):
        self.hyperbolic = hyperbolic
        self.max_radius = config["max_radius"]
        generator = np.random.default_rng(config["prototype_seed"])
        directions = generator.normal(size=(config["directions"], dimension))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.directions = torch.tensor(directions, dtype=dtype, device=device)
        k = config["radial_bins"]
        self.centers = (torch.arange(k, dtype=dtype, device=device) + 0.5) * self.max_radius / k
        radii = torch.tanh(self.centers / 2) if hyperbolic else self.centers
        self.prototypes = (radii[:, None, None] * self.directions[None]).reshape(-1, dimension)
        if hyperbolic:
            table = distance(self.prototypes[:, None], self.prototypes[None, :], True)
        else:
            table = torch.cdist(self.prototypes, self.prototypes)
        self.table = table / self.max_radius
        self.table.fill_diagonal_(0)

    def radius(self, z):
        norm = z.norm(dim=-1).clamp_min(1e-12)
        return 2 * torch.atanh(norm.clamp_max(1 - 1e-7)) if self.hyperbolic else norm

    def soft(self, z, temperature):
        rho = self.radius(z)
        # Divide by bin width so temperature has the same meaning across code budgets.
        scale = self.max_radius / len(self.centers)
        pr = torch.softmax(-((rho[..., None] - self.centers) / scale).square() / temperature, -1)
        unit = torch.nn.functional.normalize(z, dim=-1, eps=1e-12)
        ps = torch.softmax(unit @ self.directions.T / temperature, -1)
        joint = (pr[..., :, None] * ps[..., None, :]).flatten(-2)
        return rho, pr, ps, joint

    def hard(self, z):
        radius = self.radius(z)
        r = (radius[:, None] - self.centers).abs().argmin(-1)
        s = (torch.nn.functional.normalize(z, dim=-1) @ self.directions.T).argmax(-1)
        return torch.stack([r, s], -1)


def auxiliary(z, pairs, negatives, comparable, geometry, cfg, variant):
    zero = z.sum() * 0
    losses = {name: zero for name in ["radial", "sector", "collision", "structure", "hard"]}
    if variant not in ("v1", "v2"):
        return losses
    child, parent = pairs[:, 0], pairs[:, 1]
    rho, pr, ps, q = geometry.soft(z, cfg["temperature"])
    losses["radial"] = torch.relu(cfg["radial_margin"] + rho[parent] - rho[child]).mean()
    # Auxiliary negatives must be incomparable, even though the base directed
    # ranking loss permits reverse-direction pairs as negatives.
    u = child[:, None].expand_as(negatives).reshape(-1)
    w = negatives.reshape(-1)
    v = parent[:, None].expand_as(negatives).reshape(-1)
    valid = ~comparable[u, w]
    u, v, w = u[valid], v[valid], w[valid]
    if variant == "v1":
        same_direction = (ps[child] * ps[parent]).sum(-1).clamp_min(1e-12)
        losses["sector"] = -same_direction.log().mean()
        if len(u):
            losses["collision"] = (q[u] * q[w]).sum(-1).mean()
    else:
        if len(u):
            near = ((q[u] @ geometry.table) * q[v]).sum(-1)
            far = ((q[u] @ geometry.table) * q[w]).sum(-1)
            losses["structure"] = torch.relu(cfg["code_margin"] + near - far).mean()
        used = torch.unique(torch.cat([child, parent, negatives.reshape(-1)]))
        entropy = -(pr[used] * pr[used].clamp_min(1e-12).log()).sum(-1)
        entropy -= (ps[used] * ps[used].clamp_min(1e-12).log()).sum(-1)
        losses["hard"] = entropy.mean() / math.log(q.shape[-1])
    return losses


def train(graph, config, variant, seed, output):
    if variant not in ("euclidean", "poincare", "v1", "v2"):
        raise ValueError(variant)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {output}")
    cfg = config["training"]
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dtype = getattr(torch, cfg["dtype"])
    device = cfg["device"]
    hyperbolic = variant != "euclidean"
    nodes = sorted(graph["nodes"])
    index = {node: i for i, node in enumerate(nodes)}
    if not graph["closure"]:
        raise ValueError("Cannot train a graph without any observed relation")
    if cfg["negatives"] < 1:
        raise ValueError("negatives must be positive")
    pairs = torch.tensor([(index[u], index[v]) for u, v in graph["closure"]], device=device)
    n = len(nodes)
    if n > 10000:
        raise ValueError("This implementation uses dense candidate masks; use mammals, not full nouns")
    positive = torch.zeros((n, n), dtype=torch.bool, device=device)
    positive[pairs[:, 0], pairs[:, 1]] = True
    comparable = positive | positive.T | torch.eye(n, dtype=torch.bool, device=device)
    allowed = ~(positive | torch.eye(n, dtype=torch.bool, device=device))
    candidates = [torch.where(allowed[i])[0] for i in range(n)]
    has_negatives = torch.tensor([len(c) > 0 for c in candidates], device=device)
    no_negative_pairs = int((~has_negatives[pairs[:, 0]]).sum())
    z = torch.nn.Parameter(torch.empty(n, cfg["dimension"], dtype=dtype, device=device).uniform_(-0.001, 0.001))
    geometry = FixedSerializerGeometry(config["serializer"], cfg["dimension"], dtype, device, hyperbolic)
    partial = output.with_suffix(".partial.pt")
    history, prior_seconds = [], 0.0
    if partial.exists():
        saved = torch.load(partial, map_location=device, weights_only=True)
        if (saved["graph_hash"], saved["config_hash"], saved["variant"], saved["seed"]) != (graph["graph_hash"], digest(config), variant, seed):
            raise ValueError("Partial checkpoint belongs to different data/config/condition/seed")
        with torch.no_grad():
            z.copy_(saved["embeddings"])
        torch.set_rng_state(saved["rng_state"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
        history, prior_seconds = saved["history"], saved["seconds"]
    start = time.perf_counter()
    for epoch in range(len(history), cfg["epochs"]):
        totals, batches = {}, 0
        rate = cfg["learning_rate"] / (10 if epoch < cfg["burnin_epochs"] else 1)
        for batch in pairs[torch.randperm(len(pairs), device=device)].split(cfg["batch_size"]):
            neg = torch.stack([candidates[int(u)][torch.randint(len(candidates[int(u)]), (cfg["negatives"],), device=device)]
                               if len(candidates[int(u)]) else u.expand(cfg["negatives"])
                               for u in batch[:, 0]])
            source, target = z[batch[:, 0]], z[batch[:, 1]]
            valid = has_negatives[batch[:, 0]]
            negative_scores = -distance(source[:, None], z[neg], hyperbolic)
            negative_scores = negative_scores.masked_fill(~valid[:, None], -torch.inf)
            scores = torch.cat([-distance(source, target, hyperbolic)[:, None],
                                negative_scores], dim=1)
            per_pair = -torch.log_softmax(scores, 1)[:, 0]
            base = (per_pair * valid).sum() / valid.sum().clamp_min(1)
            extra = auxiliary(z, batch, neg, comparable, geometry, cfg, variant)
            loss = base + sum(cfg["weights"][k] * value for k, value in extra.items())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            with torch.no_grad():
                gradient = z.grad
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite gradient")
                if hyperbolic:
                    gradient = gradient * (1 - z.square().sum(-1, keepdim=True)).square() / 4
                z -= rate * gradient
                if hyperbolic:
                    z *= ((1 - 1e-5) / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
                z.grad = None
            for key, value in {"total": loss, "graph": base, **extra}.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            batches += 1
        history.append({"epoch": epoch + 1, **{k: v / batches for k, v in totals.items()}})
        partial.parent.mkdir(parents=True, exist_ok=True)
        tmp = partial.with_suffix(".tmp")
        torch.save({"embeddings": z.detach().cpu(), "history": history,
                    "graph_hash": graph["graph_hash"], "config_hash": digest(config),
                    "variant": variant, "seed": seed,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                    "seconds": prior_seconds + time.perf_counter() - start}, tmp)
        tmp.replace(partial)
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(f"{variant} seed={seed} epoch={epoch+1} loss={history[-1]['total']:.5f}", flush=True)
    payload = {"variant": variant, "seed": seed, "nodes": nodes, "embeddings": z.detach().cpu(),
               "embedding_scope": graph.get("metadata", {}).get("scope", "unspecified"),
               "sampling": {"no_negative_pairs": no_negative_pairs,
                            "policy": "omit_ineligible_pairs_from_graph_loss_keep_auxiliary_no_external_negatives"},
               "codes": geometry.hard(z).detach().cpu(), "config": config,
               "graph_hash": graph["graph_hash"], "config_hash": digest(config),
               "seconds": prior_seconds + time.perf_counter() - start, "history": history,
               "torch_version": str(torch.__version__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    partial.unlink(missing_ok=True)
    write(output.with_suffix(".json"), {k: v for k, v in payload.items() if k not in ("embeddings", "codes")})
    return payload


def embedding_metrics(graph, checkpoint):
    z = checkpoint["embeddings"]
    hyperbolic = checkpoint["variant"] != "euclidean"
    nodes = checkpoint["nodes"]
    idx = {v: i for i, v in enumerate(nodes)}
    targets = {i: set() for i in range(len(nodes))}
    for u, v in graph["closure"]:
        targets[idx[u]].add(idx[v])
    ranks, aps = [], []
    codes = checkpoint["codes"]
    geometry = FixedSerializerGeometry(checkpoint["config"]["serializer"], z.shape[1], z.dtype, hyperbolic=hyperbolic)
    code_ids = codes[:, 0] * len(geometry.directions) + codes[:, 1]
    correct, comparisons, collisions = 0.0, 0, 0
    rng = np.random.default_rng(901)
    for u, positives in targets.items():
        if not positives:
            continue
        d = distance(z[u], z, hyperbolic).numpy()
        candidates = [j for j in range(len(nodes)) if j != u]
        ordered = sorted(candidates, key=lambda j: (d[j], j))
        hits = 0
        precision = []
        for rank, v in enumerate(ordered, 1):
            if v in positives:
                hits += 1
                precision.append(hits / rank)
        aps.append(sum(precision) / len(positives))
        negatives = [j for j in candidates if j not in positives]
        for v in positives:
            values = d[negatives]
            ranks.append(1 + int((values < d[v]).sum()) + 0.5 * int((values == d[v]).sum()))
        unrelated = [j for j in negatives if u not in targets[j]]
        if unrelated:
            for v in sorted(positives)[:20]:
                w = int(rng.choice(unrelated))
                a = geometry.table[code_ids[u], code_ids[v]].item()
                b = geometry.table[code_ids[u], code_ids[w]].item()
                correct += float(a < b) + 0.5 * float(a == b)
                collisions += int(code_ids[u] == code_ids[w])
                comparisons += 1
    return {"mean_filtered_rank": float(np.mean(ranks)), "map": float(np.mean(aps)),
            "hard_code_triplet_accuracy": correct / comparisons if comparisons else None,
            "incomparable_collision_rate": collisions / comparisons if comparisons else None,
            "triplets": comparisons, "training_seconds": checkpoint["seconds"],
            "no_negative_pairs": checkpoint.get("sampling", {}).get("no_negative_pairs"),
            "embedding_scope": checkpoint.get("embedding_scope"),
            "evaluation_note": "Self excluded; filtered rank uses midranks; MAP uses all other nodes with ID tie-breaks. Not an exact upstream evaluator reproduction."}
