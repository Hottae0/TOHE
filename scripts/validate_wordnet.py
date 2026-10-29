"""Short real-data validation, explicitly separate from the 300-epoch experiment."""
import sys
from pathlib import Path
from collections import Counter
from tohe.io import read, write
from tohe.evidence import train_evidence
from tohe.qa import make_specs
from tohe.pipeline import evaluate

graph = read(sys.argv[1])
root = Path(sys.argv[2])
cfg = read("configs/small-qwen2.5-7b.json")
cfg["protocol"] = "real-wordnet-two-epoch-validation-NOT-final-results"
cfg["training"].update(epochs=2, batch_size=128, negatives=10, seeds=[11], burnin_epochs=1)
write(root / "config.json", cfg)
qa = make_specs(graph, cfg, root / "qa-specs.json")
results = train_evidence(qa, cfg, root / "checkpoints")
write(root / "embedding-validation.json", results)
dry = evaluate(qa, cfg, root / "dry-run", root / "checkpoints", "test", 20000, True)
write(root / "validation-summary.json", {"graph_hash": graph["graph_hash"],
      "graph_metadata": graph["metadata"], "qa_metadata": qa["metadata"],
      "questions_by_type": dict(Counter(q["spec"]["type"] for q in qa["questions"])),
      "questions_by_split": dict(Counter(q["split"] for q in qa["questions"])),
      "dry_run": dry, "llm_calls": 0, "final_experiment_run": False})
