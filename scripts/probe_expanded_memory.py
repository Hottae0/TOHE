"""Two-epoch memory probe on one 80-node evidence; not a study checkpoint."""
import argparse
import copy
import subprocess
from pathlib import Path
import torch
from tohe.io import read, write
from tohe.evidence import local_graph, VARIANTS
from tohe.model import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--specs", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
if root.exists():
    raise FileExistsError("Use a fresh memory-probe directory")
root.mkdir(parents=True)
cfg = copy.deepcopy(read(args.config))
cfg["training"].update(epochs=2, burnin_epochs=1)
dataset = read(args.specs)
sample = next(q for q in dataset["questions"] if len(q["graph"]["nodes"]) == 80)
dataset["questions"] = [q for q in dataset["questions"] if q["context_id"] == sample["context_id"]]
write(root / "specs.json", dataset)
write(root / "probe-config.json", cfg)
graph = local_graph(sample["graph"])
results = {}
for variant in VARIANTS:
    torch.cuda.reset_peak_memory_stats()
    path = root / "checkpoints" / sample["context_id"] / f"{variant}-11.pt"
    train(graph, cfg, variant, 11, path)
    torch.cuda.synchronize()
    results[variant] = dict(peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
results["gpu_after_probe"] = subprocess.check_output([
    "nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader"]).decode()
write(root / "resource-probe.json", results)
print(results)
