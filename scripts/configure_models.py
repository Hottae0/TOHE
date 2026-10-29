"""Create full and small standalone configs with the same seven conditions."""
from copy import deepcopy
from tohe.io import read, write

base = read("configs/pilot.json")
for field in ("llm", "generation_llm"):
    base[field].update(model="Qwen/Qwen2.5-7B-Instruct", api_key_env="TOHE_QWEN_KEY",
                       endpoint_env="TOHE_QWEN_ENDPOINT")
profiles = {"qwen2.5-7b": deepcopy(base)}
for name, profile in profiles.items():
    write(f"configs/{name}.json", profile)
    small = deepcopy(profile)
    small["protocol"] += "-small-diagnostic"
    # Reduce evaluation breadth and repetitions, not the optimization horizon.
    small["training"].update(seeds=[11])
    small["qa"].update(dev_graphs=5, test_graphs=15, nodes_per_graph=[20, 40])
    write(f"configs/small-{name}.json", small)
    for prefix, config in (("", profile), ("small-", small)):
        cuda = deepcopy(config)
        cuda["training"]["device"] = "cuda"
        write(f"configs/{prefix}{name}-cuda.json", cuda)
    expanded = deepcopy(profile)
    expanded["protocol"] = "wordnet-evidence-local-expanded-raw-v1"
    expanded["training"]["seeds"] = [11]
    expanded["qa"].update(multi_parent_graphs=30, multi_parent_nodes_per_graph=[80],
                          serialization_traversal="weak", require_distinct_orders=True)
    expanded["conditions"] += [f"raw_{v}" for v in ("euclidean", "poincare", "v1", "v2")]
    write(f"configs/expanded-{name}.json", expanded)
    expanded["training"]["device"] = "cuda"
    write(f"configs/expanded-{name}-cuda.json", expanded)
    print(f"Wrote full and small configs for {name}")
