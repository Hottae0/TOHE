"""Create full and small standalone configs with the same seven conditions."""
from copy import deepcopy
from tohe.io import read, write

base = read("configs/pilot.json")
base["generation_llm"].update(model="gpt-4o-mini-2024-07-18", api_key_env="OPENAI_API_KEY")
base["llm"].update(model="gpt-4o-mini-2024-07-18", api_key_env="OPENAI_API_KEY")
profiles = {"gpt4o-mini": deepcopy(base)}
base["llm"].update(model="Qwen/Qwen2.5-7B-Instruct", api_key_env="TOHE_QWEN_KEY", endpoint_env="TOHE_QWEN_ENDPOINT")
profiles["qwen2.5-7b"] = deepcopy(base)
for name, profile in profiles.items():
    write(f"configs/{name}.json", profile)
    small = deepcopy(profile)
    small["protocol"] += "-small-diagnostic"
    small["training"].update(epochs=30, burnin_epochs=3, seeds=[11])
    small["qa"].update(dev_graphs=5, test_graphs=15, nodes_per_graph=[20, 40])
    write(f"configs/small-{name}.json", small)
    print(f"Wrote full and small configs for {name}")
