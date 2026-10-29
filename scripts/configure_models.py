"""Create full, standalone run configs without changing the training protocol."""
from pathlib import Path
from tohe.io import read, write

base = read("configs/pilot.json")
base["generation_llm"].update(model="gpt-4o-mini-2024-07-18", api_key_env="OPENAI_API_KEY")
base["llm"].update(model="gpt-4o-mini-2024-07-18", api_key_env="OPENAI_API_KEY")
write("configs/gpt4o-mini.json", base)
base["llm"].update(model="Qwen/Qwen2.5-7B-Instruct", api_key_env="TOHE_QWEN_KEY", endpoint_env="TOHE_QWEN_ENDPOINT")
write("configs/qwen2.5-7b.json", base)
print("Wrote configs/gpt4o-mini.json and configs/qwen2.5-7b.json")
