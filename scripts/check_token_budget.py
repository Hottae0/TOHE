"""Check actual Qwen chat-template token lengths, including unrounded coordinates."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tohe.io import read, write, digest
from tohe.questions import canonical_question
from tohe.serialize import serialize

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--qa", required=True)
parser.add_argument("--checkpoints", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=8192)
args = parser.parse_args()
cfg, dataset = read(args.config), read(args.qa)
tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["model"], local_files_only=True)
maximum, cache, count = {}, {}, 0
for q in dataset["questions"]:
    item = dict(q, question=q["question"] or canonical_question(q["spec"]))
    for condition in cfg["conditions"]:
        values = None
        if not condition.startswith("direct_"):
            variant = condition.removeprefix("raw_")
            key = (q["context_id"], variant)
            if key not in cache:
                path = Path(args.checkpoints) / q["context_id"] / f"{variant}-11.pt"
                ckpt = torch.load(path, weights_only=True, map_location="cpu")
                cache[key] = {field: dict(zip(ckpt["nodes"], ckpt[field].tolist()))
                              for field in ("codes", "embeddings")}
            values = cache[key]["embeddings" if condition.startswith("raw_") else "codes"]
        prompt = serialize(item, condition, values)
        n = len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=True, add_generation_prompt=True))
        count += 1
        if n > maximum.get(condition, {}).get("prompt_tokens", 0):
            maximum[condition] = dict(question_id=q["id"], prompt_tokens=n,
                                      total_budget=n + cfg["llm"]["max_output_tokens"])
result = dict(prompts_checked=count, limit=args.limit, max_by_condition=maximum,
              chat_template_hash=digest(tokenizer.chat_template),
              canonical_placeholder_used=any(q["question"] is None for q in dataset["questions"]))
write(args.output, result)
print(result)
if any(x["total_budget"] > args.limit for x in maximum.values()):
    raise SystemExit("Prompt budget exceeded; increase server context length before evaluation")
