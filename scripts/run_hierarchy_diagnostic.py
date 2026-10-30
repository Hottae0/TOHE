"""Reuse frozen embeddings for a predeclared 198-question hierarchy diagnostic."""
import argparse
import collections
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import numpy as np
import torch
from tohe import hierarchy_qa as h
from tohe.evidence import evidence_graphs, checkpoint_path, check_checkpoint, VARIANTS
from tohe.io import read, write, append, rows, digest
from tohe.llm import chat, parse_json


def generate(dataset, cfg, path):
    fingerprint = digest({"protocol": h.VERSION, "source": digest(dataset), "config": cfg["generation_llm"],
                          "endpoint": os.environ.get(cfg["generation_llm"]["endpoint_env"])})
    qa = read(path) if path.exists() else copy.deepcopy(dataset)
    if path.exists() and qa.get("generation_hash") != fingerprint:
        raise ValueError("Incompatible generation cache")
    qa["generation_hash"] = fingerprint
    for item in qa["questions"]:
        if item.get("semantic_review") == "model_verified_hierarchy_v1":
            validation_error = h.wording_error(item.get('question'), item['spec'])
            if validation_error is None:
                continue
            append(path.with_suffix('.failures.jsonl'), dict(question_id=item['id'],
                error=validation_error, stage='deterministic_revalidation_before_evaluation', prior_item=copy.deepcopy(item)))
            item['semantic_review'] = 'pending'
        feedback = ""
        for attempt in range(1, 7):
            generation_cfg = dict(cfg["generation_llm"], response_format={"type": "json_schema", "json_schema": {
                "name": "english_graph_question", "schema": {"type": "object", "properties": {
                    "question": {"type": "string"}}, "required": ["question"], "additionalProperties": False}}})
            prompt = {"instruction": "Write one unambiguous English question preserving the exact specification and every output requirement. For lowest_common_ancestors explicitly include: Treat each queried node as its own ancestor. Never solve it or provide a hint about the answer. Return only JSON with question.",
                      "spec": item["spec"], "definition": item["task_definition"],
                      "semantic_reference": h.canonical_question(item["spec"]), "previous_failure": feedback}
            response = chat(generation_cfg, [{"role": "user", "content": json.dumps(prompt)}])
            question, audit = None, None
            try:
                question = parse_json(response["text"])["question"]
                error = h.wording_error(question, item["spec"])
                if error is None:
                    audit = chat(dict(cfg["generation_llm"], temperature=0.0, max_output_tokens=512,
                                      response_format=h.review_schema()), h.review_messages(question))
                    error = h.review_error(parse_json(audit["text"]), item["spec"])
            except (ValueError, KeyError, TypeError) as exc:
                error = f"Invalid generator/reviewer format: {exc}"
            if error is None:
                item.update(question=question.strip(), question_origin="llm_generated",
                    semantic_review="model_verified_hierarchy_v1", generation=response,
                    semantic_audit={**audit, "question_hash": digest(question.strip())})
                write(path, qa)
                print('AUDITED', item['id'], flush=True)
                break
            append(path.with_suffix('.failures.jsonl'), dict(question_id=item['id'], attempt=attempt,
                error=error, generation=response, review=audit))
            feedback = error
            print('REJECTED', item['id'], attempt, error, flush=True)
            if attempt == 6:
                write(path, qa)
                raise ValueError(f"Question {item['id']} failed six attempts; inspect generation failures")
    qa['metadata']['generation_status'] = 'complete'
    write(path, qa)
    return qa


def check_qa(qa):
    assert qa['metadata']['generation_status'] == 'complete'
    for item in qa['questions']:
        assert item['split'] == 'dev' and item['gold'] == h.gold_answer(item['graph'], item['spec'])
        assert item['semantic_review'] == 'model_verified_hierarchy_v1'
        assert item['semantic_audit']['question_hash'] == digest(item['question'])
        error = h.wording_error(item['question'], item['spec']) or h.review_error(parse_json(item['semantic_audit']['text']), item['spec'])
        if error:
            raise ValueError(f"Invalid question {item['id']}: {error}")


def load_checkpoints(dataset, cfg, root):
    cache = {}
    for cid, graph in evidence_graphs(dataset).items():
        for variant in VARIANTS:
            path = checkpoint_path(root, cid, variant, 11)
            ckpt = torch.load(path, weights_only=True, map_location='cpu')
            check_checkpoint(ckpt, graph, cfg, variant, 11)
            assert len(ckpt['history']) == cfg['training']['epochs']
            cache[cid, variant] = (ckpt, hashlib.sha256(path.read_bytes()).hexdigest())
    return cache


def values_for(item, condition, cache):
    if condition.startswith('direct_'):
        return None, None
    ckpt, fingerprint = cache[item['context_id'], condition.removeprefix('raw_')]
    values = ckpt['embeddings' if condition.startswith('raw_') else 'codes']
    return dict(zip(ckpt['nodes'], values.tolist())), fingerprint


def token_budget(qa, cfg, checkpoint_root, path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['llm']['model'], local_files_only=True)
    cache = load_checkpoints(qa, cfg, checkpoint_root)
    maximum, count = {}, 0
    for condition in cfg['conditions']:
        for item in qa['questions']:
            values, _ = values_for(item, condition, cache)
            prompt = h.serialize(item, condition, values)
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            n = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
            count += 1
            if n > maximum.get(condition, {}).get('prompt_tokens', 0):
                maximum[condition] = dict(question_id=item['id'], prompt_tokens=n, total_budget=n+cfg['llm']['max_output_tokens'])
    result = dict(prompts_checked=count, max_by_condition=maximum, context_limit=16384,
                  canonical_placeholder_used=any(not x['question'] for x in qa['questions']))
    write(path, result)
    assert all(v['total_budget'] <= 16384 for v in maximum.values()), 'Prompt exceeds server context'
    print('TOKEN_BUDGET_OK', count, max(v['total_budget'] for v in maximum.values()), flush=True)


def embedding_diagnostics(dataset, cfg, checkpoint_root, cache, output):
    from tohe.model import distance, FixedSerializerGeometry
    groups = collections.defaultdict(list)
    for (cid, variant), (ckpt, _) in cache.items():
        metrics_path = checkpoint_path(checkpoint_root, cid, variant, 11).with_suffix('.metrics.json')
        metric = read(metrics_path)
        nodes = {n: i for i, n in enumerate(ckpt['nodes'])}
        z, codes = ckpt['embeddings'], ckpt['codes']
        geometry = FixedSerializerGeometry(cfg['serializer'], z.shape[1], z.dtype, hyperbolic=variant != 'euclidean')
        ids = codes[:, 0] * len(geometry.directions) + codes[:, 1]
        coordinate_acc, code_acc = [], []
        for q in dataset['questions']:
            if q['context_id'] != cid or q['spec']['type'] != h.TASKS[0]:
                continue
            mapping, spec = q['graph']['original_ids'], q['spec']
            source = nodes[mapping[spec['source']]]
            candidates = [nodes[mapping[c]] for c in spec['candidates']]
            distances = distance(z[source], z[candidates], variant != 'euclidean').tolist()
            code_distances = [geometry.table[ids[source], ids[c]].item() for c in candidates]
            target = spec['candidates'].index(q['gold'][0])
            for values, result in ((distances, coordinate_acc), (code_distances, code_acc)):
                nearest = [i for i, value in enumerate(values) if abs(value-min(values)) <= 1e-12]
                result.append(1/len(nearest) if target in nearest else 0.0)
        groups[variant].append(dict(context_id=cid, **metric,
            candidate_coordinate_accuracy=float(np.mean(coordinate_acc)),
            candidate_code_accuracy=float(np.mean(code_acc))))
    result = dict(note='Observed-graph reconstruction, not held-out link prediction. Candidate retrieval uses symmetric native distances, with fractional credit for ties; it is not a directional entailment decoder. Each graph has equal weight.',
        random_four_candidate_accuracy=0.25, by_variant={})
    for variant, values in groups.items():
        result['by_variant'][variant] = dict(contexts=len(values), graphs=values,
            means={key: float(np.mean([v[key] for v in values if v[key] is not None])) for key in
                ('map', 'hard_code_triplet_accuracy', 'candidate_coordinate_accuracy', 'candidate_code_accuracy')})
    write(output, result)


def evaluate(qa, cfg, checkpoint_root, run_dir):
    check_qa(qa)
    cache = load_checkpoints(qa, cfg, checkpoint_root)
    manifest = dict(protocol=h.VERSION, qa_hash=digest(qa), config_hash=digest(cfg),
                    checkpoint_hashes={cid+'/'+v: fp for (cid, v), (_, fp) in cache.items()},
                    endpoint_hash=digest(os.environ[cfg['llm']['endpoint_env']]))
    if (run_dir/'manifest.json').exists():
        assert read(run_dir/'manifest.json') == manifest, 'Incompatible evaluation cache'
    write(run_dir/'manifest.json', manifest)
    embedding_diagnostics(qa, cfg, checkpoint_root, cache, run_dir/'embedding_diagnostics.json')
    prior = rows(run_dir/'predictions.jsonl')
    completed = {r['key']: r for r in prior}
    assert len(completed) == len(prior), 'Duplicate saved predictions'
    for condition in cfg['conditions']:
        for item in qa['questions']:
            values, ckpt_hash = values_for(item, condition, cache)
            prompt = h.serialize(item, condition, values)
            key = condition+'/'+item['id']
            if key in completed:
                assert completed[key]['prompt_hash'] == digest(prompt) and completed[key]['checkpoint_hash'] == ckpt_hash
                continue
            response = chat(cfg['llm'], [{'role': 'user', 'content': prompt}])
            try:
                prediction = parse_json(response['text'])['answer']
                scores = h.score(prediction, item)
            except (ValueError, KeyError, TypeError):
                prediction = None
                scores = dict(valid=False, correct=False, set_f1=None if item['spec']['type']==h.TASKS[1] else 0.0,
                              error_type='invalid_json_or_missing_answer')
            append(run_dir/'predictions.jsonl', dict(key=key, condition=condition,
                seed=None if condition.startswith('direct_') else 11, question_id=item['id'],
                context_id=item['context_id'], strata=item['strata'], prompt=prompt,
                prompt_hash=digest(prompt), checkpoint_hash=ckpt_hash, gold=item['gold'],
                prediction=prediction, **scores, **response))
        print('EVALUATED', condition, len(qa['questions']), flush=True)


def report(qa, cfg, run_dir):
    records = rows(run_dir/'predictions.jsonl')
    expected = {c+'/'+q['id'] for c in cfg['conditions'] for q in qa['questions']}
    actual = {r['key'] for r in records}
    gold = {q['id']: q['gold'] for q in qa['questions']}
    verification = dict(expected=len(expected), records=len(records), missing=len(expected-actual),
        unexpected=len(actual-expected), duplicates=len(records)-len(actual),
        gold_mismatches=sum(r['gold'] != gold[r['question_id']] for r in records))
    write(run_dir/'verification.json', verification)
    assert all(verification[k] == 0 for k in ('missing','unexpected','duplicates','gold_mismatches'))
    groups = collections.defaultdict(list)
    for row in records:
        groups[row['condition'], 'all', 'all'].append(row)
        for name in ('task','node_count','answer_size','answer_empty','relation_label','has_multi_parent','ancestor_hops'):
            if row['strata'].get(name) is not None:
                groups[row['condition'], name, str(row['strata'][name])].append(row)
    table = []
    for (condition, stratum, value), group in sorted(groups.items()):
        f1s = [r['set_f1'] for r in group if r['set_f1'] is not None]
        tokens = [r['usage']['prompt_tokens'] for r in group if r.get('usage') and 'prompt_tokens' in r['usage']]
        table.append(dict(condition=condition, stratum=stratum, value=value, n=len(group),
            accuracy=sum(r['correct'] for r in group)/len(group), invalid_rate=sum(not r['valid'] for r in group)/len(group),
            set_f1=float(np.mean(f1s)) if f1s else None,
            mean_prompt_tokens=float(np.mean(tokens)) if tokens else None,
            token_count_coverage=len(tokens)/len(group)))
    with (run_dir/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    baseline = {r['question_id']:r for r in records if r['condition']=='direct_id'}
    comparisons=[]; rng=np.random.default_rng(2026)
    for condition in cfg['conditions']:
        by_graph=collections.defaultdict(list)
        for r in records:
            if r['condition']==condition:
                by_graph[r['context_id']].append(float(r['correct'])-float(baseline[r['question_id']]['correct']))
        delta=np.array([np.mean(v) for v in by_graph.values()])
        interval=np.quantile(rng.choice(delta,(2000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
        comparisons.append(dict(condition=condition,paired_graphs=len(delta),accuracy_delta_vs_direct_id=float(delta.mean()),graph_bootstrap_95_interval=interval))
    q=qa['questions']
    constants={label:sum((item['gold']==label if isinstance(item['gold'],str) else item['gold']==[]) for item in q)/len(q) for label in h.LABELS}
    result=dict(comparisons=comparisons,fixed_response_baselines=constants,
        policy='Constant selected label on relation tasks; [] on set tasks. Four-way candidate random chance is 25%.',
        limitations=['Exploratory development diagnostic; no test performance claim.',
            'Same observed graphs were used to train embeddings; graph reconstruction is not held-out link prediction.',
            'One embedding seed; questions share contexts and source entities. Intervals resample contexts.',
            'All conditions contain complete directed graph facts; raw coordinates increase input length.'])
    write(run_dir/'paired_comparisons.json',result)
    print('VERIFIED',verification,flush=True)
    print('ALL_RESULTS',[r for r in table if r['stratum']=='all'],flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-root',required=True); p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--config',required=True); p.add_argument('--root',required=True)
    p.add_argument('--tokenizer-python',default='/home/work/TOHE/.venv-vllm/bin/python')
    p.add_argument('--stage',choices=('prepare','all','tokens','report'),default='all'); args=p.parse_args()
    root=Path(args.root); cfg=read(args.config); source_root=Path(args.source_root); checkpoint_root=Path(args.checkpoint_root)
    root.mkdir(parents=True,exist_ok=True)
    def status(state,stage,**extra):
        write(root/'job-status.json',dict(status=state,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**extra))
    try:
        if (root/'experiment-config.json').exists(): assert read(root/'experiment-config.json')==cfg
        write(root/'experiment-config.json',cfg)
        if args.stage=='tokens':
            token_budget(read(root/'data/qa.json'),cfg,checkpoint_root,root/'token-budget.json'); return
        if args.stage=='report':
            report(read(root/'data/qa.json'),cfg,root/'evaluation'); return
        status('running','prepare')
        dataset=h.make_specs(read(source_root/'data/specs.json'))
        assert len(dataset['questions'])==198 and dataset['metadata']['contexts']==33
        assert set(dataset['metadata']['task_counts'].values())=={66}
        assert set(dataset['metadata']['relation_label_counts'].values())=={22}
        if (root/'data/specs.json').exists(): assert read(root/'data/specs.json')==dataset
        write(root/'data/specs.json',dataset)
        load_checkpoints(dataset,cfg,checkpoint_root)
        print('FROZEN_SPECIFICATION',dataset['metadata'],flush=True)
        if args.stage=='prepare': status('prepared','prepare'); return
        endpoint=os.environ[cfg['llm']['endpoint_env']]
        with urllib.request.urlopen(endpoint.rsplit('/chat/completions',1)[0]+'/models',timeout=30) as f:
            model=next(m for m in json.load(f)['data'] if m['id']==cfg['llm']['model'])
        assert model['max_model_len']>=16384
        status('running','generate_and_audit',questions=198)
        qa=generate(dataset,cfg,root/'data/qa.json'); check_qa(qa)
        status('running','check_token_budget')
        subprocess.run([args.tokenizer_python,str(Path(__file__).resolve()),'--source-root',str(source_root),
            '--checkpoint-root',str(checkpoint_root),'--config',str(root/'experiment-config.json'),
            '--root',str(root),'--stage','tokens'],check=True)
        status('running','evaluate',expected_predictions=2178)
        evaluate(qa,cfg,checkpoint_root,root/'evaluation')
        status('running','report'); report(qa,cfg,root/'evaluation')
        status('complete','complete',summary_csv=str(root/'evaluation/summary.csv'))
    except Exception as exc:
        status('failed',args.stage,error=str(exc)); raise


if __name__=='__main__':
    main()
