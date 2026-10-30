import collections
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tohe import hierarchy_qa as h


class HierarchyDiagnosticTests(unittest.TestCase):
    def graph(self):
        nodes = [f'N{i:03}' for i in range(20)]
        return dict(nodes=nodes, edges=[(nodes[i],nodes[(i-1)//2]) for i in range(1,20)],
                    original_ids={n:n for n in nodes})

    def test_relation_direction_and_unrelated(self):
        graph=self.graph()
        cases=[('N003','N000','subtype'),('N000','N003','supertype'),('N003','N006','unrelated')]
        for left,right,label in cases:
            self.assertEqual(h.gold_answer(graph,dict(type=h.TASKS[1],left=left,right=right)),label)

    def test_lca_all_minimal_elements_and_reflexive_convention(self):
        graph=dict(nodes=['N000','N001','N002','N003','N004'],
            edges=[('N003','N001'),('N003','N002'),('N004','N001'),('N004','N002'),
                   ('N001','N000'),('N002','N000')])
        self.assertEqual(h.gold_answer(graph,dict(type=h.TASKS[2],left='N003',right='N004')),['N001','N002'])
        self.assertEqual(h.gold_answer(graph,dict(type=h.TASKS[2],left='N003',right='N001')),['N001'])
        graph['nodes'].append('N005')
        self.assertEqual(h.gold_answer(graph,dict(type=h.TASKS[2],left='N003',right='N005')),[])

    def test_frozen_design_and_candidate_balance(self):
        source={'questions':[dict(context_id=f'g{i:04}',split='dev',graph=self.graph(),
                   serialization_traversal='weak',spec=dict(type='reachability',source='N019',target='N000'))
                   for i in range(33)]}
        for i, q in enumerate(source['questions']):
            q['graph']['original_ids']={n:f'original-{i}-{n}' for n in q['graph']['nodes']}
        source['questions'].append(dict(context_id='test-only',split='test',graph=self.graph(),spec=dict(type='reachability',source='N019',target='N000')))
        a=h.make_specs(source); b=h.make_specs(source)
        self.assertEqual(a,b)
        self.assertEqual(a['metadata']['contexts'],33)
        self.assertEqual(a['metadata']['questions'],198)
        self.assertEqual(a['metadata']['task_counts'],dict(zip(h.TASKS,[66]*3)))
        self.assertEqual(a['metadata']['relation_label_counts'],dict(zip(h.LABELS,[22]*3)))
        positions=collections.Counter()
        for q in a['questions']:
            self.assertEqual(q['gold'],h.gold_answer(q['graph'],q['spec']))
            self.assertIsNone(h.wording_error(h.canonical_question(q['spec']),q['spec']))
            if q['spec']['type']==h.TASKS[0]:
                self.assertEqual(len(q['gold']),1)
                self.assertEqual(len(set(q['spec']['candidates'])),4)
                positions[q['spec']['candidates'].index(q['gold'][0])]+=1
        self.assertEqual(sorted(positions.values()),[16,16,17,17])

    def test_scoring_distinguishes_labels_ids_and_candidate_constraints(self):
        graph=self.graph()
        item=dict(graph=graph,spec=dict(type=h.TASKS[1],left='N003',right='N000'),gold='subtype')
        self.assertTrue(h.score('subtype',item)['correct'])
        self.assertFalse(h.score(True,item)['valid'])
        self.assertFalse(h.score('supertype',item)['correct'])
        item.update(spec=dict(type=h.TASKS[0],source='N003',candidates=['N000','N002','N005','N006']),gold=['N000'])
        self.assertTrue(h.score(['N000'],item)['correct'])
        self.assertFalse(h.score(['N001'],item)['valid'])
        self.assertFalse(h.score(['N000','N000'],item)['valid'])

    def test_all_conditions_preserve_the_same_directed_facts(self):
        graph=self.graph(); spec=dict(type=h.TASKS[1],left='N003',right='N000')
        item=dict(graph=graph,spec=spec,question=h.canonical_question(spec),serialization_traversal='weak')
        codes={n:[1,2] for n in graph['nodes']}; raw={n:[0.12345678901234567,0.2] for n in graph['nodes']}
        for condition in ['direct_id','direct_bfs','direct_dfs','euclidean','poincare','v1','v2','raw_euclidean','raw_poincare','raw_v1','raw_v2']:
            values=raw if condition.startswith('raw_') else codes
            prompt=h.serialize(item,condition,values)
            self.assertEqual(len([line for line in prompt.splitlines() if line.startswith('N') and ' --is_a--> ' in line]),len(graph['edges']))
            for child,parent in graph['edges']:
                self.assertIn(f'{child} --is_a--> {parent}',prompt)
            self.assertNotIn('answer must be a JSON boolean',prompt)
            if condition.startswith('raw_'):
                self.assertIn('0.12345678901234566',prompt)

    def test_semantic_reviewer_cannot_swap_relation_direction(self):
        spec=dict(type=h.TASKS[1],left='N003',right='N000')
        review=dict(interpreted_spec=spec,answer_kind='relation_label',edge_direction='outgoing_to_supertypes',
                    semantics='left_relative_to_right',ambiguous=False)
        self.assertIsNone(h.review_error(review,spec))
        review['interpreted_spec']=dict(type=h.TASKS[1],left='N000',right='N003')
        self.assertIsNotNone(h.review_error(review,spec))

    def test_evaluation_resumes_and_reports_all_eleven_conditions(self):
        import torch
        from tohe.io import digest, read, rows
        script=Path(__file__).resolve().parents[1]/'scripts/run_hierarchy_diagnostic.py'
        module_spec=importlib.util.spec_from_file_location('hierarchy_runner_test',script)
        runner=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(runner)
        source={'questions':[dict(context_id='g0',split='dev',graph=self.graph(),spec=dict(type='reachability',source='N019',target='N000'))]}
        qa=h.make_specs(source); qa['metadata']['generation_status']='complete'
        for q in qa['questions']:
            q['question']=h.canonical_question(q['spec']); q['semantic_review']='model_verified_hierarchy_v1'
            sem=dict(zip(h.TASKS,('strict_candidate_ancestor','left_relative_to_right','reflexive_minimal_common_ancestors')))[q['spec']['type']]
            audit=dict(interpreted_spec=q['spec'],answer_kind='relation_label' if q['spec']['type']==h.TASKS[1] else 'node_set',edge_direction='outgoing_to_supertypes',semantics=sem,ambiguous=False)
            q['semantic_audit']=dict(text=json.dumps(audit),question_hash=digest(q['question']))
        conditions=['direct_id','direct_bfs','direct_dfs','euclidean','poincare','v1','v2','raw_euclidean','raw_poincare','raw_v1','raw_v2']
        cfg=dict(conditions=conditions,llm=dict(endpoint_env='HIERARCHY_TEST_ENDPOINT'))
        nodes=qa['questions'][0]['graph']['nodes']
        ckpt=dict(nodes=nodes,codes=torch.zeros((len(nodes),2),dtype=torch.int64),embeddings=torch.zeros((len(nodes),5),dtype=torch.float64))
        cache={('g0',v):(ckpt,'fixed-checkpoint-'+v) for v in ('euclidean','poincare','v1','v2')}
        responses=[dict(text=json.dumps({'answer':q['gold']}),usage={'prompt_tokens':123}) for c in conditions for q in qa['questions']]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,{'HIERARCHY_TEST_ENDPOINT':'http://unused.local'}), mock.patch.object(runner,'load_checkpoints',return_value=cache), mock.patch.object(runner,'embedding_diagnostics'), mock.patch.object(runner,'chat',side_effect=responses) as llm, contextlib.redirect_stdout(io.StringIO()):
            run=Path(tmp)
            runner.evaluate(qa,cfg,run,run)
            self.assertEqual(llm.call_count,66)
            runner.evaluate(qa,cfg,run,run)
            self.assertEqual(llm.call_count,66)
            self.assertEqual(len(rows(run/'predictions.jsonl')),66)
            runner.report(qa,cfg,run)
            self.assertEqual(read(run/'verification.json')['missing'],0)
            self.assertEqual(read(run/'verification.json')['duplicates'],0)
            self.assertTrue((run/'summary.csv').exists())


if __name__=='__main__': unittest.main()
