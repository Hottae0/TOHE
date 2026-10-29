import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tohe.graph import answer, closure
from tohe.io import read, write, rows
from tohe.model import FixedSerializerGeometry, distance, train, embedding_metrics
from tohe.pipeline import score_prediction, generate_questions, evaluate
from tohe.report import report
from tohe.serialize import serialize
from tohe.data import synthetic
from tohe.qa import make_specs, semantic_key


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = read(Path(__file__).parents[1] / "configs/smoke.json")
        self.graph = synthetic(self.root / "graph.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_graph_semantics_multiple_parents_and_exact_length(self):
        graph = {"nodes": ["a", "b", "c", "d"], "edges": [("a", "b"), ("a", "c"), ("b", "c"), ("d", "b"), ("d", "c")]}
        self.assertEqual(answer(graph, {"type": "common_parents", "left": "a", "right": "d"}), ["b", "c"])
        self.assertEqual(answer(graph, {"type": "exact_k", "source": "a", "k": 2}), ["c"])
        self.assertTrue(answer(graph, {"type": "reachability", "source": "a", "target": "c"}))
        self.assertFalse(answer(graph, {"type": "reachability", "source": "c", "target": "a"}))
        with self.assertRaises(ValueError):
            closure(["a", "b"], [("a", "b"), ("b", "a")])

    def test_fixed_code_probabilities_distance_and_gradients(self):
        z = torch.tensor([[0.01, 0.02], [0.2, -0.1]], dtype=torch.float64, requires_grad=True)
        geometry = FixedSerializerGeometry(self.config["serializer"], 2)
        _, _, _, joint = geometry.soft(z, 0.25)
        self.assertTrue(torch.allclose(joint.sum(-1), torch.ones(2, dtype=z.dtype)))
        self.assertTrue(torch.allclose(distance(z[0], z[1]), distance(z[1], z[0])))
        (distance(z[0], z[1]) + joint.square().sum()).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.equal(geometry.table.diag(), torch.zeros(len(geometry.table), dtype=z.dtype)))

    def test_all_serializers_preserve_all_edges_and_no_gold(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        item = data["questions"][0]
        item["question"] = "Test question"
        codes = {node: [0, 0] for node in self.graph["nodes"]}
        expected = {f"{u} --is_a--> {v}" for u, v in item["graph"]["edges"]}
        for condition in self.config["conditions"]:
            text = serialize(item, condition, codes)
            actual = {line for line in text.splitlines() if " --is_a--> " in line and not line.startswith("-")}
            self.assertEqual(actual, expected)
            self.assertNotIn('"gold"', text)
            self.assertNotIn("original_ids", text)

    def test_task_balance_unique_contexts_and_provenance(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        self.assertEqual(len(data["questions"]), 12)
        self.assertEqual(len({q["semantic_key"] for q in data["questions"]}), 12)
        for item in data["questions"]:
            self.assertEqual(item["gold"], answer(item["graph"], item["spec"]))
            self.assertIsNone(item["question"])
            self.assertEqual(item["question_origin"], "awaiting_llm_generation")
        dev = {q["graph"]["evidence_hash"] for q in data["questions"] if q["split"] == "dev"}
        test = {q["graph"]["evidence_hash"] for q in data["questions"] if q["split"] == "test"}
        self.assertFalse(dev & test)

    def test_semantic_duplicates_ignore_aliases_and_pair_order(self):
        first = {"original_ids": {"N0": "dog", "N1": "cat"}}
        second = {"original_ids": {"X": "cat", "Y": "dog"}}
        a = {"type": "common_parents", "left": "N0", "right": "N1"}
        b = {"type": "common_parents", "left": "X", "right": "Y"}
        self.assertEqual(semantic_key(first, a), semantic_key(second, b))

    def test_generation_resume_and_scoring_strict_types(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        def fake_chat(config, messages):
            payload = json.loads(messages[0]["content"])
            return {"text": json.dumps({"question": "What is the answer to the specified task?", "spec": payload["spec"]}), "usage": None}
        with patch("tohe.pipeline.chat", side_effect=fake_chat) as mock:
            generated = generate_questions(data, self.config, self.root / "generated.json", 2)
            self.assertEqual(mock.call_count, 2)
            generated = generate_questions(data, self.config, self.root / "generated.json", 20)
            self.assertEqual(mock.call_count, 12)
        self.assertEqual(generated["metadata"]["generation_status"], "complete")
        self.assertFalse(score_prediction(1, True)["valid"])
        self.assertTrue(score_prediction(["b", "a"], ["a", "b"])["correct"])
        self.assertFalse(score_prediction(["a", "unknown"], ["a"])["correct"])

    def test_four_trainers_and_seven_condition_dry_run(self):
        for variant in ["euclidean", "poincare", "v1", "v2"]:
            output = self.root / "checkpoints" / f"{variant}-11.pt"
            ckpt = train(self.graph, self.config, variant, 11, output)
            self.assertTrue(torch.isfinite(ckpt["embeddings"]).all())
            if variant != "euclidean":
                self.assertLess(ckpt["embeddings"].norm(dim=1).max().item(), 1)
            self.assertGreaterEqual(embedding_metrics(self.graph, ckpt)["map"], 0)
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        result = evaluate(data, self.config, self.root / "eval", self.root / "checkpoints", "test", 1000, True)
        self.assertEqual(result["new_records"], 42)
        self.assertEqual(len(rows(self.root / "eval/prompts.jsonl")), 42)
        again = evaluate(data, self.config, self.root / "eval", self.root / "checkpoints", "test", 1000, True)
        self.assertEqual(again["new_records"], 0)
        with self.assertRaises(ValueError):
            evaluate(data, self.config, self.root / "real", self.root / "checkpoints", "test", 1, False)

    def test_epoch_checkpoint_resume_is_exact_on_cpu(self):
        path = self.root / "resume.pt"
        with patch("builtins.print", side_effect=RuntimeError("simulated interruption")):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train(self.graph, self.config, "v2", 11, path)
        self.assertTrue(path.with_suffix(".partial.pt").exists())
        resumed = train(self.graph, self.config, "v2", 11, path)
        uninterrupted = train(self.graph, self.config, "v2", 11, self.root / "full.pt")
        self.assertTrue(torch.equal(resumed["embeddings"], uninterrupted["embeddings"]))
        self.assertEqual(resumed["history"], uninterrupted["history"])

    def test_evaluation_resume_and_report_with_mock_predictions(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        for q in data["questions"]:
            q.update(question="Test-only mock wording", question_origin="llm_generated")
        write(self.root / "qa.json", data)
        cfg = dict(self.config, conditions=["direct_id", "direct_bfs", "direct_dfs"])
        response = {"text": '{"answer": false}', "usage": {"prompt_tokens": 100}, "seconds": 0.1}
        with patch("tohe.pipeline.chat", return_value=response) as mock:
            evaluate(data, cfg, self.root / "eval", self.root / "none", "test", 2)
            self.assertEqual(mock.call_count, 2)
            evaluate(data, cfg, self.root / "eval", self.root / "none", "test", 100)
            self.assertEqual(mock.call_count, 18)
            evaluate(data, cfg, self.root / "eval", self.root / "none", "test", 100)
            self.assertEqual(mock.call_count, 18)
        result = report(self.root / "eval", self.root / "qa.json")
        self.assertTrue(all(row["complete"] for row in result["comparisons"]))
        self.assertTrue((self.root / "eval/summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
