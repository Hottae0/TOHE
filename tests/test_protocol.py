import json
import copy
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
from tohe.evidence import local_graph, evidence_graphs, train_evidence, check_checkpoint, SCOPE
from tohe.questions import canonical_question, canonical_questions
from tohe.audited_questions import generate_audited_questions, review_messages, review_response_format
from tohe.llm import chat


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

    def test_canonical_questions_preserve_evidence_and_gold(self):
        original = make_specs(self.graph, self.config, self.root / "specs.json")
        with patch("tohe.pipeline.chat") as mock:
            data = canonical_questions(original, self.root / "canonical.json")
            mock.assert_not_called()
        for old, new in zip(original["questions"], data["questions"]):
            for field in ("id", "split", "spec", "gold", "graph", "semantic_key"):
                self.assertEqual(old[field], new[field])
            self.assertEqual(new["question_origin"], "canonical_template")
            self.assertIsNone(old["question"])
        wording = canonical_question({"type": "exact_k", "source": "N038", "k": 3})
        self.assertIn("exactly 3 directed", wording)
        self.assertIn("even if a shorter path also exists", wording)
        self.assertNotIn("How many", wording)
        cfg = dict(self.config, conditions=["direct_id"])
        with patch("tohe.pipeline.chat", return_value={"text": '{"answer": []}'}) as mock:
            result = evaluate(data, cfg, self.root / "canonical-eval", self.root / "unused", "dev", 30)
        self.assertEqual(result["new_records"], 6)
        self.assertEqual(mock.call_count, 6)
        data["questions"][0]["question"] = "How many nodes?"
        # Select the modified question's split so provenance is checked before any call.
        with patch("tohe.pipeline.chat") as mock:
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluate(data, cfg, self.root / "bad", self.root / "unused",
                         data["questions"][0]["split"], 30)
            mock.assert_not_called()

    def test_unknown_ids_and_task_specific_rules(self):
        self.assertEqual(score_prediction(["N39"], ["N039"], ["N039"])["error_type"], "unknown_node_id")
        self.assertEqual(score_prediction(False, ["N039"], ["N039"])["error_type"], "answer_type")
        self.assertTrue(score_prediction(["N039"], ["N039"], ["N039"])["correct"])
        self.assertTrue(score_prediction([], [], ["N039"])["correct"])
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        item = next(q for q in data["questions"] if q["spec"]["type"] == "exact_k")
        text = serialize(item, "direct_id")
        self.assertIn("including leading zeros", text)
        self.assertIn("not shortest-path distance", text)
        self.assertIn('Return {"answer": []}', text)

    def test_audited_generation_rejects_count_and_wrong_semantics(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        item = next(q for q in data["questions"] if q["spec"]["type"] == "exact_k")
        data["questions"] = [item]
        spec = item["spec"]
        valid_question = canonical_question(spec)
        review = dict(interpreted_spec=spec, answer_kind="node_set", edge_direction="outgoing",
                      path_semantics="exact_length", ambiguous=False)
        responses = [
            {"text": json.dumps({"question": f"How many nodes from {spec['source']}?", "spec": spec})},
            {"text": json.dumps({"question": valid_question, "spec": spec})},
            {"text": json.dumps(dict(review, path_semantics="shortest_distance"))},
            {"text": json.dumps({"question": valid_question, "spec": spec})},
            {"text": json.dumps(review)}]
        output = self.root / "audited.json"
        with patch("tohe.audited_questions.chat", side_effect=responses) as mock:
            result = generate_audited_questions(data, self.config, output, 5)
        self.assertEqual(mock.call_count, 5)
        self.assertEqual(result["metadata"]["generation_status"], "complete")
        self.assertEqual(result["questions"][0]["semantic_review"], "model_verified_v3")
        self.assertEqual(result["questions"][0]["gold"], item["gold"])
        self.assertEqual(len(rows(output.with_suffix(".generation-failures.jsonl"))), 2)
        review_request = mock.call_args_list[-1].args[1]
        self.assertEqual(review_request[-1]["content"], valid_question)
        self.assertNotIn(spec["source"], json.dumps(review_request[:-1]))
        cfg = dict(self.config, conditions=["direct_id"])
        with patch("tohe.pipeline.chat", return_value={"text": '{"answer": []}'}) as mock:
            evaluate(result, cfg, self.root / "eval", self.root / "unused", item["split"], 1)
            self.assertEqual(mock.call_count, 1)

    def test_audited_candidate_resume_and_fail_closed(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        data["questions"] = data["questions"][:1]
        item = data["questions"][0]
        spec = item["spec"]
        response = {"text": json.dumps(dict(question=canonical_question(spec), spec=spec))}
        review = {"text": json.dumps(dict(interpreted_spec=spec, answer_kind="boolean",
                    edge_direction="outgoing", path_semantics="nonempty_path", ambiguous=False))}
        output = self.root / "audited.json"
        with patch("tohe.audited_questions.chat", return_value=response) as mock:
            partial = generate_audited_questions(data, self.config, output, 1)
            self.assertEqual(mock.call_count, 1)
        self.assertIn("generation_candidate", partial["questions"][0])
        cfg = dict(self.config, conditions=["direct_id"])
        with patch("tohe.pipeline.chat") as mock:
            with self.assertRaises(ValueError):
                evaluate(partial, cfg, self.root / "pending", self.root / "unused", item["split"], 1)
            mock.assert_not_called()
        with patch("tohe.audited_questions.chat", return_value=review) as mock:
            complete = generate_audited_questions(data, self.config, output, 1)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mock.call_args.args[1][-1]["content"], canonical_question(spec))
            generate_audited_questions(data, self.config, output, 1)
            self.assertEqual(mock.call_count, 1)
        self.assertEqual(complete["metadata"]["generation_status"], "complete")
        with patch("tohe.audited_questions.chat", return_value={"text": "{}"}) as mock:
            with self.assertRaisesRegex(ValueError, "failed 3 attempts"):
                generate_audited_questions(data, self.config, self.root / "bad-audit.json", 20)
            self.assertEqual(mock.call_count, 3)

    def test_reviewer_format_failure_reuses_question_and_forces_shape(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        data["questions"] = data["questions"][:1]
        spec = data["questions"][0]["spec"]
        valid = dict(interpreted_spec=spec, answer_kind="boolean", edge_direction="outgoing",
                     path_semantics="nonempty_path", ambiguous=False)
        responses = [
            {"text": json.dumps(dict(question=canonical_question(spec), spec=spec))},
            {"text": json.dumps(dict(valid, interpreted_spec="Find nodes that are reachable"))},
            {"text": json.dumps(valid)}]
        output = self.root / "structured-audit.json"
        with patch("tohe.audited_questions.chat", side_effect=responses) as mock:
            result = generate_audited_questions(data, self.config, output, 3)
        self.assertEqual(mock.call_count, 3)  # One writer, two reviewers.
        for call in mock.call_args_list[1:]:
            request_cfg, messages = call.args
            self.assertEqual(request_cfg["response_format"]["type"], "json_schema")
            self.assertEqual(messages[-1]["content"], canonical_question(spec))
        self.assertIn("previous review format error", mock.call_args.args[1][0]["content"])
        self.assertEqual(result["metadata"]["generation_status"], "complete")
        self.assertEqual(rows(output.with_suffix(".generation-failures.jsonl"))[0]["error_stage"], "review_format")
        schema_text = json.dumps(review_response_format())
        self.assertNotIn(spec["source"], schema_text)
        self.assertIn('"count"', schema_text)  # Allow a negative verdict; don't force acceptance.

    def test_common_parents_reviewer_regression_preserves_valid_candidate(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        item = next(q for q in data["questions"] if q["spec"]["type"] == "common_parents")
        data["questions"] = [item]
        spec = item["spec"]
        valid = dict(interpreted_spec=spec, answer_kind="node_set", edge_direction="outgoing",
                     path_semantics="direct_neighbors", ambiguous=False)
        invalid = dict(valid, interpreted_spec=dict(type="reachability", source=spec["left"], target="P"))
        response = {"text": json.dumps(dict(question=canonical_question(spec), spec=spec))}
        output = self.root / "common-regression.json"
        with patch("tohe.audited_questions.chat", side_effect=[response, {"text": json.dumps(invalid)}, {"text": json.dumps(valid)}]) as mock:
            result = generate_audited_questions(data, self.config, output, 3)
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(result["questions"][0]["question"], canonical_question(spec))
        log = rows(output.with_suffix(".generation-failures.jsonl"))
        self.assertEqual(log[0]["error_stage"], "review_format")
        messages = review_messages(canonical_question(spec))
        examples = [json.loads(m["content"])["interpreted_spec"]["type"]
                    for m in messages if m["role"] == "assistant"]
        self.assertEqual(set(examples), {"reachability", "common_parents", "exact_k"})

    def test_chat_transmits_structured_response_format(self):
        cfg = dict(self.config["generation_llm"], model="test-model", endpoint_env="TOHE_TEST_ENDPOINT",
                   response_format=review_response_format())
        response = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        with patch.dict("os.environ", {"TOHE_TEST_ENDPOINT": "http://localhost:8000/v1/chat/completions"}):
            with patch("tohe.llm.urllib.request.urlopen") as mock:
                mock.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
                result = chat(cfg, [{"role": "user", "content": "Test"}])
        body = json.loads(mock.call_args.args[0].data)
        self.assertEqual(body["response_format"], review_response_format())
        self.assertEqual(result["finish_reason"], "stop")

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

    def test_generation_repairs_invalid_spec_without_changing_gold(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        item = next(q for q in data["questions"] if q["spec"]["type"] == "common_parents")
        data["questions"] = [item]
        wrong = {**item["spec"], "extra": True}
        responses = [
            {"text": json.dumps({"question": "Which parents?", "spec": wrong})},
            {"text": "not JSON"},
            {"text": json.dumps({"question": "Which direct parents are shared?", "spec": item["spec"]})}]
        output = self.root / "generated.json"
        with patch("tohe.pipeline.chat", side_effect=responses) as mock:
            result = generate_questions(data, self.config, output, 3)
        self.assertEqual(mock.call_count, 3)
        correction = mock.call_args.args[1][-1]["content"]
        self.assertIn(json.dumps(item["spec"]), correction)
        self.assertEqual(result["questions"][0]["spec"], item["spec"])
        self.assertEqual(result["questions"][0]["gold"], item["gold"])
        self.assertEqual(result["questions"][0]["generation"]["attempts"], 3)
        failures = rows(output.with_suffix(".generation-failures.jsonl"))
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0]["received_spec"], wrong)
        self.assertEqual(failures[1]["response"]["text"], "not JSON")

    def test_generation_retry_budget_exhaustion_and_resume(self):
        data = make_specs(self.graph, self.config, self.root / "specs.json")
        data["questions"] = data["questions"][:1]
        output = self.root / "generated.json"
        with patch("tohe.pipeline.chat", return_value={"text": "{}"}) as mock:
            partial = generate_questions(data, self.config, output, 1)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(partial["questions"][0]["question_origin"], "awaiting_llm_generation")
            with self.assertRaisesRegex(ValueError, "failed validation after 3 attempts"):
                generate_questions(data, self.config, output, 20)
            self.assertEqual(mock.call_count, 4)
        spec = data["questions"][0]["spec"]
        with patch("tohe.pipeline.chat", return_value={"text": json.dumps({"question": "Which nodes?", "spec": spec})}):
            complete = generate_questions(data, self.config, output, 1)
        self.assertEqual(complete["metadata"]["generation_status"], "complete")

    def test_four_trainers_and_seven_condition_dry_run(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        results = train_evidence(data, self.config, self.root / "checkpoints")
        self.assertEqual(len(results), 8)  # two contexts, not twelve questions
        for output in (self.root / "checkpoints").glob("*/*.pt"):
            ckpt = torch.load(output, weights_only=True)
            self.assertTrue(torch.isfinite(ckpt["embeddings"]).all())
            self.assertEqual(ckpt["embedding_scope"], SCOPE)
            self.assertEqual(len(ckpt["nodes"]), 8)
            if ckpt["variant"] != "euclidean":
                self.assertLess(ckpt["embeddings"].norm(dim=1).max().item(), 1)
        changed = copy.deepcopy(data)
        for q in changed["questions"]:
            q["gold"] = "not accessed by training"
            q["question"] = "not accessed by training"
        changed["graph_hash"] = "different global source"
        with patch("tohe.model.train", side_effect=AssertionError("Should reuse local embeddings")):
            train_evidence(changed, self.config, self.root / "checkpoints")
        result = evaluate(data, self.config, self.root / "eval", self.root / "checkpoints", "test", 1000, True)
        self.assertEqual(result["new_records"], 42)
        self.assertEqual(len(rows(self.root / "eval/prompts.jsonl")), 42)
        again = evaluate(data, self.config, self.root / "eval", self.root / "checkpoints", "test", 1000, True)
        self.assertEqual(again["new_records"], 0)
        with self.assertRaises(ValueError):
            evaluate(data, self.config, self.root / "real", self.root / "checkpoints", "test", 1, False)
        test_context = next(q["context_id"] for q in data["questions"] if q["split"] == "test")
        path = self.root / "checkpoints" / test_context / "v2-11.pt"
        bad = torch.load(path, weights_only=True)
        bad["embedding_scope"] = "global_mammals_reconstruction"
        torch.save(bad, path)
        with patch("tohe.pipeline.chat") as api:
            with self.assertRaisesRegex(ValueError, "Global/legacy"):
                evaluate(data, self.config, self.root / "bad", self.root / "checkpoints", "test", 1)
            api.assert_not_called()

    def test_local_closure_never_imports_outside_paths(self):
        context = {"nodes": ["A", "B", "C"], "original_ids": {v: v for v in "ABC"},
                   "edges": [["A", "C"]], "closure": [["A", "B"]], "evidence_hash": "stale"}
        # Full graph may contain A -> X -> B, but X is not visible.
        global_pairs = closure(["A", "B", "C", "X"], [("A", "X"), ("X", "B"), ("A", "C")])
        self.assertIn(("A", "B"), global_pairs)
        local = local_graph(context)
        self.assertEqual(local["closure"], [("A", "C")])
        altered = dict(context, closure=global_pairs, evidence_hash="another stale hash")
        self.assertEqual(local, local_graph(altered))
        altered = copy.deepcopy(context)
        altered["edges"].append(["C", "B"])
        self.assertNotEqual(local["graph_hash"], local_graph(altered)["graph_hash"])

    def test_conflicting_evidence_ids_fail(self):
        data = make_specs(self.graph, self.config, self.root / "qa.json")
        data["questions"][1]["graph"] = copy.deepcopy(data["questions"][1]["graph"])
        data["questions"][1]["graph"]["edges"].pop()
        with self.assertRaisesRegex(ValueError, "One context ID"):
            evidence_graphs(data)

    def test_no_external_negatives_and_checkpoint_scope_guard(self):
        graph = local_graph({"nodes": ["A", "B"], "edges": [["A", "B"]],
                             "original_ids": {"A": "A", "B": "B"}})
        for variant in ("euclidean", "poincare", "v1", "v2"):
            ckpt = train(graph, self.config, variant, 11, self.root / f"{variant}.pt")
            self.assertEqual(ckpt["nodes"], ["A", "B"])
            self.assertEqual(ckpt["sampling"]["no_negative_pairs"], 1)
            self.assertTrue(torch.isfinite(ckpt["embeddings"]).all())
            check_checkpoint(ckpt, graph, self.config, variant, 11)
        bad = dict(ckpt, embedding_scope="global_mammals_reconstruction")
        with self.assertRaisesRegex(ValueError, "Global/legacy"):
            check_checkpoint(bad, graph, self.config, "v2", 11)
        wrong = local_graph({"nodes": ["A", "B"], "edges": [["B", "A"]],
                             "original_ids": {"A": "A", "B": "B"}})
        with self.assertRaisesRegex(ValueError, "exact visible evidence"):
            check_checkpoint(ckpt, wrong, self.config, "v2", 11)

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
        data = canonical_questions(data, self.root / "canonical.json")
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
