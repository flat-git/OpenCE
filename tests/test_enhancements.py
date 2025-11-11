"""Tests for new ACE enhancement features: gating, diagnostics, and reporting."""

import json
import unittest
from pathlib import Path
import tempfile

from ace import (
    BulletGate,
    CurationRules,
    Diagnostics,
    DummyLLMClient,
    ElementExtractor,
    ElementMatch,
    EnvironmentResult,
    GatingConfig,
    Generator,
    OfflineAdapter,
    Playbook,
    PlaybookReporter,
    Reflector,
    Curator,
    Sample,
    TaskEnvironment,
)


class DiagnosticsTest(unittest.TestCase):
    """Test element extraction and diagnostics."""

    def test_element_extractor(self):
        extractor = ElementExtractor()
        
        # Test with provided elements
        elements = extractor.extract_elements(
            generator_output_elements=["element1", "element2"]
        )
        self.assertEqual(elements, ["element1", "element2"])
        
        # Test with question parsing
        question = "A device comprising: a sensor; a controller; and a display."
        elements = extractor.extract_elements(question=question)
        self.assertGreater(len(elements), 0)
    
    def test_element_matching(self):
        extractor = ElementExtractor()
        
        # Test explicit match
        match = extractor.match_element(
            "temperature sensor",
            "The device includes a temperature sensor for monitoring.",
            is_core=True
        )
        self.assertEqual(match.match, "explicit")
        self.assertEqual(match.confidence, 1.0)
        self.assertIsNotNone(match.evidence)
        
        # Test functional match
        match = extractor.match_element(
            "temperature sensor",
            "The device has a thermal sensing device.",
            is_core=True
        )
        self.assertEqual(match.match, "functional")
        self.assertGreater(match.confidence, 0)
        
        # Test no match
        match = extractor.match_element(
            "temperature sensor",
            "The device has a pressure gauge.",
            is_core=True
        )
        self.assertEqual(match.match, "none")
        self.assertEqual(match.confidence, 0.0)
    
    def test_coverage_computation(self):
        extractor = ElementExtractor()
        
        elements = [
            ElementMatch("E1", "sensor", True, "explicit", 1.0),
            ElementMatch("E2", "controller", True, "explicit", 1.0),
            ElementMatch("E3", "display", False, "none", 0.0),
        ]
        
        coverage = extractor.compute_coverage(elements, core_weight=3.0, non_core_weight=1.0)
        # (3*1.0 + 3*1.0 + 1*0.0) / (3 + 3 + 1) = 6/7 ≈ 0.857
        self.assertAlmostEqual(coverage, 6/7, places=2)
    
    def test_diagnostics_serialization(self):
        diagnostics = Diagnostics(
            elements=[
                ElementMatch("E1", "sensor", True, "explicit", 1.0, "evidence text")
            ],
            coverage=0.8,
            decision_basis_bullets=["bullet-001", "bullet-002"]
        )
        
        # Test to_dict
        data = diagnostics.to_dict()
        self.assertEqual(len(data["elements"]), 1)
        self.assertEqual(data["coverage"], 0.8)
        
        # Test from_dict
        restored = Diagnostics.from_dict(data)
        self.assertEqual(len(restored.elements), 1)
        self.assertEqual(restored.coverage, 0.8)
        self.assertEqual(restored.elements[0].id, "E1")


class GatingTest(unittest.TestCase):
    """Test bullet gating functionality."""

    def test_gating_without_embeddings(self):
        # Test fallback when embeddings not available
        config = GatingConfig()
        gate = BulletGate(config)
        gate._encoder = None  # Simulate no encoder
        
        bullets = [
            ("b1", "content 1"),
            ("b2", "content 2"),
            ("b3", "content 3"),
        ]
        
        selected = gate.select_bullets("sample text", bullets)
        # Should return all bullets when no encoder
        self.assertEqual(len(selected), 3)
    
    def test_gating_config(self):
        config = GatingConfig(top_k=10, guard_strategies=3)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.guard_strategies, 3)
    
    def test_similarity_computation_without_encoder(self):
        config = GatingConfig()
        gate = BulletGate(config)
        gate._encoder = None
        
        similarity = gate.compute_similarity("text1", "text2")
        self.assertEqual(similarity, 0.0)


class CurationRulesTest(unittest.TestCase):
    """Test curation rule validation."""

    def test_add_operation_limit(self):
        rules = CurationRules(max_add_per_iteration=2)
        playbook = Playbook()
        
        from ace import DeltaOperation
        operations = [
            DeltaOperation(type="ADD", section="test", content="content1"),
            DeltaOperation(type="ADD", section="test", content="content2"),
            DeltaOperation(type="ADD", section="test", content="content3"),
        ]
        
        # Mock similarity check to pass
        rules.bullet_gate._encoder = None  # Disable encoder
        
        validated = rules.validate_operations(operations, playbook)
        # Should limit to max_add_per_iteration
        add_count = sum(1 for op in validated if op.type == "ADD")
        self.assertLessEqual(add_count, 2)
    
    def test_update_operation_priority(self):
        rules = CurationRules()
        playbook = Playbook()
        playbook.add_bullet("test", "existing content", bullet_id="b1")
        
        from ace import DeltaOperation
        operations = [
            DeltaOperation(type="UPDATE", section="test", content="updated", bullet_id="b1"),
            DeltaOperation(type="ADD", section="test", content="new content"),
        ]
        
        validated = rules.validate_operations(operations, playbook)
        # UPDATE should be included
        self.assertTrue(any(op.type == "UPDATE" for op in validated))
    
    def test_deprecate_to_remove_conversion(self):
        rules = CurationRules()
        playbook = Playbook()
        playbook.add_bullet("test", "content", bullet_id="b1")
        
        from ace import DeltaOperation
        operations = [
            DeltaOperation(type="DEPRECATE", section="test", bullet_id="b1"),
        ]
        
        validated = rules.validate_operations(operations, playbook)
        # DEPRECATE should be converted to REMOVE
        self.assertEqual(validated[0].type, "REMOVE")


class ReportingTest(unittest.TestCase):
    """Test playbook reporting functionality."""

    def test_top_positive_contributors(self):
        playbook = Playbook()
        playbook.add_bullet("test", "content1", bullet_id="b1", metadata={"helpful": 5, "harmful": 1})
        playbook.add_bullet("test", "content2", bullet_id="b2", metadata={"helpful": 3, "harmful": 0})
        playbook.add_bullet("test", "content3", bullet_id="b3", metadata={"helpful": 1, "harmful": 3})
        
        reporter = PlaybookReporter(playbook)
        top = reporter.top_positive_contributors(top_n=2)
        
        self.assertEqual(len(top), 2)
        # Should be sorted by (helpful - harmful)
        self.assertEqual(top[0].id, "b1")  # 5-1=4
        self.assertEqual(top[1].id, "b2")  # 3-0=3
    
    def test_top_negative_contributors(self):
        playbook = Playbook()
        playbook.add_bullet("test", "content1", bullet_id="b1", metadata={"helpful": 1, "harmful": 5})
        playbook.add_bullet("test", "content2", bullet_id="b2", metadata={"helpful": 0, "harmful": 3})
        playbook.add_bullet("test", "content3", bullet_id="b3", metadata={"helpful": 3, "harmful": 1})
        
        reporter = PlaybookReporter(playbook)
        top = reporter.top_negative_contributors(top_n=2)
        
        self.assertEqual(len(top), 2)
        # Should be sorted by (harmful - helpful)
        self.assertEqual(top[0].id, "b1")  # 5-1=4
        self.assertEqual(top[1].id, "b2")  # 3-0=3
    
    def test_deprecation_candidates(self):
        playbook = Playbook()
        playbook.add_bullet("test", "content1", bullet_id="b1", metadata={"helpful": 1, "harmful": 5})
        playbook.add_bullet("test", "content2", bullet_id="b2", metadata={"helpful": 2, "harmful": 1})
        playbook.add_bullet("test", "content3", bullet_id="b3", metadata={"helpful": 0, "harmful": 1})
        
        reporter = PlaybookReporter(playbook)
        candidates = reporter.deprecation_candidates(min_harmful_ratio=0.6, min_total_uses=3)
        
        # Only b1 meets criteria: harmful/(helpful+harmful) = 5/6 ≈ 0.83 >= 0.6, uses=6 >= 3
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].id, "b1")
    
    def test_export_markdown(self):
        playbook = Playbook()
        playbook.add_bullet("test", "content1", bullet_id="b1", metadata={"helpful": 5, "harmful": 1})
        
        reporter = PlaybookReporter(playbook)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            reporter.export_markdown(str(filepath), top_n=1)
            
            self.assertTrue(filepath.exists())
            content = filepath.read_text()
            self.assertIn("# Playbook Performance Report", content)
            self.assertIn("Top 1 Positive Contributors", content)
    
    def test_export_csv(self):
        playbook = Playbook()
        playbook.add_bullet("test", "content1", bullet_id="b1", metadata={"helpful": 5, "harmful": 1})
        
        reporter = PlaybookReporter(playbook)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report"
            reporter.export_csv(str(filepath), top_n=1)
            
            positive_file = Path(tmpdir) / "report_positive.csv"
            negative_file = Path(tmpdir) / "report_negative.csv"
            deprecate_file = Path(tmpdir) / "report_deprecate.csv"
            
            self.assertTrue(positive_file.exists())
            self.assertTrue(negative_file.exists())
            self.assertTrue(deprecate_file.exists())


class ThreeClassDecisionTest(unittest.TestCase):
    """Test three-class decision logic in adaptation."""

    def test_classify_decision_positive(self):
        from ace.adaptation import AdapterBase
        from ace import Diagnostics, ElementMatch
        
        adapter = AdapterBase(
            generator=Generator(DummyLLMClient()),
            reflector=Reflector(DummyLLMClient()),
            curator=Curator(DummyLLMClient()),
        )
        
        diagnostics = Diagnostics(
            elements=[
                ElementMatch("E1", "sensor", True, "explicit", 1.0),
                ElementMatch("E2", "controller", True, "explicit", 1.0),
            ],
            coverage=0.8,
            decision_basis_bullets=[]
        )
        
        decision = adapter._classify_decision(diagnostics)
        self.assertEqual(decision, "POSITIVE")
    
    def test_classify_decision_uncertain(self):
        from ace.adaptation import AdapterBase
        from ace import Diagnostics, ElementMatch
        
        adapter = AdapterBase(
            generator=Generator(DummyLLMClient()),
            reflector=Reflector(DummyLLMClient()),
            curator=Curator(DummyLLMClient()),
        )
        
        diagnostics = Diagnostics(
            elements=[
                ElementMatch("E1", "sensor", True, "functional", 0.7),
                ElementMatch("E2", "controller", True, "explicit", 1.0),
            ],
            coverage=0.5,
            decision_basis_bullets=[]
        )
        
        decision = adapter._classify_decision(diagnostics)
        self.assertEqual(decision, "UNCERTAIN")
    
    def test_classify_decision_negative(self):
        from ace.adaptation import AdapterBase
        from ace import Diagnostics, ElementMatch
        
        adapter = AdapterBase(
            generator=Generator(DummyLLMClient()),
            reflector=Reflector(DummyLLMClient()),
            curator=Curator(DummyLLMClient()),
        )
        
        diagnostics = Diagnostics(
            elements=[
                ElementMatch("E1", "sensor", True, "none", 0.0),
                ElementMatch("E2", "controller", True, "explicit", 1.0),
            ],
            coverage=0.3,
            decision_basis_bullets=[]
        )
        
        decision = adapter._classify_decision(diagnostics)
        self.assertEqual(decision, "NEGATIVE")


class IntegrationTest(unittest.TestCase):
    """Integration tests for the enhanced ACE system."""

    def test_enhanced_adaptation_flow(self):
        """Test full adaptation flow with diagnostics."""
        client = DummyLLMClient()
        
        # Queue responses
        client.queue(
            json.dumps({
                "reasoning": "Test reasoning",
                "bullet_ids": [],
                "final_answer": "positive",
            })
        )
        client.queue(
            json.dumps({
                "reasoning": "Reflection reasoning",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "Keep approach",
                "key_insight": "Test insight",
                "error_categories": [],
                "culprit_bullets": [],
                "bullet_tags": [],
            })
        )
        client.queue(
            json.dumps({
                "reasoning": "Curation reasoning",
                "operations": [],
            })
        )
        
        playbook = Playbook()
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client, enable_validation=False)
        
        adapter = OfflineAdapter(
            playbook=playbook,
            generator=generator,
            reflector=reflector,
            curator=curator,
            enable_gating=False,
        )
        
        class TestEnvironment(TaskEnvironment):
            def evaluate(self, sample, generator_output):
                # Return result with diagnostics
                diagnostics = Diagnostics(
                    elements=[
                        ElementMatch("E1", "element", True, "explicit", 1.0)
                    ],
                    coverage=0.8,
                    decision_basis_bullets=[]
                )
                return EnvironmentResult(
                    feedback="correct",
                    ground_truth="positive",
                    metrics={"accuracy": 1.0},
                    diagnostics=diagnostics,
                )
        
        sample = Sample(question="Test question", ground_truth="positive")
        results = adapter.run([sample], TestEnvironment(), epochs=1)
        
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].environment_result.diagnostics)


if __name__ == "__main__":
    unittest.main()
