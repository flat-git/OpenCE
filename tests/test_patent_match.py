"""Tests for patent matching components."""

import json
import tempfile
import unittest
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import Sample, GeneratorOutput, DummyLLMClient, Generator

# Import patent matching components
sys.path.insert(0, str(ROOT / "scripts" / "patent_match"))
from loader import load_patent_samples
from environment import PatentMatchEnvironment
from prompts import GENERATOR_PROMPT_PATENT_CLS


class LoaderTest(unittest.TestCase):
    """Test patent data loader."""
    
    def test_load_patent_samples_basic(self):
        """Test loading basic patent samples."""
        test_data = [
            {
                "question": "Test patent question",
                "positive_ctxs": [
                    {"id": "pos-1", "text": "Positive context 1"}
                ],
                "negative_ctxs": [
                    {"id": "neg-1", "text": "Negative context 1"}
                ],
                "hard_negative_ctxs": [
                    {"id": "hard-1", "text": "Hard negative context 1"}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        try:
            samples = load_patent_samples(temp_path)
            self.assertEqual(len(samples), 1)
            
            sample = samples[0]
            self.assertEqual(sample.question, "Test patent question")
            
            # Parse context
            context_data = json.loads(sample.context)
            candidates = context_data["candidates"]
            ground_truth_ids = context_data["ground_truth_ids"]
            
            self.assertEqual(len(candidates), 3)
            self.assertEqual(len(ground_truth_ids), 1)
            self.assertIn("pos-1", ground_truth_ids)
            
            # Check candidate structure
            pos_candidate = next(c for c in candidates if c["id"] == "pos-1")
            self.assertEqual(pos_candidate["label"], "positive")
            self.assertEqual(pos_candidate["type"], "positive")
            
            neg_candidate = next(c for c in candidates if c["id"] == "neg-1")
            self.assertEqual(neg_candidate["label"], "negative")
            self.assertEqual(neg_candidate["type"], "negative")
            
            hard_candidate = next(c for c in candidates if c["id"] == "hard-1")
            self.assertEqual(hard_candidate["label"], "negative")
            self.assertEqual(hard_candidate["type"], "hard_negative")
        finally:
            temp_path.unlink()
    
    def test_load_multiple_samples(self):
        """Test loading multiple samples."""
        test_data = [
            {
                "question": "Question 1",
                "positive_ctxs": [{"id": "p1", "text": "text1"}],
                "negative_ctxs": [],
                "hard_negative_ctxs": []
            },
            {
                "question": "Question 2",
                "positive_ctxs": [{"id": "p2", "text": "text2"}],
                "negative_ctxs": [{"id": "n2", "text": "text3"}],
                "hard_negative_ctxs": []
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        try:
            samples = load_patent_samples(temp_path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].question, "Question 1")
            self.assertEqual(samples[1].question, "Question 2")
        finally:
            temp_path.unlink()


class EnvironmentTest(unittest.TestCase):
    """Test PatentMatchEnvironment."""
    
    def test_evaluate_perfect_classification(self):
        """Test evaluation with perfect classification."""
        # Create sample with known candidates
        context_data = {
            "candidates": [
                {"id": "pos-1", "text": "text1", "label": "positive", "type": "positive"},
                {"id": "neg-1", "text": "text2", "label": "negative", "type": "negative"}
            ],
            "ground_truth_ids": ["pos-1"]
        }
        sample = Sample(
            question="Test question",
            context=json.dumps(context_data)
        )
        
        # Create generator output with correct predictions
        generator_output = GeneratorOutput(
            reasoning="Test reasoning",
            final_answer="Classified correctly",
            bullet_ids=[],
            raw={
                "predictions": [
                    {"id": "pos-1", "label": "positive", "reason": "Correct positive"},
                    {"id": "neg-1", "label": "negative", "reason": "Correct negative"}
                ]
            }
        )
        
        env = PatentMatchEnvironment()
        result = env.evaluate(sample, generator_output)
        
        # Check metrics
        self.assertEqual(result.metrics["accuracy"], 1.0)
        self.assertEqual(result.metrics["precision"], 1.0)
        self.assertEqual(result.metrics["recall"], 1.0)
        self.assertEqual(result.metrics["f1"], 1.0)
        self.assertEqual(result.metrics["tp"], 1.0)
        self.assertEqual(result.metrics["fp"], 0.0)
        self.assertEqual(result.metrics["fn"], 0.0)
        self.assertEqual(result.metrics["tn"], 1.0)
    
    def test_evaluate_with_false_positive(self):
        """Test evaluation with false positive error."""
        context_data = {
            "candidates": [
                {"id": "pos-1", "text": "text1", "label": "positive", "type": "positive"},
                {"id": "neg-1", "text": "text2", "label": "negative", "type": "negative"}
            ],
            "ground_truth_ids": ["pos-1"]
        }
        sample = Sample(
            question="Test question",
            context=json.dumps(context_data)
        )
        
        # Create generator output with FP (neg-1 labeled as positive)
        generator_output = GeneratorOutput(
            reasoning="Test reasoning",
            final_answer="Made an error",
            bullet_ids=[],
            raw={
                "predictions": [
                    {"id": "pos-1", "label": "positive", "reason": "Correct"},
                    {"id": "neg-1", "label": "positive", "reason": "Incorrectly labeled positive"}
                ]
            }
        )
        
        env = PatentMatchEnvironment()
        result = env.evaluate(sample, generator_output)
        
        # Check metrics
        self.assertEqual(result.metrics["tp"], 1.0)
        self.assertEqual(result.metrics["fp"], 1.0)
        self.assertEqual(result.metrics["fn"], 0.0)
        self.assertEqual(result.metrics["precision"], 0.5)
        self.assertEqual(result.metrics["recall"], 1.0)
        
        # Check error details in feedback
        self.assertIn("JSON:", result.feedback)
        json_part = result.feedback.split("JSON:")[1].strip()
        error_data = json.loads(json_part)
        self.assertIn("neg-1", error_data["fp_from_negative_ids"])
    
    def test_evaluate_with_false_negative(self):
        """Test evaluation with false negative error."""
        context_data = {
            "candidates": [
                {"id": "pos-1", "text": "text1", "label": "positive", "type": "positive"},
                {"id": "pos-2", "text": "text2", "label": "positive", "type": "positive"}
            ],
            "ground_truth_ids": ["pos-1", "pos-2"]
        }
        sample = Sample(
            question="Test question",
            context=json.dumps(context_data)
        )
        
        # Create generator output with FN (pos-2 labeled as negative)
        generator_output = GeneratorOutput(
            reasoning="Test reasoning",
            final_answer="Missed one",
            bullet_ids=[],
            raw={
                "predictions": [
                    {"id": "pos-1", "label": "positive", "reason": "Correct"},
                    {"id": "pos-2", "label": "negative", "reason": "Incorrectly labeled negative"}
                ]
            }
        )
        
        env = PatentMatchEnvironment()
        result = env.evaluate(sample, generator_output)
        
        # Check metrics
        self.assertEqual(result.metrics["tp"], 1.0)
        self.assertEqual(result.metrics["fp"], 0.0)
        self.assertEqual(result.metrics["fn"], 1.0)
        self.assertEqual(result.metrics["precision"], 1.0)
        self.assertEqual(result.metrics["recall"], 0.5)
        
        # Check error details
        json_part = result.feedback.split("JSON:")[1].strip()
        error_data = json.loads(json_part)
        self.assertIn("pos-2", error_data["fn_positive_ids"])
    
    def test_evaluate_with_hard_negative_fp(self):
        """Test evaluation distinguishes hard negative FP."""
        context_data = {
            "candidates": [
                {"id": "pos-1", "text": "text1", "label": "positive", "type": "positive"},
                {"id": "hard-1", "text": "text2", "label": "negative", "type": "hard_negative"}
            ],
            "ground_truth_ids": ["pos-1"]
        }
        sample = Sample(
            question="Test question",
            context=json.dumps(context_data)
        )
        
        # Mislabel hard negative as positive
        generator_output = GeneratorOutput(
            reasoning="Test reasoning",
            final_answer="Error",
            bullet_ids=[],
            raw={
                "predictions": [
                    {"id": "pos-1", "label": "positive", "reason": "Correct"},
                    {"id": "hard-1", "label": "positive", "reason": "Tricked by hard negative"}
                ]
            }
        )
        
        env = PatentMatchEnvironment()
        result = env.evaluate(sample, generator_output)
        
        # Check that hard negative FP is tracked separately
        json_part = result.feedback.split("JSON:")[1].strip()
        error_data = json.loads(json_part)
        self.assertIn("hard-1", error_data["fp_from_hard_negative_ids"])
        self.assertEqual(len(error_data["fp_from_negative_ids"]), 0)


class PromptsTest(unittest.TestCase):
    """Test prompt templates."""
    
    def test_generator_prompt_has_required_fields(self):
        """Test generator prompt has necessary placeholders."""
        self.assertIn("{playbook}", GENERATOR_PROMPT_PATENT_CLS)
        self.assertIn("{reflection}", GENERATOR_PROMPT_PATENT_CLS)
        self.assertIn("{question}", GENERATOR_PROMPT_PATENT_CLS)
        self.assertIn("{context}", GENERATOR_PROMPT_PATENT_CLS)
        self.assertIn("predictions", GENERATOR_PROMPT_PATENT_CLS)


if __name__ == "__main__":
    unittest.main()
