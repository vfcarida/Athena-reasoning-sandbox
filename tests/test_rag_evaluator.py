"""Unit test suite for RAG Evaluator and SageMaker LLM.

Implements unit tests verifying the TDD RAG evaluation pipeline, including mock
SageMaker judge model invocations, validation handling, payload checks, and
CI/CD Quality Gate behavior. All comments and docstrings are in international English.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM

from src.utils.rag_evaluator import SageMakerLLM, evaluate_rag_triad


class MockSageMakerLLM(DeepEvalBaseLLM):
    """Mock Judge LLM simulation class.

    Returns structured JSON responses matching DeepEval's expectations for RAG metrics,
    allowing controlled failure states to verify CI/CD Quality Gate thresholds.
    """

    def __init__(
        self,
        model_name: str = "Mock SageMaker LLM",
        fail_faithfulness: bool = False,
        fail_relevancy: bool = False,
        fail_precision: bool = False,
    ) -> None:
        """Initialize Mock LLM state."""
        self.model_name = model_name
        self.fail_faithfulness = fail_faithfulness
        self.fail_relevancy = fail_relevancy
        self.fail_precision = fail_precision
        self.calls: list[str] = []

    def load_model(self) -> Any:
        return self

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        """Simulate LLM response dynamically based on the evaluation prompt.

        If schema is provided, returns a populated Pydantic model instance that conforms to
        the requested structure and test scenario flags.
        """
        self.calls.append(prompt)
        prompt_lower = prompt.lower()

        # If a schema is specified, we populate and return a Pydantic instance of it
        if schema is not None:
            module_name = schema.__module__.lower()
            schema_name = schema.__name__.lower()
            
            # Determine which metric is calling
            is_faithfulness = "faithfulness" in module_name
            is_relevancy = "answer_relevancy" in module_name
            is_precision = "contextual_precision" in module_name
            
            # Determine fail state
            should_fail = (
                (is_faithfulness and self.fail_faithfulness) or
                (is_relevancy and self.fail_relevancy) or
                (is_precision and self.fail_precision)
            )
            
            verdict_val = "no" if should_fail else "yes"
            
            if should_fail:
                if is_faithfulness:
                    reason_val = "The context states Berlin is the capital, contradicting Paris."
                elif is_relevancy:
                    reason_val = "The answer discusses Rome instead of the requested topic."
                else:
                    reason_val = "The retrieved node is irrelevant to the query."
            else:
                reason_val = "Passes validation check"

            data = {}
            for f_name, f_field in schema.model_fields.items():
                annotation_str = str(f_field.annotation)
                if "List" in annotation_str or "list" in annotation_str:
                    try:
                        inner_type = f_field.annotation.__args__[0]
                        if issubclass(inner_type, BaseModel):
                            # List of BaseModel (e.g. verdicts, claims)
                            inner_data = {}
                            for inner_f_name, inner_f_field in inner_type.model_fields.items():
                                if "verdict" in inner_f_name:
                                    inner_data[inner_f_name] = verdict_val
                                elif "reason" in inner_f_name:
                                    inner_data[inner_f_name] = reason_val
                                elif "key_point" in inner_f_name:
                                    inner_data[inner_f_name] = "Relevancy check target point"
                                elif "statement" in inner_f_name:
                                    inner_data[inner_f_name] = "Paris is the capital of France."
                                elif "claim" in inner_f_name:
                                    inner_data[inner_f_name] = "Paris is the capital of France."
                                else:
                                    inner_data[inner_f_name] = "test"
                            data[f_name] = [inner_type(**inner_data)]
                        else:
                            # List of strings/etc
                            if "statement" in f_name or "claim" in f_name:
                                data[f_name] = ["Paris is the capital of France."]
                            elif "truth" in f_name:
                                data[f_name] = ["Paris is the capital and most populous city of France."]
                            else:
                                data[f_name] = ["test"]
                    except Exception:
                        data[f_name] = []
                else:
                    # Scalar fields (like reasons in ScoreReason schemas)
                    if "reason" in f_name:
                        data[f_name] = reason_val
                    else:
                        data[f_name] = "test"
            
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            else:
                return schema(**data)

        # Fallback to string if schema is None
        if (
            "extract" in prompt_lower
            or "statements" in prompt_lower
            or "claims" in prompt_lower
            or "breakdown" in prompt_lower
        ):
            return json.dumps({
                "statements": ["Paris is the capital of France."],
                "claims": ["Paris is the capital of France."],
            })

        if "faithfulness" in prompt_lower or "truth" in prompt_lower or "ground" in prompt_lower:
            if self.fail_faithfulness:
                return json.dumps({
                    "verdicts": [
                        {
                            "verdict": "no",
                            "reason": "The context states Berlin is the capital, contradicting Paris.",
                        }
                    ]
                })
            return json.dumps({
                "verdicts": [
                    {
                        "verdict": "yes",
                        "reason": "The context explicitly supports that Paris is the capital.",
                    }
                ]
            })

        if "relevancy" in prompt_lower or "relevant" in prompt_lower:
            if self.fail_relevancy:
                return json.dumps({
                    "verdicts": [
                        {
                            "verdict": "no",
                            "reason": "The answer discusses Rome instead of the requested topic.",
                        }
                    ]
                })
            return json.dumps({
                "verdicts": [
                    {
                        "verdict": "yes",
                        "reason": "The generated answer directly addresses the user query.",
                    }
                ]
            })

        if "precision" in prompt_lower or "node" in prompt_lower:
            if self.fail_precision:
                return json.dumps({
                    "verdicts": [
                        {
                            "verdict": "no",
                            "reason": "The retrieved node is irrelevant to the query.",
                        }
                    ]
                })
            return json.dumps({
                "verdicts": [
                    {
                        "verdict": "yes",
                        "reason": "The retrieved node contains direct information about the query.",
                    }
                ]
            })

        return json.dumps({
            "statements": ["Paris is the capital of France."],
            "verdicts": [{"verdict": "yes", "reason": "Default positive feedback."}],
        })

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


def test_evaluate_rag_triad_success() -> None:
    """Verify that when all criteria are met, the evaluation passes successfully."""
    judge = MockSageMakerLLM()

    results = evaluate_rag_triad(
        input_text="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        retrieval_context=["Paris is the capital and most populous city of France."],
        expected_output="Paris is the capital of France.",
        threshold=0.85,
        judge_model=judge,
    )

    assert "FaithfulnessMetric" in results
    assert "AnswerRelevancyMetric" in results
    assert "ContextualPrecisionMetric" in results

    assert results["FaithfulnessMetric"]["success"] is True
    assert results["FaithfulnessMetric"]["score"] >= 0.85
    assert results["AnswerRelevancyMetric"]["success"] is True
    assert results["AnswerRelevancyMetric"]["score"] >= 0.85
    assert results["ContextualPrecisionMetric"]["success"] is True
    assert results["ContextualPrecisionMetric"]["score"] >= 0.85


def test_evaluate_rag_triad_faithfulness_failure() -> None:
    """Verify that a Faithfulness failure is caught and raises an AssertionError."""
    judge = MockSageMakerLLM(fail_faithfulness=True)

    with pytest.raises(AssertionError) as exc_info:
        evaluate_rag_triad(
            input_text="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["Berlin is the capital of Germany."],
            expected_output="Paris is the capital of France.",
            threshold=0.85,
            judge_model=judge,
        )

    err_msg = str(exc_info.value)
    assert "FaithfulnessMetric failed" in err_msg
    assert "contradicting Paris" in err_msg


def test_evaluate_rag_triad_relevancy_failure() -> None:
    """Verify that an AnswerRelevancy failure is caught and raises an AssertionError."""
    judge = MockSageMakerLLM(fail_relevancy=True)

    with pytest.raises(AssertionError) as exc_info:
        evaluate_rag_triad(
            input_text="What is the capital of France?",
            actual_output="Rome is the capital of Italy.",
            retrieval_context=["Paris is the capital of France."],
            expected_output="Paris is the capital of France.",
            threshold=0.85,
            judge_model=judge,
        )

    err_msg = str(exc_info.value)
    assert "AnswerRelevancyMetric failed" in err_msg
    assert "discusses Rome instead of" in err_msg


def test_evaluate_rag_triad_precision_failure() -> None:
    """Verify that a ContextualPrecision failure is caught and raises an AssertionError."""
    judge = MockSageMakerLLM(fail_precision=True)

    with pytest.raises(AssertionError) as exc_info:
        evaluate_rag_triad(
            input_text="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["The weather in Madrid is sunny today."],
            expected_output="Paris is the capital of France.",
            threshold=0.85,
            judge_model=judge,
        )

    err_msg = str(exc_info.value)
    assert "ContextualPrecisionMetric failed" in err_msg
    assert "irrelevant to the query" in err_msg


def test_evaluate_rag_triad_validation_errors() -> None:
    """Verify that passing empty/invalid values raises ValueError."""
    judge = MockSageMakerLLM()

    with pytest.raises(ValueError, match="input_text cannot be null or empty"):
        evaluate_rag_triad("", "Paris is capital", ["Context"], judge_model=judge)

    with pytest.raises(ValueError, match="actual_output cannot be null or empty"):
        evaluate_rag_triad("Query", "", ["Context"], judge_model=judge)

    with pytest.raises(ValueError, match="retrieval_context cannot be empty"):
        evaluate_rag_triad("Query", "Paris is capital", [], judge_model=judge)


def test_sagemaker_llm_payload_and_invocation() -> None:
    """Verify that SageMakerLLM serializes payloads correctly and parses common response shapes."""
    mock_runtime_client = Mock()

    # Mock SageMaker response body returned by boto3 runtime client
    mock_body = Mock()
    # Case 1: TGI list format response
    mock_body.read.return_value = json.dumps([{"generated_text": "Response from TGI"}]).encode("utf-8")
    mock_runtime_client.invoke_endpoint.return_value = {"Body": mock_body}

    llm = SageMakerLLM(
        endpoint_name="test-endpoint",
        region_name="us-east-1",
        model_name="Custom Judge",
        sagemaker_client=mock_runtime_client,
    )

    # Validate model details
    assert llm.get_model_name() == "Custom Judge"

    # Execute text generation
    response = llm.generate("Sample prompt")
    assert response == "Response from TGI"

    # Assert invoke_endpoint parameter formats
    mock_runtime_client.invoke_endpoint.assert_called_once()
    kwargs = mock_runtime_client.invoke_endpoint.call_args[1]
    assert kwargs["EndpointName"] == "test-endpoint"
    assert kwargs["ContentType"] == "application/json"
    assert kwargs["Accept"] == "application/json"
    
    payload = json.loads(kwargs["Body"])
    assert payload["inputs"] == "Sample prompt"
    assert payload["parameters"]["temperature"] == 0.01

    # Case 2: Dict response with "generation" key
    mock_body.read.return_value = json.dumps({"generation": "Response from Dict"}).encode("utf-8")
    response_dict = llm.generate("Sample prompt 2")
    assert response_dict == "Response from Dict"


@pytest.mark.asyncio
async def test_sagemaker_llm_async_generate() -> None:
    """Verify that async wrapper properly returns results without exceptions."""
    mock_runtime_client = Mock()
    mock_body = Mock()
    mock_body.read.return_value = json.dumps([{"generated_text": "Async output"}]).encode("utf-8")
    mock_runtime_client.invoke_endpoint.return_value = {"Body": mock_body}

    llm = SageMakerLLM(
        endpoint_name="test-endpoint",
        sagemaker_client=mock_runtime_client,
    )

    response = await llm.a_generate("Async prompt")
    assert response == "Async output"
