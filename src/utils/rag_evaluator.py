"""RAG Evaluator Module.

Provides custom integration with AWS SageMaker using DeepEval's LLM-as-a-Judge pattern,
and a rigid CI/CD Quality Gate to evaluate the RAG Triad: Faithfulness, Answer Relevance,
and Context Precision. All comments and docstrings are in international English.
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Optional

import boto3
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

from pydantic import BaseModel

logger = logging.getLogger("athena.rag_evaluator")


class SageMakerLLM(DeepEvalBaseLLM):
    """Custom LLM wrapper for AWS SageMaker endpoints, compatible with DeepEval.

    Interfaces with SageMaker runtime endpoints using boto3 to serve as the
    evaluation Judge model in RAG evaluations.
    """

    def __init__(
        self,
        endpoint_name: str,
        region_name: str = "us-east-1",
        model_name: str = "SageMaker LLM Judge",
        sagemaker_client: Any = None,
    ) -> None:
        """Initialize the SageMaker LLM wrapper.

        Args:
            endpoint_name: The deployed AWS SageMaker endpoint name.
            region_name: The AWS region where the endpoint is hosted.
            model_name: Identifier name for the model used in reports.
            sagemaker_client: Pre-configured boto3 sagemaker-runtime client.
        """
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.model_name = model_name
        self.client = sagemaker_client

    def load_model(self) -> Any:
        """Load and return the boto3 SageMaker Runtime client.

        Returns:
            The SageMaker Runtime client.
        """
        if self.client is None:
            # We initialize lazy boto3 client to support flexible credentials configuration
            self.client = boto3.client("sagemaker-runtime", region_name=self.region_name)
        return self.client

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        """Invoke the SageMaker endpoint to generate a text completion.

        Uses standard HuggingFace Text Generation Inference (TGI) payload format.

        Args:
            prompt: The string prompt input.
            schema: Optional Pydantic model for structured output validation.

        Returns:
            The generated response string, or a parsed Pydantic model instance if schema is provided.
        """
        client = self.load_model()
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.01,
                "return_full_text": False,
            },
        }

        try:
            response = client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )
            response_body = response["Body"].read().decode("utf-8")
            data = json.loads(response_body)

            # Support various standard response structures: TGI list format vs dictionary
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                raw_text = data[0]["generated_text"].strip()
            elif isinstance(data, dict):
                if "generated_text" in data:
                    raw_text = data["generated_text"].strip()
                elif "generation" in data:
                    raw_text = data["generation"].strip()
                elif "output" in data:
                    raw_text = data["output"].strip()
                else:
                    raw_text = str(data).strip()
            else:
                raw_text = str(data).strip()

            if schema is not None:
                # DeepEval expects an instance of the pydantic model `schema`
                # We clean up any markdown JSON code fences if generated
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```json") or lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()

                if hasattr(schema, "model_validate_json"):
                    return schema.model_validate_json(cleaned)
                elif hasattr(schema, "parse_raw"):
                    return schema.parse_raw(cleaned)
                else:
                    parsed_json = json.loads(cleaned)
                    return schema(**parsed_json)

            return raw_text
        except Exception as e:
            logger.error("SageMaker invocation failed on endpoint %s: %s", self.endpoint_name, e)
            raise RuntimeError(f"SageMaker endpoint invocation failed: {e}") from e

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        """Asynchronously invoke the SageMaker endpoint to generate text.

        Executes the synchronous generate method in an event loop executor
        to prevent blocking main task execution.

        Args:
            prompt: The string prompt input.
            schema: Optional Pydantic model for structured output validation.

        Returns:
            The generated response string, or a parsed Pydantic model instance if schema is provided.
        """
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        """Get the identifier name of the model.

        Returns:
            Model name string.
        """
        return self.model_name


def evaluate_rag_triad(
    input_text: str,
    actual_output: str,
    retrieval_context: list[str],
    expected_output: Optional[str] = None,
    threshold: float = 0.85,
    judge_model: Optional[DeepEvalBaseLLM] = None,
) -> dict[str, Any]:
    """Evaluate a RAG prediction using the RAG Triad metrics.

    Evaluates:
    - Faithfulness (Groundedness): Is the answer derived from the context?
    - Answer Relevance: Does the answer directly address the user query?
    - Context Precision: Are relevant context items ranked higher?

    Acts as a rigid CI/CD Quality Gate, throwing AssertionError if any metric
    fails to meet the threshold.

    Args:
        input_text: The user query.
        actual_output: The generated RAG prediction.
        retrieval_context: List of context documents/nodes retrieved.
        expected_output: Optional ground-truth answer.
        threshold: Strict threshold (>= 0.85) required to pass.
        judge_model: Custom Judge LLM instance.

    Returns:
        Dictionary mapping metric names to their score, success status, and reason.

    Raises:
        ValueError: If inputs are invalid or null.
        AssertionError: If any metric fails the threshold.
    """
    if not input_text or not input_text.strip():
        raise ValueError("input_text cannot be null or empty")
    if not actual_output or not actual_output.strip():
        raise ValueError("actual_output cannot be null or empty")
    if not retrieval_context:
        raise ValueError("retrieval_context cannot be empty")

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Initialize metrics with explicit threshold & custom judge model
    faithfulness = FaithfulnessMetric(
        threshold=threshold, model=judge_model, include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=threshold, model=judge_model, include_reason=True
    )
    precision = ContextualPrecisionMetric(
        threshold=threshold, model=judge_model, include_reason=True
    )

    metrics = [faithfulness, relevancy, precision]
    results = {}
    failures = []

    # Run evaluations sequentially
    for metric in metrics:
        metric_name = metric.__class__.__name__
        try:
            metric.measure(test_case)
            score = metric.score
            success = metric.is_successful()
            reason = getattr(metric, "reason", "No reason provided")
        except Exception as e:
            logger.error("Failed to measure metric %s: %s", metric_name, e)
            score = 0.0
            success = False
            reason = f"Metric evaluation crashed: {e}"

        results[metric_name] = {
            "score": score,
            "success": success,
            "reason": reason,
        }

        if not success:
            failures.append((metric_name, score, reason))

    # Log results for observability & tracing
    for m_name, res in results.items():
        if res["success"]:
            logger.info("[Quality Gate Pass] %s: Score %.4f (Threshold: %.2f)", m_name, res["score"], threshold)
        else:
            logger.error(
                "[Quality Gate Fail] %s: Score %.4f (Threshold: %.2f). Reason: %s",
                m_name,
                res["score"],
                threshold,
                res["reason"],
            )

    # Enforce rigid quality gate block
    if failures:
        errors = []
        for f_name, f_score, f_reason in failures:
            errors.append(
                f"- {f_name} failed (Score: {f_score:.4f} < Threshold: {threshold:.2f}). Reason: {f_reason}"
            )
        raise AssertionError("RAG Quality Gate Check Failed:\n" + "\n".join(errors))

    return results
