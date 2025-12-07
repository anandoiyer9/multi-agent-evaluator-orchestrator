"""
Multi-Agent Evaluator-Orchestrator (MAEO)

A production-grade framework for building multi-agent LLM systems using the
evaluator-optimizer pattern.

Author: Anand Ramakrishnan
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class GenerationResult:
    """Result from a generation attempt."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result from an evaluation."""
    dimension: str
    score: float  # 0-100
    feedback: str
    model: str
    passed: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Final result from the orchestration loop."""
    content: str
    score: float
    iterations: int
    evaluations: list[EvaluationResult]
    feedback: str
    converged: bool
    routed_to_human: bool
    total_latency_ms: float
    total_cost: float
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """Chunk for streaming responses."""
    type: str  # "generation", "evaluation", "iteration", "complete"
    content: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    iteration: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """Generate content from the LLM."""
        pass


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude models."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """Generate content using Claude."""
        import anthropic
        import time
        
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerationResult(
            content=response.content[0].text,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            metadata={"stop_reason": response.stop_reason},
        )


class OpenAIClient(LLMClient):
    """Client for OpenAI's GPT models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """Generate content using GPT."""
        import openai
        import time
        
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerationResult(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens,
            latency_ms=latency_ms,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )


class GeneratorAgent:
    """Agent specialized for content generation."""
    
    def __init__(
        self,
        model: str,
        specialty: str,
        client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.specialty = specialty
        self.client = client
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on specialty."""
        prompts = {
            "factual": (
                "You are a precise, factual content generator. "
                "Focus on accuracy, clarity, and verifiable information. "
                "Cite sources when possible. Avoid speculation."
            ),
            "creative": (
                "You are a creative, engaging content generator. "
                "Focus on compelling narratives, vivid language, and emotional resonance. "
                "Be original and memorable while staying on-brand."
            ),
            "technical": (
                "You are a technical content generator. "
                "Focus on precision, correct terminology, and clear explanations. "
                "Include code examples and technical details where appropriate."
            ),
        }
        return prompts.get(self.specialty, "You are a helpful content generator.")
    
    async def generate(
        self,
        task: str,
        context: Optional[dict] = None,
        previous_feedback: Optional[str] = None,
    ) -> GenerationResult:
        """Generate content for the given task."""
        prompt = self._build_prompt(task, context, previous_feedback)
        return await self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )
    
    def _build_prompt(
        self,
        task: str,
        context: Optional[dict],
        previous_feedback: Optional[str],
    ) -> str:
        """Build the generation prompt."""
        parts = [f"Task: {task}"]
        
        if context:
            parts.append(f"\nContext: {context}")
        
        if previous_feedback:
            parts.append(
                f"\n\nPrevious attempt received this feedback:\n{previous_feedback}"
                "\n\nPlease improve upon the previous attempt addressing this feedback."
            )
        
        return "\n".join(parts)


class EvaluatorAgent:
    """Agent specialized for evaluating generated content."""
    
    def __init__(
        self,
        dimension: str,
        model: str = "claude-3-opus-20240229",
        client: Optional[LLMClient] = None,
        threshold: float = 80.0,
        weight: float = 1.0,
        scoring_rubric: Optional[str] = None,
    ):
        self.dimension = dimension
        self.model = model
        self.client = client
        self.threshold = threshold
        self.weight = weight
        self.scoring_rubric = scoring_rubric or self._default_rubric()
    
    def _default_rubric(self) -> str:
        """Generate default scoring rubric based on dimension."""
        rubrics = {
            "accuracy": """
Score 0-100 based on:
- Factual correctness: Are all claims verifiable? (40%)
- Source attribution: Are sources cited where appropriate? (30%)
- No hallucinations: Are there any fabricated facts, quotes, or statistics? (30%)
""",
            "brand_safety": """
Score 0-100 based on:
- Tone alignment: Does the content match the brand voice? (35%)
- No controversial content: Is the content free of potentially offensive material? (35%)
- Professional quality: Does it meet professional standards? (30%)
""",
            "readability": """
Score 0-100 based on:
- Clarity: Is the content easy to understand? (40%)
- Structure: Is the content well-organized? (30%)
- Engagement: Is the content compelling to read? (30%)
""",
            "technical_accuracy": """
Score 0-100 based on:
- Correctness: Are technical claims accurate? (50%)
- Completeness: Are important details included? (25%)
- Best practices: Does it follow industry standards? (25%)
""",
        }
        return rubrics.get(self.dimension, "Score 0-100 based on overall quality.")
    
    async def evaluate(
        self,
        content: str,
        task: str,
        context: Optional[dict] = None,
    ) -> EvaluationResult:
        """Evaluate the generated content."""
        prompt = self._build_evaluation_prompt(content, task, context)
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=self._evaluation_system_prompt(),
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        
        score, feedback = self._parse_evaluation(result.content)
        
        return EvaluationResult(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            model=self.model,
            passed=score >= self.threshold,
            metadata={"raw_response": result.content},
        )
    
    def _evaluation_system_prompt(self) -> str:
        """System prompt for evaluation."""
        return (
            "You are an expert content evaluator. Your job is to critically assess "
            "content quality across specific dimensions. Be rigorous and objective. "
            "Provide actionable feedback for improvement."
        )
    
    def _build_evaluation_prompt(
        self,
        content: str,
        task: str,
        context: Optional[dict],
    ) -> str:
        """Build the evaluation prompt."""
        return f"""
Evaluate the following content for the dimension: {self.dimension}

Original Task: {task}
Context: {context or 'None provided'}

Content to Evaluate:
---
{content}
---

Scoring Rubric:
{self.scoring_rubric}

Provide your evaluation in this exact format:
SCORE: [number 0-100]
FEEDBACK: [detailed feedback with specific suggestions for improvement]
"""
    
    def _parse_evaluation(self, response: str) -> tuple[float, str]:
        """Parse the evaluation response to extract score and feedback."""
        lines = response.strip().split("\n")
        score = 50.0  # Default
        feedback = ""
        
        for i, line in enumerate(lines):
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    score = float(score_str.split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
                # Include remaining lines as part of feedback
                feedback += "\n" + "\n".join(lines[i + 1:])
                break
        
        return score, feedback.strip()


class ConvergenceStrategy(ABC):
    """Abstract base class for convergence strategies."""
    
    def __init__(self, threshold: float = 85.0):
        self.threshold = threshold
    
    @abstractmethod
    def should_converge(self, evaluations: list[EvaluationResult]) -> bool:
        """Determine if the output has converged to acceptable quality."""
        pass


class WeightedAverageConvergence(ConvergenceStrategy):
    """Convergence based on weighted average of evaluation scores."""
    
    def should_converge(self, evaluations: list[EvaluationResult]) -> bool:
        """Check if weighted average exceeds threshold."""
        if not evaluations:
            return False
        
        total_weight = sum(1.0 for _ in evaluations)  # Equal weights for now
        weighted_sum = sum(e.score for e in evaluations)
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
        
        return weighted_avg >= self.threshold


class AllPassConvergence(ConvergenceStrategy):
    """Convergence requires all evaluators to pass their thresholds."""
    
    def should_converge(self, evaluations: list[EvaluationResult]) -> bool:
        """Check if all evaluations passed."""
        return all(e.passed for e in evaluations)


class HumanReviewHandler:
    """Handler for routing to human review."""
    
    def __init__(
        self,
        trigger_threshold: float = 70.0,
        callback: Optional[Callable] = None,
    ):
        self.trigger_threshold = trigger_threshold
        self.callback = callback
    
    def should_trigger(
        self,
        evaluations: list[EvaluationResult],
        iterations: int,
        max_iterations: int,
    ) -> bool:
        """Determine if human review should be triggered."""
        if iterations < max_iterations:
            return False
        
        avg_score = sum(e.score for e in evaluations) / len(evaluations)
        return avg_score < self.trigger_threshold
    
    async def trigger(
        self,
        content: str,
        evaluations: list[EvaluationResult],
        task: str,
    ) -> None:
        """Trigger human review."""
        if self.callback:
            await asyncio.to_thread(
                self.callback,
                content,
                {e.dimension: e.score for e in evaluations},
                "\n".join(e.feedback for e in evaluations),
            )


class Orchestrator:
    """
    Central orchestrator for the multi-agent evaluator-optimizer pattern.
    
    Manages the generation-evaluation loop until convergence or max iterations.
    """
    
    def __init__(
        self,
        generators: list[GeneratorAgent],
        evaluators: list[EvaluatorAgent],
        max_iterations: int = 3,
        convergence_threshold: float = 85.0,
        convergence_strategy: Optional[ConvergenceStrategy] = None,
        human_review_handler: Optional[HumanReviewHandler] = None,
    ):
        self.generators = generators
        self.evaluators = evaluators
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_strategy = convergence_strategy or WeightedAverageConvergence(
            threshold=convergence_threshold
        )
        self.human_review_handler = human_review_handler
        
        # Build generator lookup by specialty
        self._generator_map = {g.specialty: g for g in generators}
    
    def _select_generator(self, task: str, context: Optional[dict]) -> GeneratorAgent:
        """Select the appropriate generator based on task characteristics."""
        # Simple heuristic - can be made more sophisticated
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ["fact", "data", "statistic", "accurate"]):
            specialty = "factual"
        elif any(kw in task_lower for kw in ["creative", "story", "engage", "compelling"]):
            specialty = "creative"
        elif any(kw in task_lower for kw in ["code", "technical", "api", "implement"]):
            specialty = "technical"
        else:
            specialty = "factual"  # Default to factual
        
        return self._generator_map.get(specialty, self.generators[0])
    
    async def generate(
        self,
        task: str,
        context: Optional[dict] = None,
    ) -> OrchestrationResult:
        """
        Generate content with quality guarantees through the evaluator-optimizer loop.
        """
        import time
        
        start_time = time.time()
        total_cost = 0.0
        
        generator = self._select_generator(task, context)
        
        current_content = ""
        previous_feedback = None
        all_evaluations = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Generate
            gen_result = await generator.generate(
                task=task,
                context=context,
                previous_feedback=previous_feedback,
            )
            current_content = gen_result.content
            total_cost += self._estimate_cost(gen_result)
            
            logger.info(f"Generated {len(current_content)} chars with {generator.model}")
            
            # Evaluate with all evaluators
            evaluations = await asyncio.gather(*[
                evaluator.evaluate(current_content, task, context)
                for evaluator in self.evaluators
            ])
            all_evaluations = list(evaluations)
            
            for eval_result in evaluations:
                total_cost += 0.01  # Rough cost estimate
                logger.info(
                    f"  {eval_result.dimension}: {eval_result.score:.1f} "
                    f"({'PASS' if eval_result.passed else 'FAIL'})"
                )
            
            # Check convergence
            if self.convergence_strategy.should_converge(evaluations):
                logger.info(f"Converged after {iteration} iteration(s)")
                return self._build_result(
                    content=current_content,
                    evaluations=evaluations,
                    iterations=iteration,
                    converged=True,
                    routed_to_human=False,
                    start_time=start_time,
                    total_cost=total_cost,
                )
            
            # Prepare feedback for next iteration
            previous_feedback = self._aggregate_feedback(evaluations)
        
        # Max iterations reached - check if human review needed
        routed_to_human = False
        if self.human_review_handler:
            if self.human_review_handler.should_trigger(
                all_evaluations, self.max_iterations, self.max_iterations
            ):
                await self.human_review_handler.trigger(
                    current_content, all_evaluations, task
                )
                routed_to_human = True
        
        logger.warning(f"Did not converge after {self.max_iterations} iterations")
        
        return self._build_result(
            content=current_content,
            evaluations=all_evaluations,
            iterations=self.max_iterations,
            converged=False,
            routed_to_human=routed_to_human,
            start_time=start_time,
            total_cost=total_cost,
        )
    
    async def generate_stream(
        self,
        task: str,
        context: Optional[dict] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate content with streaming updates."""
        generator = self._select_generator(task, context)
        previous_feedback = None
        
        for iteration in range(1, self.max_iterations + 1):
            yield StreamChunk(type="iteration", iteration=iteration)
            
            # Generate
            gen_result = await generator.generate(
                task=task,
                context=context,
                previous_feedback=previous_feedback,
            )
            
            yield StreamChunk(type="generation", content=gen_result.content)
            
            # Evaluate
            evaluations = await asyncio.gather(*[
                evaluator.evaluate(gen_result.content, task, context)
                for evaluator in self.evaluators
            ])
            
            for eval_result in evaluations:
                yield StreamChunk(
                    type="evaluation",
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    metadata={"dimension": eval_result.dimension},
                )
            
            # Check convergence
            if self.convergence_strategy.should_converge(list(evaluations)):
                yield StreamChunk(
                    type="complete",
                    content=gen_result.content,
                    score=sum(e.score for e in evaluations) / len(evaluations),
                    metadata={"iterations": iteration, "converged": True},
                )
                return
            
            previous_feedback = self._aggregate_feedback(list(evaluations))
        
        # Did not converge
        yield StreamChunk(
            type="complete",
            content=gen_result.content,
            score=sum(e.score for e in evaluations) / len(evaluations),
            metadata={"iterations": self.max_iterations, "converged": False},
        )
    
    def _aggregate_feedback(self, evaluations: list[EvaluationResult]) -> str:
        """Aggregate feedback from all evaluators."""
        feedback_parts = []
        for eval_result in evaluations:
            if not eval_result.passed:
                feedback_parts.append(
                    f"[{eval_result.dimension}] Score: {eval_result.score:.1f}\n"
                    f"{eval_result.feedback}"
                )
        return "\n\n".join(feedback_parts)
    
    def _estimate_cost(self, gen_result: GenerationResult) -> float:
        """Estimate cost based on tokens used."""
        # Rough estimates - adjust based on actual pricing
        cost_per_1k = {
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.0005,
        }
        
        for model_key, cost in cost_per_1k.items():
            if model_key in gen_result.model.lower():
                return (gen_result.tokens_used / 1000) * cost
        
        return 0.01  # Default estimate
    
    def _build_result(
        self,
        content: str,
        evaluations: list[EvaluationResult],
        iterations: int,
        converged: bool,
        routed_to_human: bool,
        start_time: float,
        total_cost: float,
    ) -> OrchestrationResult:
        """Build the final orchestration result."""
        import time
        
        avg_score = sum(e.score for e in evaluations) / len(evaluations)
        combined_feedback = "\n".join(
            f"[{e.dimension}] {e.feedback}" for e in evaluations
        )
        
        return OrchestrationResult(
            content=content,
            score=avg_score,
            iterations=iterations,
            evaluations=evaluations,
            feedback=combined_feedback,
            converged=converged,
            routed_to_human=routed_to_human,
            total_latency_ms=(time.time() - start_time) * 1000,
            total_cost=total_cost,
        )


# Convenience exports
__all__ = [
    "Orchestrator",
    "GeneratorAgent",
    "EvaluatorAgent",
    "GenerationResult",
    "EvaluationResult",
    "OrchestrationResult",
    "StreamChunk",
    "ConvergenceStrategy",
    "WeightedAverageConvergence",
    "AllPassConvergence",
    "HumanReviewHandler",
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
]
