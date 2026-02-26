# Multi-Agent Evaluator-Orchestrator

Reference implementation of the **evaluator-optimizer pattern** for production LLM systems. Sanitized from proprietary systems running at 50M+ user scale.

The core insight: **models are better critics than generators.** Separate generation from evaluation, use different model families for each, and iterate to convergence. This pattern reduced hallucination rates from ~15% to <2% in production.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                      │
│  Selects generator → runs eval loop → checks gates   │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
     ┌─────────▼─────────┐    ┌────────▼────────────┐
     │  GENERATOR AGENTS │    │  EVALUATOR AGENTS   │
     │  Claude (factual) │    │  Accuracy (0-100)   │
     │  GPT-4 (creative) │    │  Brand safety       │
     │ Custom (technical)│    │  Readability        │
     └─────────┬─────────┘    └────────┬────────────┘
               │                       │
               └───────────┬───────────┘
                           ▼
                  ┌─────────────────┐
                  │  CONVERGENCE    │
                  │  All pass? Done │
                  │  Below 70?      │
                  │    → Human      │
                  │  Else: retry    │
                  │   with feedback │
                  └─────────────────┘
```

## What's Here

`orchestrator.py` — Single-file implementation (~725 lines) containing:

- **`Orchestrator`** — Generation-evaluation loop with configurable convergence
- **`GeneratorAgent`** / **`EvaluatorAgent`** — Specialized agents with provider-agnostic LLM clients
- **`ConvergenceStrategy`** — Pluggable strategies (weighted average, all-pass)
- **`HumanReviewHandler`** — Routes to human review when quality gates aren't met after max iterations
- **`AnthropicClient`** / **`OpenAIClient`** — Async LLM clients for cross-model evaluation

## Key Design Decisions

**1. Cross-model evaluation catches correlated failures.**
If GPT-4 generates, Claude evaluates (and vice versa). Same-model evaluation has blind spots — the generator's biases leak into its own quality assessment.

**2. Evaluators are specialized, not generalist.**
One evaluator per dimension (accuracy, safety, readability). A single "rate this 0-100" evaluator underperforms specialized critics with targeted rubrics.

**3. Convergence is pluggable.**
`WeightedAverageConvergence` for cost-sensitive deployments (good enough = ship it). `AllPassConvergence` for high-stakes content where every dimension must clear its gate.

**4. Cap iterations at 3.**
Diminishing returns after 3 rounds. If it hasn't converged, the problem is usually the prompt or the task decomposition, not the iteration count. Route to human.

**5. Cost tracking is built in.**
Every generation and evaluation is tracked. In production, the 2-3x inference cost of quality gates is cheaper than one hallucinated article reaching 50M users.

## Usage

```python
import asyncio
from orchestrator import (
    Orchestrator, GeneratorAgent, EvaluatorAgent,
    AnthropicClient, OpenAIClient, AllPassConvergence,
)

async def main():
    # Cross-model: generate with GPT-4, evaluate with Claude
    gpt_client = OpenAIClient(api_key="...", model="gpt-4")
    claude_client = AnthropicClient(api_key="...", model="claude-3-sonnet-20240229")

    orchestrator = Orchestrator(
        generators=[
            GeneratorAgent(model="gpt-4", specialty="factual", client=gpt_client),
        ],
        evaluators=[
            EvaluatorAgent(dimension="accuracy", client=claude_client, threshold=85),
            EvaluatorAgent(dimension="brand_safety", client=claude_client, threshold=90),
        ],
        max_iterations=3,
        convergence_strategy=AllPassConvergence(threshold=85),
    )

    result = await orchestrator.generate(
        task="Write a product description for an enterprise AI platform",
        context={"audience": "CTOs", "tone": "professional"},
    )

    print(f"Converged: {result.converged} after {result.iterations} iteration(s)")
    print(f"Score: {result.score:.1f}")
    print(f"Cost: ${result.total_cost:.4f}")

asyncio.run(main())
```

## Production Learnings

This pattern emerged from running multi-agent content systems at scale. Some things that aren't obvious from the code:

- **Evaluation temperature should be low (0.1).** You want consistent scoring, not creative interpretation of quality.
- **Feedback aggregation matters.** Only failed dimensions feed back to the generator. Passing dimensions are noise that dilutes the signal.
- **The human review threshold (70) is lower than convergence (85).** This creates a middle zone where the system keeps trying before escalating.
- **Streaming (`generate_stream`) exists for UX.** Users watching a generation-evaluation loop need progress signals, not a loading spinner followed by a wall of text.

## What's Not Here

This is a sanitized reference implementation, not the full production system. The production version includes:

- MongoDB-backed state management and audit logging
- Circuit breakers for provider failover
- Regression test gates blocking deployment on quality drift
- 2k-item grounded QA evaluation sets
- Semantic caching layer for inference cost reduction

## Requirements

```
anthropic>=0.18.0
openai>=1.12.0
```

## Author

**Anand Ramakrishnan** — [LinkedIn](https://www.linkedin.com/in/anandoiyer9/)

Built from production experience scaling multi-agent systems at Foundry (Blackstone PE), serving 50M+ monthly users.
