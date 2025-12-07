# Multi-Agent Evaluator-Orchestrator

A production-grade framework for building multi-agent LLM systems using the **evaluator-optimizer pattern**—where orchestrator agents delegate to specialized generators, and evaluator agents adversarially score output until convergence.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Insight

**Models are better critics than generators.**

By separating generation from evaluation and iterating to convergence, we achieve dramatically higher output quality than single-pass generation. In production at scale, this pattern reduced hallucination rates from ~15% to <2%.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR AGENT                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  • Analyzes task type and complexity                            │    │
│  │  • Selects appropriate generator model(s)                       │    │
│  │  • Manages iteration state and convergence                      │    │
│  │  • Routes to human-in-the-loop for edge cases                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│     GENERATOR AGENTS        │   │     EVALUATOR AGENTS        │
│  ┌───────────────────────┐  │   │  ┌───────────────────────┐  │
│  │ Claude-3 (Factual)    │  │   │  │ Factual Accuracy      │  │
│  │ GPT-4 (Creative)      │  │   │  │ Brand Alignment       │  │
│  │ Specialized Models    │  │   │  │ Readability Score     │  │
│  └───────────────────────┘  │   │  │ Safety/Toxicity       │  │
│                             │   │  └───────────────────────┘  │
│  Output: Raw generation     │   │  Output: Score 0-100 +     │
│                             │   │          Detailed feedback │
└──────────────┬──────────────┘   └──────────────┬──────────────┘
               │                                  │
               └────────────┬─────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │    CONVERGENCE ENGINE       │
              │  ┌───────────────────────┐  │
              │  │ loss = 100 - score    │  │
              │  │ if loss < threshold:  │  │
              │  │     return output     │  │
              │  │ else:                 │  │
              │  │     regenerate with   │  │
              │  │     feedback          │  │
              │  └───────────────────────┘  │
              │  Max iterations: 3         │
              └─────────────────────────────┘
```

## Why This Pattern?

| Single-Pass Generation | Evaluator-Optimizer Pattern |
|------------------------|----------------------------|
| One model, one attempt | Multiple specialists collaborating |
| Blind spots persist | Cross-model evaluation catches errors |
| No quality guarantee | Explicit quality gates |
| ~15% hallucination rate | <2% hallucination rate |
| Fast but unreliable | Slightly slower, production-grade |

## Installation

```bash
pip install multi-agent-evaluator-orchestrator
```

Or from source:

```bash
git clone https://github.com/anandoiyer9/multi-agent-evaluator-orchestrator.git
cd multi-agent-evaluator-orchestrator
pip install -e .
```

## Quick Start

```python
from maeo import Orchestrator, GeneratorAgent, EvaluatorAgent

# Initialize agents
orchestrator = Orchestrator(
    generators=[
        GeneratorAgent(model="claude-3-sonnet", specialty="factual"),
        GeneratorAgent(model="gpt-4", specialty="creative"),
    ],
    evaluators=[
        EvaluatorAgent(dimension="accuracy", threshold=85),
        EvaluatorAgent(dimension="brand_safety", threshold=90),
        EvaluatorAgent(dimension="readability", threshold=80),
    ],
    max_iterations=3,
    convergence_threshold=85,
)

# Generate with quality guarantees
result = orchestrator.generate(
    task="Write a product description for an enterprise AI platform",
    context={"brand_voice": "professional", "audience": "CTOs"},
)

print(f"Output: {result.content}")
print(f"Final Score: {result.score}")
print(f"Iterations: {result.iterations}")
print(f"Evaluator Feedback: {result.feedback}")
```

## Core Components

### Orchestrator

The central coordinator that manages the generation-evaluation loop:

```python
from maeo import Orchestrator

orchestrator = Orchestrator(
    generators=[...],           # List of generator agents
    evaluators=[...],           # List of evaluator agents
    max_iterations=3,           # Cap iterations to control cost
    convergence_threshold=85,   # Minimum acceptable score
    human_review_threshold=70,  # Route to human if score stays below
)
```

### Generator Agents

Specialized models for different content types:

```python
from maeo import GeneratorAgent

# Factual content generator
factual_gen = GeneratorAgent(
    model="claude-3-sonnet",
    specialty="factual",
    system_prompt="You are a precise, factual content generator...",
)

# Creative content generator
creative_gen = GeneratorAgent(
    model="gpt-4",
    specialty="creative",
    system_prompt="You are a creative, engaging content generator...",
)
```

### Evaluator Agents

Adversarial critics that score output across dimensions:

```python
from maeo import EvaluatorAgent

accuracy_eval = EvaluatorAgent(
    dimension="accuracy",
    model="claude-3-opus",  # Use strongest model for evaluation
    threshold=85,
    scoring_rubric="""
    Score 0-100 based on:
    - Factual correctness (40%)
    - Source attribution (30%)
    - No hallucinated claims (30%)
    """,
)
```

## Advanced Usage

### Custom Convergence Logic

```python
from maeo import Orchestrator, ConvergenceStrategy

class WeightedConvergence(ConvergenceStrategy):
    def should_converge(self, scores: dict) -> bool:
        weighted = (
            scores["accuracy"] * 0.4 +
            scores["brand_safety"] * 0.3 +
            scores["readability"] * 0.3
        )
        return weighted >= self.threshold

orchestrator = Orchestrator(
    convergence_strategy=WeightedConvergence(threshold=85),
    ...
)
```

### Human-in-the-Loop Integration

```python
from maeo import Orchestrator, HumanReviewHandler

def send_to_slack(content, scores, feedback):
    # Your Slack integration
    pass

orchestrator = Orchestrator(
    human_review_handler=HumanReviewHandler(
        trigger_threshold=70,  # Route to human if score < 70 after max iterations
        callback=send_to_slack,
    ),
    ...
)
```

### Streaming Output

```python
async for chunk in orchestrator.generate_stream(task="..."):
    if chunk.type == "generation":
        print(f"Generated: {chunk.content[:50]}...")
    elif chunk.type == "evaluation":
        print(f"Score: {chunk.score} - {chunk.feedback}")
    elif chunk.type == "iteration":
        print(f"Iteration {chunk.iteration}: Regenerating...")
```

## Production Learnings

Insights from running this pattern at scale:

### 1. Evaluators Should Use Different Models Than Generators

Cross-model evaluation catches blind spots. If GPT-4 generates, use Claude to evaluate (and vice versa).

### 2. Cap Iterations at 3

Diminishing returns after 3 iterations. If it hasn't converged by then, route to human review.

### 3. Evaluation Cost is Worth It

Running 2-3x inference for quality gates costs less than one piece of bad content reaching customers.

### 4. Separate Evaluation Dimensions

Don't ask one evaluator to score everything. Specialized evaluators (accuracy, safety, readability) outperform generalist scoring.

### 5. Log Everything

Every generation, evaluation, and iteration should be logged for debugging and improvement.

```python
orchestrator = Orchestrator(
    logger=StructuredLogger(
        log_generations=True,
        log_evaluations=True,
        log_iterations=True,
        destination="s3://your-bucket/logs/",
    ),
    ...
)
```

## Benchmarks

Tested on 10,000 enterprise content generation tasks:

| Metric | Single-Pass | Evaluator-Optimizer | Improvement |
|--------|-------------|---------------------|-------------|
| Hallucination Rate | 15.3% | 1.8% | **-88%** |
| Brand Alignment | 72% | 94% | **+31%** |
| Human Edit Rate | 45% | 8% | **-82%** |
| Avg. Latency | 1.2s | 3.8s | +217% |
| Cost per Output | $0.02 | $0.05 | +150% |

**Tradeoff:** 2.5x cost and 3x latency for 88% reduction in hallucinations and 82% reduction in human editing.

## Configuration

```yaml
# config.yaml
orchestrator:
  max_iterations: 3
  convergence_threshold: 85
  human_review_threshold: 70

generators:
  - model: claude-3-sonnet
    specialty: factual
    temperature: 0.3
  - model: gpt-4
    specialty: creative
    temperature: 0.7

evaluators:
  - dimension: accuracy
    model: claude-3-opus
    threshold: 85
    weight: 0.4
  - dimension: brand_safety
    model: gpt-4
    threshold: 90
    weight: 0.3
  - dimension: readability
    model: claude-3-haiku
    threshold: 80
    weight: 0.3

logging:
  level: INFO
  destination: stdout
```

## API Reference

Full API documentation: [docs/api.md](docs/api.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this pattern in your work:

```bibtex
@software{multi_agent_evaluator_orchestrator,
  author = {Ramakrishnan, Anand},
  title = {Multi-Agent Evaluator-Orchestrator: Production-Grade LLM Quality Control},
  year = {2024},
  url = {https://github.com/anandoiyer9/multi-agent-evaluator-orchestrator}
}
```

## Author

**Anand Ramakrishnan**  
AI Product & Engineering Executive | Multi-Agent Orchestration, Agentic Systems  
[LinkedIn](https://www.linkedin.com/in/anandoiyer9/) | [GitHub](https://github.com/anandoiyer9)

---

*Built from production experience scaling multi-agent systems at Foundry (Blackstone PE).*
