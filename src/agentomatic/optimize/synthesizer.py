"""Synthetic dataset generation and augmentation engine.

Generates evaluation datasets from:
- Agent descriptions / manifests
- Seed Q/A pairs (augmentation)
- Domain descriptions
- Existing prompts

Augmentation strategies:
- Paraphrase: Rephrase queries while preserving intent
- Perturbation: Add typos, informal language, edge cases
- Expansion: Generate related questions from seed data
- Adversarial: Generate tricky/ambiguous queries
- Cross-domain: Generate queries from adjacent topics

All generation uses LLM calls — configurable model.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from agentomatic.optimize.dataset import DataPoint, Dataset


class DataSynthesizer:
    """Synthetic dataset generator and augmenter.

    Example::

        synth = DataSynthesizer(model="ollama/mistral:7b")

        # Generate from description
        dataset = await synth.generate(
            description="HR assistant that answers policy questions",
            n_samples=50,
            categories=["leave", "benefits", "expenses"],
        )

        # Augment existing dataset
        augmented = await synth.augment(
            dataset=existing_dataset,
            strategies=["paraphrase", "perturbation", "adversarial"],
            multiplier=3,
        )

    Args:
        model: LLM model for generation (e.g., "ollama/mistral:7b").
        temperature: Generation temperature (higher = more diverse).
        api_base: Ollama/LLM API base URL.
    """

    def __init__(
        self,
        model: str = "ollama/mistral:7b",
        temperature: float = 0.8,
        api_base: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.api_base = api_base

    # =================================================================
    # Generate from scratch
    # =================================================================

    async def generate(
        self,
        description: str,
        n_samples: int = 30,
        categories: list[str] | None = None,
        difficulty_levels: list[str] | None = None,
        include_context: bool = False,
    ) -> Dataset:
        """Generate a synthetic dataset from a description.

        Args:
            description: What the agent does (e.g., "HR policy assistant").
            n_samples: Number of Q/A pairs to generate.
            categories: Optional topic categories to cover.
            difficulty_levels: e.g., ["easy", "medium", "hard"].
            include_context: Generate context documents too.

        Returns:
            Dataset with generated DataPoints.
        """
        difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        batch_size = min(n_samples, 10)
        all_points: list[DataPoint] = []

        # Generate in batches for reliability
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            remaining = n_samples - len(all_points)
            if remaining <= 0:
                break

            current_batch_size = min(batch_size, remaining)

            prompt = self._build_generation_prompt(
                description=description,
                n=current_batch_size,
                categories=categories,
                difficulty=difficulty_levels[batch_idx % len(difficulty_levels)],
                include_context=include_context,
                existing_queries=[p.query for p in all_points[-5:]],
            )

            batch_points = await self._generate_batch(prompt)
            all_points.extend(batch_points)
            logger.debug(
                f"Generated batch {batch_idx + 1}/{n_batches}: "
                f"{len(batch_points)} points (total: {len(all_points)})"
            )

        dataset = Dataset(points=all_points[:n_samples])
        logger.info(f"🧪 Generated {len(dataset)} synthetic data points")
        return dataset

    async def generate_from_prompt(
        self,
        system_prompt: str,
        n_samples: int = 20,
    ) -> Dataset:
        """Generate evaluation data from an existing system prompt.

        Analyzes the prompt to understand what the agent does,
        then generates Q/A pairs that test its capabilities.
        """
        prompt = (
            f"You are a QA dataset generator. Analyze this system prompt and "
            f"generate {n_samples} diverse question-answer pairs that would "
            f"thoroughly test an agent using this prompt.\n\n"
            f"## System Prompt\n```\n{system_prompt}\n```\n\n"
            f"## Rules\n"
            f"- Cover different aspects mentioned in the prompt\n"
            f"- Include easy, medium, and hard questions\n"
            f"- Include edge cases and ambiguous queries\n"
            f"- Make answers realistic and detailed\n"
            f"- Vary question styles (direct, conversational, complex)\n\n"
            f"Reply with a JSON array: "
            f'[{{"query": "...", "expected_answer": "...", "difficulty": "easy|medium|hard"}}]\n'
        )
        return Dataset(points=await self._generate_batch(prompt))

    # =================================================================
    # Augmentation
    # =================================================================

    async def augment(
        self,
        dataset: Dataset,
        strategies: list[str] | None = None,
        multiplier: int = 3,
    ) -> Dataset:
        """Augment an existing dataset with synthetic variations.

        Args:
            dataset: Original dataset to augment.
            strategies: Augmentation strategies to apply.
                Options: paraphrase, perturbation, expansion,
                adversarial, formality_shift.
            multiplier: How many variations per original point.

        Returns:
            New Dataset with original + augmented points.
        """
        strategies = strategies or ["paraphrase", "perturbation", "expansion"]
        all_points = list(dataset.points)  # Keep originals

        strategy_map = {
            "paraphrase": self._augment_paraphrase,
            "perturbation": self._augment_perturbation,
            "expansion": self._augment_expansion,
            "adversarial": self._augment_adversarial,
            "formality_shift": self._augment_formality,
        }

        for strategy_name in strategies:
            fn = strategy_map.get(strategy_name)
            if fn is None:
                logger.warning(
                    f"Unknown augmentation strategy: '{strategy_name}'. "
                    f"Available: {list(strategy_map.keys())}"
                )
                continue

            per_strategy = max(1, multiplier // len(strategies))
            augmented = await fn(dataset.points, per_strategy)
            all_points.extend(augmented)
            logger.debug(
                f"Augmentation '{strategy_name}': +{len(augmented)} points"
            )

        result = Dataset(points=all_points)
        logger.info(
            f"🔄 Augmented: {len(dataset)} → {len(result)} points "
            f"(+{len(result) - len(dataset)} synthetic)"
        )
        return result

    # =================================================================
    # Augmentation strategies
    # =================================================================

    async def _augment_paraphrase(
        self, points: list[DataPoint], n_per: int
    ) -> list[DataPoint]:
        """Rephrase queries while preserving intent."""
        prompt = self._build_augmentation_prompt(
            points=points,
            n_per=n_per,
            instruction=(
                "Paraphrase each query in different ways while keeping the SAME intent.\n"
                "Vary: sentence structure, vocabulary, formality level, question style.\n"
                "The expected answer should remain valid for each paraphrase."
            ),
        )
        return await self._generate_batch(prompt)

    async def _augment_perturbation(
        self, points: list[DataPoint], n_per: int
    ) -> list[DataPoint]:
        """Add realistic noise: typos, informal language, abbreviations."""
        prompt = self._build_augmentation_prompt(
            points=points,
            n_per=n_per,
            instruction=(
                "Create NOISY variations of each query:\n"
                "- Add realistic typos and misspellings\n"
                "- Use informal/casual language\n"
                "- Use abbreviations and shorthand\n"
                "- Use broken grammar or sentence fragments\n"
                "- Mix languages if relevant\n"
                "The expected answer should still be the same."
            ),
        )
        return await self._generate_batch(prompt)

    async def _augment_expansion(
        self, points: list[DataPoint], n_per: int
    ) -> list[DataPoint]:
        """Generate related follow-up questions from seed data."""
        prompt = self._build_augmentation_prompt(
            points=points,
            n_per=n_per,
            instruction=(
                "For each query, generate RELATED follow-up questions:\n"
                "- Deeper questions about the same topic\n"
                "- Adjacent/related topics\n"
                "- 'What if' scenarios\n"
                "- Comparative questions\n"
                "Provide realistic expected answers."
            ),
        )
        return await self._generate_batch(prompt)

    async def _augment_adversarial(
        self, points: list[DataPoint], n_per: int
    ) -> list[DataPoint]:
        """Generate tricky, edge-case, and ambiguous queries."""
        prompt = self._build_augmentation_prompt(
            points=points,
            n_per=n_per,
            instruction=(
                "Create ADVERSARIAL variations designed to test edge cases:\n"
                "- Ambiguous queries with multiple valid interpretations\n"
                "- Questions with trick assumptions\n"
                "- Queries that combine multiple topics\n"
                "- Questions about exceptions to rules\n"
                "- Very long or very short queries\n"
                "- Questions with negation\n"
                "Provide the CORRECT expected answer for each."
            ),
        )
        return await self._generate_batch(prompt)

    async def _augment_formality(
        self, points: list[DataPoint], n_per: int
    ) -> list[DataPoint]:
        """Shift formality: casual ↔ professional."""
        prompt = self._build_augmentation_prompt(
            points=points,
            n_per=n_per,
            instruction=(
                "Rewrite each query at DIFFERENT formality levels:\n"
                "- Very casual / slang\n"
                "- Standard conversational\n"
                "- Formal / professional\n"
                "- Official / legal tone\n"
                "The expected answer should remain the same."
            ),
        )
        return await self._generate_batch(prompt)

    # =================================================================
    # Internal helpers
    # =================================================================

    def _build_generation_prompt(
        self,
        description: str,
        n: int,
        categories: list[str] | None,
        difficulty: str,
        include_context: bool,
        existing_queries: list[str],
    ) -> str:
        """Build the prompt for dataset generation."""
        parts = [
            f"You are a QA dataset generator. Generate {n} diverse "
            f"question-answer pairs for the following agent:\n\n"
            f"## Agent Description\n{description}\n",
        ]

        if categories:
            parts.append(
                f"\n## Categories to Cover\n"
                + "\n".join(f"- {c}" for c in categories)
            )

        parts.append(f"\n## Difficulty Level: {difficulty}\n")

        if existing_queries:
            parts.append(
                f"\n## Already Generated (DO NOT REPEAT)\n"
                + "\n".join(f"- {q}" for q in existing_queries)
            )

        context_field = ', "context": ["relevant source document"]' if include_context else ""

        parts.append(
            f"\n## Output Format\n"
            f"Reply with ONLY a JSON array:\n"
            f'[{{"query": "...", "expected_answer": "..."{context_field}}}]\n'
            f"\n## Rules\n"
            f"- Make questions realistic and diverse\n"
            f"- Answers should be factual and helpful\n"
            f"- Do NOT repeat any existing queries\n"
            f"- Vary question styles and lengths\n"
        )

        return "\n".join(parts)

    def _build_augmentation_prompt(
        self,
        points: list[DataPoint],
        n_per: int,
        instruction: str,
    ) -> str:
        """Build prompt for augmentation."""
        # Sample up to 10 seed points
        seed_points = points[:10]
        seed_json = json.dumps(
            [
                {"query": p.query, "expected_answer": p.expected_answer or ""}
                for p in seed_points
            ],
            indent=2,
        )

        return (
            f"You are a dataset augmentation expert.\n\n"
            f"## Instruction\n{instruction}\n\n"
            f"## Seed Data\n```json\n{seed_json}\n```\n\n"
            f"Generate {n_per} variations PER seed query "
            f"(total: {n_per * len(seed_points)} new points).\n\n"
            f"Reply with ONLY a JSON array:\n"
            f'[{{"query": "...", "expected_answer": "..."}}]\n'
        )

    async def _generate_batch(self, prompt: str) -> list[DataPoint]:
        """Call LLM and parse the JSON response into DataPoints."""
        raw = await self._call_llm(prompt)
        return self._parse_response(raw)

    def _parse_response(self, text: str) -> list[DataPoint]:
        """Parse LLM response into DataPoints."""
        if not text:
            return []

        # Try to extract JSON array from the response
        points: list[DataPoint] = []
        try:
            # Find JSON array in the text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                json_str = text[start : end + 1]
                items = json.loads(json_str)

                for item in items:
                    if isinstance(item, dict) and "query" in item:
                        points.append(
                            DataPoint(
                                query=item["query"],
                                expected_answer=item.get("expected_answer"),
                                context=item.get("context", []),
                                metadata={
                                    k: v
                                    for k, v in item.items()
                                    if k not in ("query", "expected_answer", "context")
                                },
                            )
                        )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"Failed to parse LLM response as JSON: {exc}")
            # Try line-by-line parsing as fallback
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{") and "query" in line:
                    try:
                        item = json.loads(line)
                        points.append(
                            DataPoint(
                                query=item["query"],
                                expected_answer=item.get("expected_answer"),
                            )
                        )
                    except json.JSONDecodeError:
                        continue

        return points

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        import httpx

        model_name = (
            self.model.replace("ollama/", "")
            if self.model.startswith("ollama/")
            else self.model
        )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "")
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")

        return ""


# =====================================================================
# Convenience functions
# =====================================================================


async def generate_dataset(
    description: str,
    n_samples: int = 30,
    model: str = "ollama/mistral:7b",
    categories: list[str] | None = None,
    **kwargs: Any,
) -> Dataset:
    """Convenience function to generate a synthetic dataset.

    Example::

        dataset = await generate_dataset(
            description="Customer support bot for an e-commerce platform",
            n_samples=50,
            categories=["orders", "returns", "shipping"],
        )
        dataset.to_jsonl("eval_data.jsonl")
    """
    synth = DataSynthesizer(model=model)
    return await synth.generate(
        description=description,
        n_samples=n_samples,
        categories=categories,
        **kwargs,
    )


async def augment_dataset(
    dataset: Dataset,
    strategies: list[str] | None = None,
    multiplier: int = 3,
    model: str = "ollama/mistral:7b",
) -> Dataset:
    """Convenience function to augment a dataset.

    Example::

        original = Dataset.from_jsonl("seed_qa.jsonl")
        augmented = await augment_dataset(
            original,
            strategies=["paraphrase", "adversarial"],
            multiplier=5,
        )
        augmented.to_jsonl("augmented_qa.jsonl")
    """
    synth = DataSynthesizer(model=model)
    return await synth.augment(
        dataset=dataset,
        strategies=strategies,
        multiplier=multiplier,
    )
