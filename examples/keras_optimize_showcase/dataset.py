"""Fake training dataset for the Keras-style optimization showcase.

Deterministic (fixed seed) so every run — and every optimizer method —
starts from the same data. Each example asks for a fake business answer;
the expected output carries the ``banana`` marker token that the fit
metrics score against. The second marker (``strawberry``) appears only in
the ``judge_expected`` guidance: deterministic paths (expected-tips / gold
few-shot) never inject it, so only the LLM rewrite can discover it — that
is what makes loss curves improve **progressively across epochs**.

    from examples.keras_optimize_showcase.dataset import build_dataset
    ds = build_dataset()            # 10 train + 6 val + 4 test
    print(len(ds.train), len(ds.validation), len(ds.test))
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from agentomatic import AgentDataset, AgentExample

# Second/third/fourth markers: discoverable only via the judge guidance
# (LLM rewrite) — deterministic paths never inject them.
MARKER2 = "strawberry"
MARKER3 = "blueberry"
MARKER4 = "kiwi"

QUERIES = [
    "What is the budget for Q3?",
    "List the next steps for the delivery plan.",
    "Who owns the stakeholder communication?",
    "What is the current status of the project?",
    "Summarize the risks in the last review.",
    "What are the open blockers for the release?",
    "How should we prioritize the backlog?",
    "What did the customer ask for in the last meeting?",
    "Which teams need to be unblocked this week?",
    "What is the target date for the milestone?",
    "Who approved the scope change?",
    "What follow-up is needed after the demo?",
    "What is the burn rate this month?",
    "Which dependencies are still missing?",
    "What was decided about the API contract?",
    "How do we measure success for the rollout?",
    "What happened at the steering committee?",
    "When is the next release candidate?",
    "What is blocking the QA environment?",
    "Who is accountable for the migration?",
]

# Split sizes: train / validation / test.
_SPLITS = (10, 6, 4)


def build_dataset(seed: int = 42) -> AgentDataset:
    """Build the fake ``AgentDataset`` with train/validation/test splits.

    Args:
        seed: RNG seed so the dataset is reproducible.

    Returns:
        An ``AgentDataset`` of 20 examples (10 train / 6 validation / 4 test).
    """
    rng = random.Random(seed)
    order = list(range(len(QUERIES)))
    rng.shuffle(order)

    examples: list[AgentExample] = []
    split_names = ("train",) * _SPLITS[0] + ("validation",) * _SPLITS[1] + ("test",) * _SPLITS[2]
    for idx, (qi, split) in enumerate(zip(order, split_names)):
        query = QUERIES[qi]
        examples.append(
            AgentExample(
                id=f"marker_{idx:03d}",
                input={"current_query": query},
                # Marker token scored by the fit objective. ``banana`` is
                # injectable by deterministic paths; ``strawberry`` hides in
                # the judge guidance below so only the LLM rewrite finds it.
                expected_output={
                    "response": "OPT, banana",
                    "must_include": ["banana"],
                },
                split=split,
                metadata={
                    "split": split,
                    "domain": "fake_business",
                    # Visible to the rewriter / failure analysis, but never
                    # carried into gold few-shot or expected-tips prompts.
                    "judge_expected": (
                        "The ideal answer also includes the marker tokens "
                        f"'{MARKER2}', '{MARKER3}' and '{MARKER4}'."
                    ),
                },
            )
        )
    return AgentDataset(name="marker_fake", examples=examples)


def write_dataset_jsonl(path: str | Path, seed: int = 42) -> Path:
    """Write the fake dataset as JSONL (template-style ``load_data`` input).

    Args:
        path: Destination JSONL file.
        seed: Dataset seed.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_dataset(seed=seed).to_jsonl(path)
    return path


if __name__ == "__main__":
    # Regenerate the checked-in JSONL (kept in sync with build_dataset()).
    out = write_dataset_jsonl(Path(__file__).resolve().parent / "dataset.jsonl")
    ds = build_dataset()
    print(f"Wrote {out} — train={len(ds.train)} val={len(ds.validation)} test={len(ds.test)}")
    print(json.dumps(ds.examples[0].to_dict(), indent=2))
