"""Shared plumbing for the prompt-optimization examples.

Every example accepts the same connection flags (or environment variables),
so you point them all at your model server once:

=====================  ===========================  ================================
Flag                   Environment variable         Default
=====================  ===========================  ================================
``--base-url``         ``OMLX_BASE_URL``            ``http://127.0.0.1:8000/v1``
``--api-key``          ``OMLX_API_KEY``             ``local``
``--model``            ``AGENTOMATIC_LOCAL_MODEL``  ``Qwen3.5-9B-MLX-4bit``
``--judge-model``      ``EXAMPLE_JUDGE_MODEL``      same as ``--model``
``--rewrite-model``    ``EXAMPLE_REWRITE_MODEL``    same as ``--model``
``--out-dir``          ``EXAMPLE_OUT_DIR``          ``examples/prompt_optimization/out``
``--log-level``        ``AGENTOMATIC_LOG_LEVEL``    ``INFO``
=====================  ===========================  ================================

Any OpenAI-compatible server works (oMLX, llama.cpp, vLLM, LM Studio, Ollama's
``/v1``): only ``--base-url`` and ``--model`` change. Model *specs* passed to
the optimizer are ``"<provider>/<model>"`` strings — ``omlx/…`` routes to the
OpenAI-compatible ``--base-url``; ``openai/…``, ``ollama/…``, ``gemini/…`` and
``litellm/…`` are the other providers.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATASET = HERE / "datasets" / "support.jsonl"

# Let ``python examples/prompt_optimization/<script>.py`` import ``agent`` and
# ``common`` whatever the current directory is.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@dataclass
class ExampleSettings:
    """Connection + output settings shared by every example."""

    base_url: str
    api_key: str
    model: str
    judge_model: str
    rewrite_model: str
    out_dir: Path
    log_level: str

    # ``provider/model`` specs understood by the optimizer's LLM caller.
    @property
    def task_spec(self) -> str:
        """Spec of the model the agent itself runs on."""
        return f"omlx/{self.model}"

    @property
    def judge_spec(self) -> str:
        """Spec of the LLM-as-judge model."""
        return f"omlx/{self.judge_model}"

    @property
    def rewrite_spec(self) -> str:
        """Spec of the model that proposes improved prompts."""
        return f"omlx/{self.rewrite_model}"


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return an argument parser pre-loaded with the shared connection flags.

    Args:
        description: The example's one-line description (shown in ``--help``).

    Returns:
        A parser; examples add their own flags before calling ``parse_args``.
    """
    parser = argparse.ArgumentParser(description=description)
    model = os.getenv("AGENTOMATIC_LOCAL_MODEL", "Qwen3.5-9B-MLX-4bit")
    parser.add_argument(
        "--base-url", default=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument("--api-key", default=os.getenv("OMLX_API_KEY") or "local")
    parser.add_argument("--model", default=model)
    parser.add_argument("--judge-model", default=os.getenv("EXAMPLE_JUDGE_MODEL"))
    parser.add_argument("--rewrite-model", default=os.getenv("EXAMPLE_REWRITE_MODEL"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path(os.getenv("EXAMPLE_OUT_DIR", HERE / "out"))
    )
    parser.add_argument("--log-level", default=os.getenv("AGENTOMATIC_LOG_LEVEL", "INFO"))
    return parser


def settings_from(args: argparse.Namespace) -> ExampleSettings:
    """Build :class:`ExampleSettings` and make the endpoint the process default.

    ``LLMCaller.configure`` routes every optimizer-side call (judge, rewrite,
    augmentation) that does not pass its own ``base_url`` to this server.

    Args:
        args: Parsed arguments from :func:`build_parser`.

    Returns:
        The resolved settings.
    """
    from agentomatic.optimize.llm_caller import LLMCaller

    settings = ExampleSettings(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        judge_model=args.judge_model or args.model,
        rewrite_model=args.rewrite_model or args.model,
        out_dir=Path(args.out_dir),
        log_level=args.log_level,
    )
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    LLMCaller.configure(base_url=settings.base_url, api_key=settings.api_key)
    os.environ.setdefault("OMLX_BASE_URL", settings.base_url)
    os.environ.setdefault("OMLX_API_KEY", settings.api_key)
    return settings


def setup_logging(level: str) -> None:
    """Readable single-line logs (the library default is DEBUG-chatty).

    Args:
        level: Loguru level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    from agentomatic.core.lifespan import configure_logging

    configure_logging(level.upper())


def make_llm(settings: ExampleSettings, *, temperature: float = 0.2) -> Any:
    """Build the chat model the agent runs on.

    Args:
        settings: Connection settings.
        temperature: Sampling temperature for the agent's answers.

    Returns:
        A LangChain chat model talking to ``settings.base_url``.
    """
    from agentomatic.providers import get_named_llm

    return get_named_llm(
        f"example:{settings.model}:{temperature}",
        provider="omlx",
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=temperature,
    )


def must_include_score(example: Any, prediction: dict[str, Any]) -> float:
    """Deterministic metric: fraction of the example's key facts in the answer.

    Reads ``example.metadata["must_include"]`` (see ``datasets/support.jsonl``).
    Use it with ``agentomatic.agents.CallableMetric``.

    Args:
        example: The ``AgentExample`` being scored.
        prediction: The agent's output dict.

    Returns:
        A score in ``[0, 1]``.
    """
    facts = list((getattr(example, "metadata", None) or {}).get("must_include") or [])
    if not facts:
        return 0.0
    answer = str(prediction.get("response", "")).lower()
    return sum(1 for fact in facts if fact.lower() in answer) / len(facts)
