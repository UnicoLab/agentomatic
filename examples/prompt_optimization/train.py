from agentomatic.agents import (
    PromptFitterBridge, OptimizeMetricAdapter,
    WeightedMetric, MetricLoss,
)
from agent import MyAgent2Agent
from agentomatic.optimize import LocalJudgeMetric, CustomMetric, PromptSearchSpace, prepare_dataset, load_data
from agentomatic.agents import AgentDataset
from agentomatic.stacks.manager import StackManager
from agentomatic.providers import apply_stack_defaults, get_llm_for_agent
from agentomatic.config.settings import load_environment
from agentomatic.optimize import TrainCliSettings, print_train_result, train_and_report
from agentomatic.optimize import PromptSearchSpace
from pathlib import Path
from rich.console import Console
from agentomatic.config.settings import load_environment
from loguru import logger


logger.info("Starting training script.")
ROOT = Path(__file__).resolve().parents[2]  
AGENT = "my_agent"
HERE = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"  # nested layouts (e.g. ai_platform/) may use ROOT.parent
console = Console()

# --- environment / stack ---
argv = []
cli = TrainCliSettings.parse(argv)

# loading stack
logger.info("Loading environment and stack.")
load_environment(ENV_PATH)
stacks = StackManager(ROOT / "stacks")
stacks.load(cli.stack)
apply_stack_defaults(stacks)
logger.info("Stack loaded and defaults applied.")


# define agent
llm = get_llm_for_agent(AGENT, role="default", stack_manager=stacks)
agent = MyAgent2Agent(llm=llm)
logger.info("Agent initialized successfully.")

# local dataset
dataset, _full_data_path = prepare_dataset(
    dataset=load_data("datasets/all.jsonl"),
    seed_path="datasets/all.jsonl",
    augment=True,
    n_examples=100,
    persist=True,
    persist_path="datasets/all_augmented.jsonl",
    model="omlx/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    llm_base_url="http://127.0.0.1:8000/v1",  # local omlx / Ollama
    llm_api_key=None,
    strategies=["expansion", "paraphrase"],  # ["paraphrase", "perturbation", "expansion"]
)
logger.info("Dataset prepared successfully.")

# dataset = AgentDataset("datasets/all.jsonl")
# logger.debug(f"{dataset=}")

# single judge metric
logger.info("Setting up judge metric.")
judge = LocalJudgeMetric(
    model="omlx/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    criteria="Is the response relevant and accurate between 0-1 scoring?",
    dimensions=["correctness", "completeness", "relevance"],  # ["correctness", "completeness", "relevance"]
    weight=1.0,
    temperature=0.7,
)
# we can ahve multi hudge (mixture of experts)
# panel = MultiJudgePanel(
#     judges=[
#         LocalJudgeMetric(name="judge_qwen", model="ollama/qwen2.5:7b"),
#         LocalJudgeMetric(name="judge_llama", model="ollama/llama3.1:8b"),
#     ],
#     aggregation="average",
# )


# This bridges the existing optimization metrics (e.g. ``LocalJudgeMetric``,
# ``LLMJudgeMetric``, ``CompositeMetric``) to the class-agent evaluation
# system, which expects a synchronous ``score(example, prediction) -> float``
# interface.
judge_m = OptimizeMetricAdapter(
    optimize_metric=judge,
    name="judge",
)
logger.info("Judge metric set up successfully.")

# agents.WeightedMetric has .score() — safe to use with MetricLoss
loss = WeightedMetric(
    [("judge", judge_m, 1)],
    name="composite_loss",
)
logger.info("Weighted metric (loss) set up successfully.")

optimizer = PromptFitterBridge(
    agent_name="test_agent",
    # defining models
    task_model="omlx/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    rewrite_model="omlx/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    # live agent injected automatically from optimize()
    llm_base_url="http://127.0.0.1:8000/v1",  # local omlx / Ollama
    llm_api_key=None,
    max_trials=8,
    metric=judge_m,
    search_space=PromptSearchSpace(
        optimize_system_prompt=True,
        optimize_user_template=False,
        optimize_model_params=False,
        optimize_few_shot=False,
    ),
    optimizer="gepa_like",
    auto_report=True,
    concurrency=4,
    min_absolute_improvement=0.001,
    patience=2,
)
logger.info("Optimizer set up successfully.")


logger.info("Compiling agent with dataset, metrics, optimizer, and loss.")
agent.compile(
    dataset,
    metrics=[judge_m],
    optimizer=optimizer,
    loss=MetricLoss(loss),
)
logger.info("Agent compiled successfully.")

logger.info("Defining callbacks")
callbacks = [
    EpochDiffCallback(epochs=epochs),
    EarlyStopping(monitor="val_loss", patience=3, mode="min"),
]

logger.info("Starting training.")
history = agent.fit(
    dataset=dataset,
    # validation_data=dataset.test,
    epochs=2,
    verbose=2,
    callbacks=callbacks,
)

logger.info("Training completed.")
logger.info(history.summary())
logger.info(history.history)   # list[float] — per-round best scores