# Agentomatic studion -> shoudl adapt the forms to agents inputs and outputs based on schema dynamically
-> currently we can just provide input but we do not know what agent needs in the studio which is hard for debuggin so we shoudl use what info we have for schemas etc like langgraph sdutio does to display dynamically all the required inputs etc !


# Improve fit (fix) — DONE (v1.8.3–1.8.5)

Shipped: epoch learnings + `prompt_history`, always-on generalization holdout,
sequential/default concurrency=1, post-fit drain, apply guard (refuse zero
improvement / overfit prompts unless `force=True`), richer judge context,
`TrainConfig` + `train_and_report` / HolySheet fit dashboards (score/loss
curves, Keras epoch history, prompt evolution). Covered by
`test_fit_learning_generalization`, `test_fit_optimizer_matrix`,
`test_train_api_and_holysheet`.

Historical bugs that motivated the redesign (now addressed):
- Judge weak expected_output → structural keys + richer judge context
- Score oscillation / apply-without-improvement → apply guards + min_absolute_improvement
- Connection errors after async fit → persistent event loop + drain_seconds


# Plugins / audit / logs — DONE (v1.8.8–1.8.9)

Shipped:
- Auditable retrain history via `OptimizationRunStore` + `fit_store`
  (`retrain_history.jsonl` always; SQLAlchemy DB when store bound)
- Artefacts/models persist through `SQLAlchemyStore` (Postgres / SQLite /
  any SQLAlchemy async URL) — not MemoryStore when a DB is configured
- `logs_history` / `allow_logsllm_analysis` (+ env
  `AGENTOMATIC_LOGS_HISTORY` / `AGENTOMATIC_ALLOW_LOGSLLM_ANALYSIS`) with
  REST `/logs` + `/logs/analyze`
- **1.8.9**: Multi-resource logs (plugin/pipeline/ingestion/endpoint) +
  global `/api/v1/logs?resource=&name=` + in-process pipeline step logging
- Train UX: `print_train_result` / `TrainResult.print_summary()`
Covered by `test_logs_history` + `test_logs_history_resources`.


# Docs / templates for train-eval APIs — DONE

Shipped with this docs pass: MkDocs `guide/optimization.md` + changelog,
scaffold `_train_py` / `_eval_py` (`print_train_result`, augment knobs,
`persist_fit_store`, `evaluate_and_report`), agent primer, README pointer.


# Dual-tier Keras fit surface — DONE (v1.8.11)

Staged public API (`compile_agent` / `fit_agent` / `evaluate_agent` +
`build_default_metrics`) with `train_and_report` on the same primitives.
Docs (both styles), scaffold commented staged example, tests.


# Remaining product ideas

(none from the previous logs/DB/train-UX backlog — see studio schema UX above)
