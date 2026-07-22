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


# Plugins etc — still open

-> it woudl be nice to keep retrain history and everything so it's well auditable somehow
-> make sure the artefacts and all the models are correctly stored in the DB and not in memory so we have persistance !


# Remaining product ideas

- We should have main parameter like logs-hisyory=true and allow-logsllm-analysis=true that will automystore entire per agent history a with all inputs and outputs and metadata - everything. ! If activated .. we can then have a analyser agent providing recommendations based on the logs per agent with scoring, summary « , status, etc !!  So we have a real live LLM based analysis based on all logs etc !
