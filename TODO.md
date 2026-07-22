# Agentomatic studion -> shoudl adapt the forms to agents inputs and outputs based on schema dynamically
-> currently we can just provide input but we do not know what agent needs in the studio which is hard for debuggin so we shoudl use what info we have for schemas etc like langgraph sdutio does to display dynamically all the required inputs etc !


# Improve fit (fix)

+ imrpove this fitting mechanism much much further to make it great and fully functional for any optimizer and any metric etc -> make sure it's well passing all info that can help write new prompt and eveluate everything correctly, keep history and learning to guide new prompt generation across several epochs progressively ... So make sure we have everything in place to make it work fully, then test it ! Make it great in terms of UX and displayed info (like keras fit), show progress, loss evolution and curve at the end etc ! + let's make sure that usage of processes or threads for each model is well controlled cause I can see 3 threads for LM2.5 and one for qwen3 and some are not used  .. looks like during the train we use one thread of lm2.5 and one qwen but on lm2.5 I se several 305 thredas with generating part ... but always one active and streaming ! Well when I run the train script it looks liek we are not imrpoving at all, which is surprising to see ... cause new prompt shoudl give better results is we provide sufficient data and context and hhistory etc !!! So Deep dive in all the parts and make sure the are well connected and passing everything nicely correctly ! We will need to make sure we provide enough signal at each epoch and learnings, and synthesis what was correct and what was not so that next prompt rewrite could be based on these learning, examples, previosu evolutions etc ... this means we are progressively building a context and learning and everything to know how to imrpove the prompt and we are measuring it etc ... and also SLM judge shoudl rpovide extensive motivation and justification for it's scoring so that we can learn and improve ! So redesing this implementation (adding additional validaiton, steps, context, storage etc ...) everything you need to make it actually work ! We are free to desing any architecture here that will work, can be multi pass where we preselect candidates condensate learning and run another pass by LLM etc ... we just need to be carefull for budges if using LLMs throug APIs. 

There are still some bugs to be fixed for fitting process but we need it to work perfectly and be a great tested and productiont ready building block of this package ! So make sure it works perfectly !

We also have some recent learning:
Judge gives 0.33 consistently: The val example as_next_steps has expected_output: {"next_action": True} → converted to "Response must include: 'next_action'" — useless as a quality reference. The judge has nothing to compare against.
Score oscillates 0.33↔0.67: Agent runs at temp=0.1, non-deterministic — different response each run, different judge score.
Apply-without-improvement bug: fit_result.apply() is called even when absolute_improvement = 0.0.
Connection errors: No drain time after the async optimization loop.

looks liek we need to give much more data to the judge and to the prompt optimizer as well.. Maybe async optimization is not the best option and sequential will be more controllable ! We can parallelize what we can but let's make sure we have everything implemented correctly !

# Plugins etc

-> it woudl be nice to keep retrain history and everything so it's well auditable somehow
-> make sure the artefacts and all the models are correctly stored in the DB and not in memory so we have persistance !


- fit should avoid overfitting when optimizing prompts (tends to overfit to respond to provided data examples )- we need to add generalization safety net always
- Prompts history with learnings at each epoch so we can trace the evolution and all learnings of optimization etc ! No
- We should have main parameter like logs-hisyory=true and allow-logsllm-analysis=true that will automystore entire per agent history a with all inputs and outputs and metadata - everything. ! If activated .. we can then have a analyser agent providing recommendations based on the logs per agent with scoring, summary « , status, etc !!  So we have a real live LLM based analysis based on all logs etc !
