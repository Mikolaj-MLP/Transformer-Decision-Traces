# Current CLI Surface

Active scripts in this directory:

- `extract_csqa_trace_feature_tables.py`
  - one-shot raw trace-feature extraction for `csqa` or `aqua_rat`
- `run_csqa_logit_feature_steering_pipeline.py`
  - end-to-end univariate logit-feature detector + feature-targeted intervention pipeline for `csqa` or `aqua_rat`
- `run_csqa_adaptive_contrastive_pipeline.py`
  - end-to-end detector + adaptive contrastive intervention pipeline
- `run_csqa_logit_feature_steering_pipeline_qwen25_suite.py`
  - no-arg Qwen2.5 CSQA suite (`0.5B,3B,7B`)
- `run_csqa_logit_feature_steering_pipeline_llama32_suite.py`
  - no-arg Llama 3.2 CSQA suite (`1B,3B`)
- `run_aqua_rat_logit_feature_steering_pipeline_qwen25_suite.py`
  - no-arg Qwen2.5 AQuA-RAT suite (`1.5B,3B,7B`)
- `run_aqua_rat_logit_feature_steering_pipeline_llama32_suite.py`
  - no-arg Llama 3.2 AQuA-RAT suite (`1B,3B`)

Older exploratory and generic extraction CLIs were moved to `src/legacy/cli/`.
