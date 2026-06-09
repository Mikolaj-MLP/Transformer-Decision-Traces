# Current CLI Surface

Active scripts in this directory:

- `extract_csqa_trace_feature_tables.py`
  - one-shot raw trace-feature extraction for `csqa`
- `run_csqa_logit_feature_steering_pipeline.py`
  - end-to-end univariate logit-feature detector + feature-targeted intervention pipeline for `csqa`
- `run_csqa_adaptive_contrastive_pipeline.py`
  - end-to-end detector + adaptive contrastive intervention pipeline
- `run_csqa_logit_feature_steering_pipeline_qwen25_suite.py`
  - no-arg Qwen2.5 CSQA suite (`0.5B,3B,7B`)
- `run_csqa_logit_feature_steering_pipeline_llama32_suite.py`
  - no-arg Llama 3.2 CSQA suite (`1B,3B`)
- `run_csqa_tuned_lens_gap_steering_oneoff.py`
  - one-off CSQA experiment: tuned-lens readout, single feature (`top1-top2 logit gap`), same detector/control pipeline

Older exploratory and generic extraction CLIs were moved to `src/legacy/cli/`.
