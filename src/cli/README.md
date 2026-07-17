# Pipeline'y końcowego eksperymentu

Pliki `run_*.py` są cienkimi entrypointami. Parsują argumenty i przekazują sterowanie do kodu wykonawczego; nie zawierają definicji matematycznych.

Najważniejsze entrypointy:

- `logit_feature_score_suite/run_csqa_logit_feature_diagnostics_pipeline.py`
  - diagnostyka cech logitowych i przebiegu śladu decyzyjnego dla pojedynczego modelu
- `logit_feature_score_suite/run_csqa_logit_feature_diagnostics_suite.py`
  - diagnostyka rodzin Qwen 2.5 i Llama 3.2
- `logit_feature_score_suite/run_csqa_logit_feature_score_control_pipeline.py`
  - pełny eksperyment interwencyjny dla pojedynczego modelu: `ascent`, `descent`, `random_same_norm`
- `logit_feature_score_suite/run_csqa_logit_feature_score_control_cap_sweep.py`
  - wspólny runner dla rodzin Qwen i Llama oraz wielu wartości capa

Podział odpowiedzialności:

- `src/score/` — cechy, KS, KDE, funkcja `score` i matematyka perturbacji;
- `logit_feature_score_suite/experiment_core.py` — wspólna sekwencja ekstrakcji i dopasowania;
- `logit_feature_score_suite/model_runtime.py` — obsługa modelu, readoutów i hooka;
- `logit_feature_score_suite/score_policy.py` — pętla wariantu ascent;
- `logit_feature_score_suite/control_policy.py` — pętla ascent/descent/random;
- `logit_feature_score_suite/intervention_experiment.py` — orkiestracja i zapis wyników;
- `logit_feature_score_suite/diagnostics_experiment.py` — orkiestracja diagnostyki.

Najkrótszą ścieżką do zrozumienia metody jest [opis warstwy badawczej](../score/README.md), a nie kod CLI.

Analizy zapisanych wyników znajdują się w katalogu `scripts/`; wszystkie dotyczą końcowego wariantu score. Skrypty generujące notebooki oraz pipeline'y wcześniejszych metod zostały usunięte.
