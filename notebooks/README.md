# Notebooki końcowego eksperymentu

Repozytorium zawiera wyłącznie notebooki dotyczące diagnostyki cech logitowych i interwencji opartej na funkcji `score`:

- `intervention/score_suite/inspect_score_diagnostics_runs.ipynb` — diagnostyka przebiegu cech, ich rozkładów oraz funkcji `score`;
- `intervention/score_suite/inspect_score_intervention_runs.ipynb` — główne porównania wyników interwencji i kontroli;
- `intervention/score_suite/inspect_score_intervention_layers.ipynb` — zależność efektów od warstwy i jej względnej głębokości.

Notebooki zawierają wyłącznie agregacje i wykresy odpowiadające pytaniom badawczym. Wspólne wczytywanie oraz porządkowanie eksportów znajduje się w `src/analysis`, dzięki czemu ta sama kilkusetwierszowa komórka nie jest powielana w każdym notebooku.

Eksporty są wyszukiwane w katalogu danych obok repozytorium albo w ścieżce wskazanej zmienną `TRANSFORMER_DECISION_TRACES_DATA`.
