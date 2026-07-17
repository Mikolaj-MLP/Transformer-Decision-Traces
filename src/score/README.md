# Warstwa badawcza eksperymentu score

Ten katalog zawiera część, którą można czytać niezależnie od obsługi modeli Hugging Face, batchingu i zapisu plików.

- `features.py` — definicje entropii, luki top-1/top-2 oraz varentropy;
- `statistics.py` — statystyka KS i wybór warstw;
- `density.py` — estymacja `p(x | correct)`, `p(x | incorrect)` oraz funkcji `s(x)`;
- `intervention.py` — gradient `s(f(h))`, ograniczenie normy perturbacji i wariant losowy;
- `constants.py` — parametry końcowego eksperymentu.

Kod uruchamiający modele, przygotowujący batche i zapisujący Parquet pozostaje w `src/cli/logit_feature_score_suite`. Dzięki temu definicje metodologiczne nie są wymieszane z infrastrukturą eksperymentu.
