# Experimentos CatBoost

Los experimentos y sus corridas quedan bajo `runs/`. Usar el corte temporal, las métricas y las reglas de ID establecidas en `../../NON_NEGOTIABLE.md`.

Ejecutar el pipeline desde la raíz del proyecto con `uv run datafest run`. Cada corrida conserva `manifest.json`, `metrics.json`, predicciones de validación, submission, modelo de validación y modelo final serializados. `experiments/index.json` enlaza las corridas.

Verificar una corrida con `uv run datafest verify --manifest <ruta-al-manifest.json>`.
