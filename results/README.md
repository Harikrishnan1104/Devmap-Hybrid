# Results

Every number in the paper can be recomputed from the files here. The large per-row prediction
files are stored gzipped (`*.csv.gz`) to fit GitHub's limits; run `python scripts/unpack_results.py` once to unpack them. Per-row prediction files keep
one line per exam row, so models can be compared row by row (McNemar test, application bootstrap).

| Folder / file | What it holds |
|---|---|
| `colab_all/preds__*.csv` | Fine-tuned CodeBERTa and GraphCodeBERT, all four exams; 3 training runs on unseen applications |
| `colab_all/embeddings/` | Frozen CodeBERTa and GraphCodeBERT embeddings of every distinct kernel |
| `colab_phase1/embeddings/` | Newer code readers: CodeSage, CodeT5+, UniXcoder, GraphCodeBERT in pieces |
| `colab_phase2/` | TabPFN v2 as the decision model |
| `colab_q9/` | Faculty question 9: GraphCodeBERT deciding end to end, numbers fed as numbers (3 runs) |
| `colab_q10/` | Faculty point 10: Laya (open-source alternative to Jev), fine-tuned |
| `colab_phase3/agent_results.csv` | Gemini junior + senior agent (AMD complete; NVIDIA in progress) |
| `baseline_*`, `name_tag_*` | Majority guess, memoriser, Random Forest baselines; job-number name-tag test |
| `hybrid_*` | Hybrid models (frozen embeddings + counts/ratios + Random Forest) and their tests |
| `decision_makers*` | Random Forest vs XGBoost, LightGBM, SVM, kNN, MLP, logistic regression |
| `point8_*`, `merged_*`, `q9_tests.csv`, `feature_importance.csv`, `repeats_*` | Summaries and significance tests |
