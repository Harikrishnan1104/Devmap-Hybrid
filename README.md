# Where should this kernel run?

**Leakage-aware CPU/GPU device mapping of OpenCL kernels with GraphCodeBERT and a Random Forest hybrid**

M.Tech. (AI & ML) project, Vellore Institute of Technology, by Harikrishnan J (26MAI0060).
Guide: [faculty name].

Given an OpenCL kernel and the size of the job, should it run on the CPU or the GPU? This repository
re-implements a recent Transformer study of that question (Siino et al., *EAAI* 163, 2026), shows how
much of its accuracy comes from leakage between training and test data, and evaluates new models
under a fair split in which whole applications are held out.

## What we found

| Model (unseen applications) | AMD | NVIDIA |
|---|---|---|
| Always the more common device | 58.8 | 56.9 |
| CodeBERTa, fine-tuned (base paper's best model) | 62.5 | 64.3 |
| GraphCodeBERT, fine-tuned | 72.5 | 64.5 |
| Random Forest on code counts + raw job numbers | 76.1 | 68.0 |
| **Hybrid: frozen GraphCodeBERT + counts + ratios + Random Forest** | **75.8** | **72.1** |

- The base paper's 92.8% / 89.1% reproduces (we get 92.1% / 88.8%) but relies on a random split.
  Identical kernels and application-specific job sizes sit on both sides of it; held-out
  applications bring every model down to 61-78%.
- GraphCodeBERT, not used for device mapping before, beats CodeBERTa by 10 points on AMD.
- The hybrid is the most accurate model on NVIDIA, a reliable +4.1 points over the feature-only forest.
- Seven other decision models, an end-to-end Transformer, a merged two-machine dataset, the
  open-source Jev alternative Laya and a Gemini agent did not do better.
- NPB is 77.5% of the rows and the only large suite with several input sizes per kernel, and on
  NVIDIA no model is faster in practice than always using the GPU. Report per-suite results and
  achieved speed, not only overall accuracy.

The full write-up is in `paper/DevMap_IEEE_paper_draft.docx`.

## Repository layout

```
data/       DevMap (both machines) + provenance notes
scripts/    all code: data handling, exams, models, statistics, notebook builders, tests
colab/      ready-to-run Google Colab notebooks (generated from scripts/ by build_*.py)
results/    per-row predictions, embeddings and summary tables behind every number in the paper
paper/      IEEE two-column draft, figures, and the script that draws them
```

Key modules in `scripts/`:

| File | Role |
|---|---|
| `devmap.py` | Loads DevMap, builds the four exams (random 5-fold as in the paper, unseen program, unseen application, unseen suite) |
| `train_lib.py` | The base paper's fine-tuning recipe, step for step, with resumable runs |
| `features.py` | Code counts, job numbers, scale-free ratios, Grewe et al.'s features |
| `embed_lib.py` | Frozen embeddings from any Hugging Face code model |
| `05_hybrid.py` | The proposed hybrid |
| `06_repeats.py`, `07_hybrid_tests.py` | Repeated runs, McNemar tests, application-level bootstrap |
| `08_decision_makers.py`, `12_point8_compare.py` | Random Forest vs seven other decision models |
| `10_end_to_end.py`, `16_q9_tests.py` | GraphCodeBERT deciding by itself, with a numeric branch |
| `13_merged.py`, `14_merged_finetune.py` | AMD + NVIDIA merged into one dataset |
| `15_jev_alternatives.py` | Laya and Kev, open-source alternatives to the closed Jev classifier |
| `agent_lib.py`, `09_agent_eval.py` | Random Forest junior + Gemini senior agent |
| `11_feature_importance.py` | Permutation importance, SHAP, mutual information |

## Reproducing the results

**Tree models and statistics (CPU, minutes):**

```bash
pip install -r requirements.txt
cd scripts
python 02_baselines.py                 # majority, memoriser, Random Forest on all exams
python 04_name_tag_test.py             # job-number name-tag test
python 05_hybrid.py ../results/colab_all/embeddings
python 07_hybrid_tests.py ../results/colab_all
python 08_decision_makers.py ../results/colab_all/embeddings --learners rf xgboost lightgbm
python 12_point8_compare.py
python 13_merged.py
python 16_q9_tests.py
```

**Transformers (GPU):** open the notebooks in `colab/` on Google Colab with a T4 GPU and choose
Runtime > Run all. Each notebook downloads the data, writes the tested helper code, saves every
finished fold to Google Drive and resumes after a disconnect.

| Notebook | Purpose | Approx. T4 time |
|---|---|---|
| `recreate_base_paper.ipynb` | Base paper recipe on all exams (CodeBERTa, GraphCodeBERT) | 1-2 h per model |
| `extract_embeddings.ipynb` | Frozen embeddings for the hybrid | 15-30 min |
| `phase2_tabpfn.ipynb` | TabPFN v2 decision model | 30 min |
| `faculty_q9_end_to_end.ipynb` | Transformer deciding by itself | 1.5 h |
| `faculty_q7_merged_gcb.ipynb` | Merged AMD + NVIDIA, GraphCodeBERT | 1 h per seed |
| `faculty_q10_jev_alternatives.ipynb` | Laya and Kev fine-tuned | Laya 1.3 h; Kev several hours |
| `phase3_agent.ipynb` | Gemini agent (CPU runtime; needs a Gemini API key in Colab Secrets) | free daily quota |

**Tests:** `python test_train_lib.py`, `test_hybrid.py`, `test_agent.py`, `test_e2e.py`,
`test_merged_finetune.py`, `test_jev.py` check the plumbing offline with tiny stand-in models.

Never commit API keys: the agent notebook reads the Gemini key from Colab Secrets only.

## Licence and data

The dataset comes from Cummins et al. (PACT 2017), released under GPL-3.0, via the base paper's
repository; see `data/README.md`. To stay compatible, this repository is released under GPL-3.0
(see `LICENSE`).

## Citation

```
Harikrishnan J, "Where Should This Kernel Run? A Leakage-Aware Re-evaluation of Transformer Models
for CPU-GPU Device Mapping and a GraphCodeBERT-Random Forest Hybrid," M.Tech. project,
Vellore Institute of Technology, 2026.
```

Base paper: M. Siino et al., "A transformer-based approach for source code classification for
heterogeneous device mapping," *Engineering Applications of Artificial Intelligence*, vol. 163,
112987, 2026.
