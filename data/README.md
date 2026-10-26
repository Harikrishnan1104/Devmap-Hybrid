# DevMap dataset

`dataset-devmap-amd.csv` and `dataset-devmap-nvidia.csv` are the DevMap device-mapping dataset of
Cummins et al., *End-to-end deep learning of optimization heuristics*, PACT 2017
(original release: https://github.com/ChrisCummins/paper-end2end-dl, files `cgo17-amd.csv` and
`cgo17-nvidia.csv`, GPL-3.0), in the copy distributed by the base paper,
Siino et al., EAAI 163 (2026), https://github.com/marco-siino/GM_SOURCE_CODE (`AMD_DS/`, `NVIDIA_DS/`).

`scripts/00_get_data.py` downloads the same two files again if you prefer to fetch them yourself.

| | AMD file | NVIDIA file |
|---|---|---|
| Hardware | Intel i7-3820 CPU vs AMD Radeon HD 7970 | Intel i7-3820 CPU vs NVIDIA GTX 970 |
| Rows | 680 | 680 |
| GPU faster | 280 (41.2%) | 387 (56.9%) |

Both files hold the same 680 rows (256 kernels, 249 distinct sources, 71 applications, 7 suites);
only the GPU run time and the label differ.

**Checked against the 2017 release:** all numbers and all labels are identical in every row. The
kernel source differs in 184 rows (27%), where the base paper's copy adds missing helper-function
declarations or restores field accesses such as `r.x` that the original lost.

Columns: `benchmark`, `dataset` (input size name), `comp`, `rational`, `mem`, `localmem`,
`coalesced` (counted separately, can exceed `mem`), `atomic` (always 0), `transfer` (bytes copied),
`wgsize` (work-group size), `oracle` (faster device = label), `runtime_cpu`, `runtime_gpu` (never
used as inputs), `src` (kernel source, identifiers anonymised), `seq` (truncated in every row; unused),
plus two leftover row-number columns.
