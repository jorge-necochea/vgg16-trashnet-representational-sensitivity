# Reproducibility files for the 2026 VGG-16 revision

This directory contains the code, controls, and reported results for the robustness and addendum analyses of the 2,527-image VGG-16 and TrashNet study.

| Directory | Contents |
| --- | --- |
| [`revision_robustez_v2/`](revision_robustez_v2/) | Input validation, robustness analysis, and [results](revision_robustez_v2/run_v2/). The results include the adjusted Friedman and Wilcoxon tests and the indices for 2,000 bootstrap replicates and 200 balanced samples. |
| [`adenda1_colab_v2/`](adenda1_colab_v2/) | GPU extraction script, [Colab instructions](adenda1_colab_v2/INSTRUCCIONES_COLAB.md), and local tests. |
| [`extraction_control/`](extraction_control/) | Pixel and embedding checks, gamma calibration, rotation controls, extraction log, and the manifest for the audited full extraction. |
| [`analisis_cpu_v2/`](analisis_cpu_v2/) | CPU analysis script, [execution instructions](analisis_cpu_v2/INSTRUCCIONES_ANALISIS.md), and tests. |
| [`analisis_adenda_v2/`](analisis_adenda_v2/) | [Results](analisis_adenda_v2/19_redaccion_interpretaciones.md) and supporting CSV tables for gamma, rotation, zoom, pre-GAP sparsity, and the pretrained ImageNet head. |

## Verify the results

From this directory, run the following commands in a shell with `sha256sum`.

```sh
sha256sum -c checksums.sha256
(cd revision_robustez_v2/run_v2 &&
  gzip -dc bootstrap_indices.npy.part-*.gz > bootstrap_indices.npy &&
  sha256sum -c checksums_outputs.sha256)
(cd analisis_adenda_v2 && sha256sum -c checksums_outputs.sha256)
(cd extraction_control && awk '$2 !~ /^conjuntos\//' H_checksums.sha256 | sha256sum -c -)
```

The `gzip` command reconstructs the exact audited bootstrap file from the numbered parts. The last command checks the 12 public extraction records. The complete extraction manifest also records the SHA-256 of the 35 NPZ files under `conjuntos/`. Those files and the original TrashNet images are separate analysis inputs.

## What GitHub alone verifies and what needs the external files

GitHub is enough to verify the published tables. Repeating the real CPU analysis requires the audited extraction kept on Google Drive, whose `extraction_control/H_checksums.sha256` has SHA-256 `409ff94360f15f08e7ad79468863bfb740be04ac57bb020be74cd97a24b770e4`.

From this repository alone, you can:

- check all files in this folder with `checksums.sha256`, rebuild the audited bootstrap file (SHA-256 `c2f882d2113cb86bcdcfae60d2dc3d9701494c3e7733fc96483ba8b4cecd8ff5`), and check the 28 robustness outputs, the 23 addendum outputs, and 12 of the 47 extraction records;
- run the tests that need no study data: `python analisis_cpu_v2/tests/test_analisis.py` and `python adenda1_colab_v2/tests/test_local.py --repo <clean checkout of 6672c2c>`.

The following inputs are not stored in GitHub. The audited copies are kept by the author on Google Drive; new copies can be regenerated from the public TrashNet dataset (see [`docs/DATA_AVAILABILITY.md`](../../docs/DATA_AVAILABILITY.md)):

- the 35 extraction NPZ files under `conjuntos/` (about 1 GB), whose SHA-256 values are listed in `extraction_control/H_checksums.sha256`;
- the original and 12 transformed reference embeddings, used by `run_robustness_v2.py`, the GPU extraction, and `analisis_cpu_v2/tests/regresion_v2.py`;
- the original TrashNet JPEG files and the transformed PNG files.

Without these files, the tables can be checked for integrity but not recomputed. `analyze_adenda_v2.py` accepts only the audited extraction: with any other `H_checksums.sha256`, it stops with exit code 2.

## Rerun the analyses

Create a clean checkout of the base commit for the `--repo` argument. The extraction script checks this commit and the tracked files it uses.

```sh
# Run from the root of this repository checkout.
git worktree add --detach ../vgg16-baseline 6672c2c7e4b84a5b0381de0d70476a97438223e5
```

1. Use the [base reproducibility guide](../../docs/REPRODUCIBILITY.md) and notebook 02 to prepare the original and 12 transformed reference embeddings. Run `revision_robustez_v2/run_robustness_v2.py` with `--emb-root`, `--repo`, and a new `--out` directory.
2. Follow the [GPU extraction guide](adenda1_colab_v2/INSTRUCCIONES_COLAB.md) with the original JPEG files, the transformed PNG files, and the reference embeddings. Adapt the example Drive paths to your own storage.
3. Reconstruct the bootstrap file with the command above. Use `analisis_cpu_v2/analyze_adenda_v2.py` with `--v2-package` pointing to `revision_robustez_v2/run_v2/` in this checkout. Supply the complete audited extraction for `--extraction`, the clean base checkout for `--repo`, and a new `--out` directory. The CPU script is pinned to the recorded extraction manifest hash. A fresh extraction has its own provenance and must be validated before being used in a new analysis.

The study uses frozen robustness criteria v1.3 and addendum v1.2. Their SHA-256 identifiers, environment details, and input hashes are recorded in the result manifests.
