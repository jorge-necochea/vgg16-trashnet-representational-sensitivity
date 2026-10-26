# Technical editorial follow-up — 20 October 2026

Manuscript: `jimaging-4632369` (*Journal of Imaging*).

This document records the patch prepared against the 18 October snapshot. Subsequent publication of the patch to the `revision-jimaging-2026` branch and independent verification using a fresh Colab clone **were completed on 20 October 2026**. The validation establishes published-file integrity and key derived-data consistency, not end-to-end notebook re-execution.

## Changes, no recomputation

- **C7:** the direct JPG–PNG cosine (0.9985 pooled mean; 0.9949 lowest of 60 cell means) appears only in the cosine row of `data/summary_tables/storage_format_aggregate_summary.csv`. The renamed columns identify the averaging unit; the minimum is **not** the individual-pair minimum. The primary metric means and significance counts are preserved.
- **C8:** retained the three exact Kruskal–Wallis effect-size values and renamed their CSV column `eta_squared_h`. The maintained R5 notebook uses the same name and computes `max(0, (H-k+1)/(n-k))`. Historical original notebooks are unmodified.
- **Supplement:** replaced the previously published S1–S10 Word supplement with the consolidated S1–S13 Word source. Its source tables for S11–S13 are in `analysis/robustness822/`.
- Updated README, data dictionary, reproducibility notes, R6 checklist and repository validator.
- Regenerated `checksums.sha256` for all repository files (manifest excluded).

## Publication protocol (historical steps; completed for the current revision branch)

1. Open the target branch (previously `revision-jimaging-2026`) and **verify its contents match the 18 October integrated snapshot** before copying this patch. If not, use a local clone and merge the small changes selectively; do not delete other work.
2. Upload each file from the patch preserving directories; replace the pre-existing files, without deleting unrelated files.
3. Run the repository validator locally or in Google Colab on a clean checkout/download.
4. Independently check public access and confirm the supplemental Word contains S11–S13.
5. Update the manuscript Data Availability URL and editor-response R6 status only after public-branch verification, ideally using a stable release or commit permalink. Do not claim a full end-to-end rerun without doing one.

**This patch does not include trained weights, original TrashNet images or full embedding matrices.**

## Main-branch publication follow-up — 26 October 2026

The historical upload procedure above was completed. Pull request #1 merged the four editorial commits from `revision-jimaging-2026` into the public default branch `main` on 20 October 2026 (merge commit `f510196`). A fresh checkout of `main` passed the repository's integrity and selected derived-data consistency checks on that date. The public repository tree, including the consolidated S1–S13 supplement, was inspected again on 26 October 2026.

This follow-up updates the Data Availability URL, the R6 and resubmission checklists, the README and the historical upload guide; checksums are refreshed. No experimental measurements, embeddings, statistical results, notebooks or figure assets were changed in this documentation-only commit. The full end-to-end computational replay remains unverified.
