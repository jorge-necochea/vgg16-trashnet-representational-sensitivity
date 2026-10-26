# R6: release verification status — 2026-10-20

| Requirement | Evidence / outstanding action |
|---|---|
| JPEG encoding | Historical JPG notebook uses Pillow JPEG writer with implicit quality=75; precise historical libjpeg version unrecorded. |
| Original images | Original TrashNet images are JPEG; transformed PNGs are lossless *storage derivatives* of lossy JPEG sources. |
| Software | Environment JSON files and `requirements.txt` retained; successful fresh-environment dependency installation is not yet verified. |
| Random seeds | Seed 2026 as documented in notebooks and methods. |
| Derived metrics | Main and repaired R1 each have 151,620 rows; validator now verifies all `.gz` streams and formula consistency. |
| Three-preprocessing experiment | `analysis/robustness822/` includes code, metadata, and 822-image summary tables. Full embedding matrices omitted; regenerate from sources. |
| Other historical analyses | Original notebooks under `notebooks/originals/` provide additional provenance, but an end-to-end replay remains to be demonstrated. |
| Release | Four-commit PR #1 was merged into public `main` on 20 October 2026 (merge commit `f510196`). A fresh Google Colab clone of `main` passed the released validator: checksums/gzip integrity; 151,620 aligned main and R1 records with adjusted-sensitivity formula consistency; and 30,324 transformation-manifest entries with 822-image robustness summaries. These checks do not constitute clean end-to-end notebook execution. Documentation-only updates dated 26 October 2026 harmonize the published release notes. |
| Archival DOI | Optional Zenodo publication after stable release. |

**Editorial alignment:** corrected the C7 aggregate CSV to avoid duplicating direct JPG–PNG cosine, corrected R5 `eta_squared_h` nomenclature (C8), and replaced the journal supplement with the consolidated Figures S1–S8 / Tables S1–S13 version. Original notebooks remain historical.

**Outstanding scientific/reproducibility limitations:** the manifest's historical transformation-root naming discrepancy must be mapped in the runtime; a fresh dependency installation and complete notebook replay have not been demonstrated; the exact historical JPEG library version is unknown; and original images and complete embedding matrices are not redistributed. The manuscript formatting and editorial resubmission are separate pending steps. Do not present file-integrity validation as an independent rerun of the experiments.
