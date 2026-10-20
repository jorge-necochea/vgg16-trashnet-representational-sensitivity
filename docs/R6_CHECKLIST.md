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
| Release | Published on the public `revision-jimaging-2026` branch. On 20 October 2026, a fresh Google Colab `git clone --depth 1` of that branch passed the released repository validator for checksums/gzip integrity, 151,620 aligned main and R1 records with formula consistency, and 30,324 transformation-manifest entries plus 822-image robustness summaries. This is not an end-to-end notebook replay. |
| Archival DOI | Optional Zenodo publication after stable release. |

**Editorial alignment:** corrected the C7 aggregate CSV to avoid duplicating direct JPG–PNG cosine, corrected R5 `eta_squared_h` nomenclature (C8), and replaced the journal supplement with the consolidated Figures S1–S8 / Tables S1–S13 version. Original notebooks remain historical.

**Outstanding:** correct/document the manifest's historical transformation-root naming difference; execute a clean dependency and notebook run; decide whether to merge this release branch into `main`. Manuscript, supplement and letter must identify the public revision branch until then. Avoid calling file-validation results a full independent rerun of all analysis notebooks.
