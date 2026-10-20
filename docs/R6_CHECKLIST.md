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
| Release | The 20 October package was built and validated **locally**; public GitHub publication and clean-download validation still require author action. |
| Archival DOI | Optional Zenodo publication after stable release. |

**Editorial alignment:** corrected the C7 aggregate CSV to avoid duplicating direct JPG–PNG cosine, corrected R5 `eta_squared_h` nomenclature (C8), and replaced the journal supplement with the consolidated Figures S1–S8 / Tables S1–S13 version. Original notebooks remain historical.

**Outstanding:** correct/document the manifest's historical transformation-root naming difference; execute a clean dependency and notebook run; ensure the manuscript, supplement, and response letter refer to exactly the publicly released files and values. Avoid calling file-validation results a full independent rerun of all analysis notebooks.
