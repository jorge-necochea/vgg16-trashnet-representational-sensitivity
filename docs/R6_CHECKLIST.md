# R6: release verification status — 2026-10-18

| Requirement | Evidence / outstanding action |
|---|---|
| JPEG encoding | Historical JPG notebook uses Pillow JPEG writer with implicit quality=75; precise historical libjpeg version unrecorded. |
| Original images | Original TrashNet images are JPEG; transformed PNGs are lossless *storage derivatives* of lossy JPEG sources. |
| Software | Environment JSON files and `requirements.txt` retained; successful fresh-environment dependency installation is not yet verified. |
| Random seeds | Seed 2026 as documented in notebooks and methods. |
| Derived metrics | Main and repaired R1 each have 151,620 rows; validator now verifies all `.gz` streams and formula consistency. |
| Three-preprocessing experiment | `analysis/robustness822/` includes code, metadata, and 822-image summary tables. Full embedding matrices omitted; regenerate from sources. |
| Other historical analyses | Original notebooks under `notebooks/originals/` provide additional provenance, but an end-to-end replay remains to be demonstrated. |
| Release | The public GitHub repository must be updated with this patch and a fresh download must pass validation **before claiming completion**. |
| Archival DOI | Optional Zenodo publication after stable release. |

**Outstanding:** correct/document the manifest's historical transformation-root naming difference; execute a clean dependency and notebook run; ensure the manuscript, supplement, and response letter refer to exactly the publicly released files and values. Avoid calling file-validation results a full independent rerun of all analysis notebooks.
