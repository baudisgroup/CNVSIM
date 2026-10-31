"""Within-diagnosis mutation associations and follow-up checks.

association: reconstruct eligible cohorts and test subtype associations while
adjusting for assay, global CNV burden and the tested gene's local CNV.
center_consistency/sensitivity: check centre consistency and sensitivity to mutation density
and sample type. variant_description: optional, post hoc description of selected cases;
protein references do not enter the association tests. export: format results.

GENIE data and specimen-level intermediate files are user-supplied/local only.
The CLI documents the AACR data-access terms. One JSON run record stores
paths, environment, stages, counts and exports; runtime file hashing is not used.
"""
