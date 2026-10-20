# Conservation score reliability

Code and data for *Two opposite failure modes in alignment-based conservation
scores, one inflating and one concealing constraint, are computable before the
score is used*, Karagöl and Karagöl.

A conservation score is a measurement of a position, a sequence set and a depth
parameter. Treating only the first as real produces two failure modes with
opposite signs. Gap inflation makes a position read as more conserved than it
is, because a measure that treats the gap as the twenty-first symbol of the
alphabet scores absence of evidence as constraint. Clade partitioning makes a
position read as more variable than it is, because divergence between
paralogous subfamilies is counted as tolerance within the query's own clade.

## Layout

```
codes/        analysis scripts, the figure builder and the control verifier
data/         per-position output tables, run logs, family alignments
figures/      every figure at 400 dpi PNG and SVG
structures/   PDB 7DE5 and its mapping to human lactoperoxidase numbering
supplementary/ the long per-position tables as a workbook
```

## Running it

Python 3.9 or later. `pip install -r codes/requirements.txt`.

```
cd codes
python3 01_positive_control_peroxidase.py   # positive control, halts if it fails
python3 02_depth_ladder_peroxidase.py       # depth ladder, one protein in detail
python3 03_gap_inflation_evcouplings.py     # the gap mode across the cohort
python3 04_gap_survival_dynamics_aware.py   # the gap correction on published scores
python3 05_reliability_panel.py   --fasta ... --ref ... --own ... --subfamilies ...
python3 06_predictor_miscalibration_gap.py  # does the gap flag predict the error
python3 07_tree_aware_null.py     --fasta ... --ref ... --subfamilies ... --name ...
python3 08_fetch_family.py        --name ... --genes ... --ref-gene ... --clade ...
python3 09_sister_state_test.py             # the registered prediction, pooled
python3 10_depth_ladder_families.py --fasta ... --ref ... --own ... --tag ...
python3 11_structure_mapping.py             # PDB to human numbering
python3 12_orthology_check.py               # the criterion against Ensembl Compara
python3 13_clinical_auc.py                  # AUC against ClinVar labels
python3 build_figures.py                    # every figure from data/
python3 verify_positive_control.py          # the control, independent code path
```

Every script that takes arguments prints them with `--help`. Script 13 takes
none and reads its input from `CDI_CLINVAR`. Every output table in `data/` is
accompanied by the run log that produced it.

Scripts 03, 04 and 06 read the EVcouplings alignment output and published score
table of the dynamics-aware cohort, which are not part of this repository and
are cited in the manuscript. Point `CDI_DYNAMICS_DIR` at that folder. The
optional variant arm of script 05 reads a UniProt humsavar release; point
`CDI_HUMSAVAR` at it, or leave it unset and the arm switches off. Script 13
reads a ClinVar export carrying `UniProt_ID`, `Residue_Pos` and
`ClinVar_Significance`; point `CDI_CLINVAR` at it. ClinVar is not
redistributed here, but the 86 residue-level labels script 13 derives are in
`data/clinical_auc_labels.csv` and its results in `data/clinical_auc.csv`, so
that comparison can be checked without re-running the join.

## Verification

`verify_positive_control.py` recomputes the positive control from the source
alignment without importing the implementation of the statistic it checks; it
shares only the FASTA reader, the aligner and the subfamily labels from
`partition.py`. It builds its own contingency tables and recomputes entropy
and mutual information directly, so an error in the statistic cannot pass its
own check.
It runs 26 checks and exits non-zero on any mismatch.

```
python3 verify_positive_control.py
```

Every number quoted in the paper is checked against these deposited tables by
the same method, and every run log in `data/` records the values the run was
given.

## Halting conditions

Every script stops on a failed condition instead of warning and continuing.
Among them: a positive control failing in either direction, a gapped reference
row that does not reconstruct its input sequence, an alignment or depth below a
stated sequence count, an orthologue set overlapping the paralogue set, a
subfamily pattern matching sequences in more than one group, a rate calibration
that fails for most columns, and the generic depth ladder failing to reproduce
the detailed implementation on the family both cover.

## Orthology criterion

Orthologues are reviewed UniProt entries whose primary gene name equals the
requested symbol, case-insensitively, within a stated clade. This is weaker
than reciprocal best hits or tree reconciliation and cannot establish
one-to-one orthology. The manuscript states what the criterion can and cannot
establish, and quantifies its consequence against Ensembl Compara.

## Data not held here

The EVcouplings alignment output and published score table of the
dynamics-aware cohort, AlphaMissense scores, and the UniProt humsavar release
are all cited in the manuscript rather than redistributed here. Source
sequences are public in UniProt.
