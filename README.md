# Two opposite failure modes in alignment-based conservation scores, one inflating and one concealing constraint, are computable before the score is used

Taner Karagöl<sup>1</sup>, Alper Karagöl<sup>2</sup>

<sup>1</sup>McGovern Institute for Brain Research, Massachusetts Institute of Technology, Cambridge, MA, USA
<sup>2</sup>Department of Mechanical Engineering, Massachusetts Institute of Technology, Cambridge, MA, USA

[![PyPI version](https://img.shields.io/pypi/v/gapcheck)](https://pypi.org/project/gapcheck/) [![Web Tool](https://img.shields.io/badge/Web%20Tool-GapCheck-0f4c5c)](https://www.karagolresearch.com/gapcheck) [![Mirror](https://img.shields.io/badge/Mirror-tanerkaragol.com-475569)](https://www.tanerkaragol.com/gapcheck) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone Python package and command-line interface ([`gapcheck`](https://pypi.org/project/gapcheck/) on PyPI) and browser-based alignment diagnostics at [karagolresearch.com/gapcheck](https://www.karagolresearch.com/gapcheck) ([mirror](https://www.tanerkaragol.com/gapcheck)).

## Abstract

Conservation scores are read as properties of an amino acid position, when each
measures that position together with a sequence set and a depth parameter, and
the same position receives opposite calls under defensible choices of the latter
two. We characterise two failure modes that push in opposite directions and show
that both are computable from the alignment before the score is used. Where
conservation is 1 − H/log₂ 21 over an alphabet whose twenty-first symbol is the
gap, a fully gapped column scores the maximum by construction. Across 55,816
columns the reported value rises with gappiness at r = +0.553, collapsing to
+0.066 when conservation is recomputed over ungapped residues, and an independent
predictor arbitrates: the gap-free recomputation tracks AlphaMissense at ρ =
0.551 versus 0.416 for the reported value. The opposite mode is clade
partitioning, in which between-subfamily divergence is read as tolerance. Varying
only the comparator set moves a position by a per-family median of 2.19 grades on
a nine-grade scale over 5,092 positions in eight families and flips 2.7 to 8.3
per cent, median 6.6, from variable to conserved. Label-permutation nulls cannot
separate this from shared ancestry, calling 68.3 to 100 per cent of variable
columns diagnostic, where a tree-aware null with per-column parsimony rate
matching returns 4.1 to 69.1 per cent. The same four syndecan genes at two
sampling depths give 4.1 and 29.8 per cent, so no proteome-wide rate is well
posed without the decomposition behind it. The resulting per-position panel
reports gap fraction, effective comparator count and the tree-corrected
subfamily-explained fraction with the direction of the expected error. It
generates a testable prediction, that variants to a sister subfamily's fixed
state are benign-enriched, which holds at an odds ratio of 2.46 (1.40 to 4.33)
resting on 87 such variants among 1,565 classified. Applying the gap correction
to a published conserved-score analysis raises the rigid score's discrimination
in the human cohort from AUC 0.832 to 0.919.

## Repository layout

```
codes/
  01_positive_control_peroxidase.py    positive control on one protein
  02_depth_ladder_peroxidase.py        depth ladder, one protein in detail
  03_gap_inflation_evcouplings.py      the gap mode across the cohort
  04_gap_survival_dynamics_aware.py    the gap correction on published scores
  05_reliability_panel.py              the per-position panel
  06_predictor_miscalibration_gap.py   the gap flag against predictor error
  07_tree_aware_null.py                the tree-aware null
  08_fetch_family.py                   family assembly from UniProt
  09_sister_state_test.py              the registered prediction
  10_depth_ladder_families.py          depth ladder across families
  11_structure_mapping.py              PDB to human numbering
  12_orthology_check.py                the criterion against Ensembl Compara
  13_clinical_auc.py                   discrimination against ClinVar labels
  build_figures.py                     every figure from data/
  fetch_structure.py                   downloads and checks PDB 7DE5
  partition.py                         shared readers, aligner, subfamily labels
  requirements.txt
data/
  *.csv                                per-position output tables
  *.txt                                the run log for each table
  families/                            assembled alignments and exclusion lists
  ensembl_cache/                       cached Ensembl Compara responses
figures/
  Figure1 to Figure6, PNG at 400 dpi
structures/
  7DE5_to_human.json                   chain A to human lactoperoxidase numbering
supplementary/
  Supplementary_Information.pdf
  Supplementary_Tables.xlsx
```

## Dataset summary

| Path | Contents | Files |
|---|---|---|
| `data/*.csv` | per-position outputs: reliability panels, tree-null runs, depth ladders, sister-state tables, clinical AUC | 49 |
| `data/*.txt` | the run log that produced each table | 35 |
| `data/families/` | assembled family alignments with their exclusion lists | 24 |
| `data/ensembl_cache/` | cached Ensembl Compara responses for the orthology check | 55 |
| `figures/` | the main figures, PNG at 400 dpi | 6 |
| `structures/` | PDB 7DE5 chain A to human lactoperoxidase residue mapping | 1 |
| `supplementary/` | Supplementary Information and the long per-position workbook | 2 |

The longest tables are the per-position reliability panels and tree-null runs,
at 2,009 rows for SCN1A and 976 for EPHA2. The workbook in `supplementary/`
holds 52,454 rows across ten sheets.

## Reproducing

Python 3.9 or later. `pip install -r codes/requirements.txt`.

```
cd codes
python3 fetch_structure.py                  # PDB 7DE5 into structures/
python3 01_positive_control_peroxidase.py
python3 02_depth_ladder_peroxidase.py
python3 03_gap_inflation_evcouplings.py
python3 04_gap_survival_dynamics_aware.py
python3 05_reliability_panel.py   --fasta ... --ref ... --own ... --subfamilies ...
python3 06_predictor_miscalibration_gap.py
python3 07_tree_aware_null.py     --fasta ... --ref ... --subfamilies ... --name ...
python3 08_fetch_family.py        --name ... --genes ... --ref-gene ... --clade ...
python3 09_sister_state_test.py
python3 10_depth_ladder_families.py --fasta ... --ref ... --own ... --tag ...
python3 11_structure_mapping.py
python3 12_orthology_check.py
python3 13_clinical_auc.py
python3 build_figures.py
```

Every script that takes arguments prints them with `--help`. Script 13 takes
none and reads its input from `CDI_CLINVAR`. Each script stops on a failed
condition instead of continuing: among them a positive control failing in either
direction, a gapped reference row that does not reconstruct its input sequence,
an alignment or depth below a stated sequence count, an orthologue set
overlapping the paralogue set, a subfamily pattern matching sequences in more
than one group, and a rate calibration that fails for most columns.

For running pre-scoring alignment diagnostics directly on any FASTA alignment
without cloning the full repository, install the standalone
[`gapcheck`](https://pypi.org/project/gapcheck/) Python package from PyPI:

```
pip install gapcheck

gapcheck -i alignment.fasta -o gapcheck_audit.csv

gapcheck -i family.fasta -o panel.csv --ref P18827 --own SDC1 \
  --subfamilies "SDC1=SDC1_;SDC2=SDC2_;SDC3=SDC3_;SDC4=SDC4_"
```

## Software dependencies

| Package | Version | Used by |
|---|---|---|
| numpy | >= 1.24 | `partition.py`, the permutation null, the tree-aware null |
| pyfamsa | >= 0.7 | `partition.py`, FAMSA multiple alignment |
| pandas | >= 2.0 | scripts 03, 04 and 06 |
| scipy | >= 1.10 | script 02, the rank correlations |
| matplotlib | >= 3.7 | `build_figures.py` |
| openpyxl | >= 3.1 | the per-position workbook |

Position mapping goes through the gapped reference row, which is checked to
reconstruct its input sequence, so no sequence parser is required.

## Data availability

Source sequences are public in UniProt. PDB 7DE5 is public at RCSB and is
downloaded by `codes/fetch_structure.py`, which checks the chain A residue range
and the file checksum against the copy used here.

Three inputs are cited in the paper rather than redistributed. Scripts 03, 04
and 06 read the EVcouplings alignment output and published score table of the
dynamics-aware cohort; point `CDI_DYNAMICS_DIR` at that folder. The optional
variant arm of script 05 reads a UniProt humsavar release; point `CDI_HUMSAVAR`
at it, or leave it unset and the arm switches off. Script 13 reads a ClinVar
export carrying `UniProt_ID`, `Residue_Pos` and `ClinVar_Significance`; point
`CDI_CLINVAR` at it. The 86 residue-level labels script 13 derives are in
`data/clinical_auc_labels.csv` and its results in `data/clinical_auc.csv`, so
that comparison can be checked without re-running the join.

## Citation

Karagöl T, Karagöl A. Two opposite failure modes in alignment-based conservation
scores, one inflating and one concealing constraint, are computable before the
score is used. bioRxiv, 2026. https://doi.org/10.64898/2026.10.20.760363

## Contact

Taner Karagöl, taner.karagol@gmail.com, karagolt@mit.edu
Alper Karagöl, alper.karagol@gmail.com, akaragol@mit.edu

## License

The code in this repository is released under the MIT License, in `LICENSE`.
Upstream data remains under the licences of UniProt, RCSB, Ensembl, ClinVar and
AlphaMissense.
