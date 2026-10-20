#!/usr/bin/env python3

import csv
import gzip
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SCORES = os.path.join(DATA, "gap_survival_scores.csv.gz")
LABELS_OUT = os.path.join(DATA, "clinical_auc_labels.csv")
TABLE_OUT = os.path.join(DATA, "clinical_auc.csv")
LOG_OUT = os.path.join(DATA, "clinical_auc_log.txt")

PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
VARIANTS = ["A_as_published", "B_gap_filter", "C_gap_corrected", "D_both"]


BOOTSTRAP = 50000
SEED = 11
STABILITY_BOOTSTRAP = 2000
STABILITY_SEEDS = 30

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def fatal(msg):
    raise SystemExit("fatal: " + msg)


def auc(pairs):

    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    srt = sorted(pos + neg)
    ranks = {}
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[srt[i]] = (i + j) / 2.0 + 1
        i = j + 1
    r = sum(ranks[s] for s in pos)
    return (r - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def paired_interval(labelled, a_scores, b_scores, boot=None, seed=None):

    rnd = random.Random(SEED if seed is None else seed)
    keys = list(labelled)
    out = []
    for _ in range(BOOTSTRAP if boot is None else boot):
        smp = [keys[rnd.randrange(len(keys))] for _ in keys]
        pa = [(a_scores[k], labelled[k]) for k in smp]
        pb = [(b_scores[k], labelled[k]) for k in smp]
        x, y = auc(pa), auc(pb)
        if x is not None and y is not None:
            out.append(y - x)
    if not out:
        return None, None, None
    out.sort()
    obs_a = auc([(a_scores[k], labelled[k]) for k in keys])
    obs_b = auc([(b_scores[k], labelled[k]) for k in keys])
    if obs_a is None or obs_b is None:
        return None, None, None
    return obs_b - obs_a, out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def interval(pairs):

    rnd = random.Random(SEED)
    out = []
    for _ in range(BOOTSTRAP):
        sample = [pairs[rnd.randrange(len(pairs))] for _ in pairs]
        a = auc(sample)
        if a is not None:
            out.append(a)
    if not out:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main():
    src = os.environ.get("CDI_CLINVAR")
    if not src:
        fatal("CDI_CLINVAR is not set; point it at the ClinVar export "
              "described in the module docstring")
    if not os.path.exists(src):
        fatal("CDI_CLINVAR points at %s, which does not exist" % src)

    calls = defaultdict(set)
    seen_values = set()
    with open(src, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for col in ("UniProt_ID", "Residue_Pos", "ClinVar_Significance"):
            if col not in (rdr.fieldnames or []):
                fatal("%s has no %s column" % (src, col))
        for row in rdr:
            sig = (row["ClinVar_Significance"] or "").strip()
            seen_values.add(sig)
            try:
                pos = int(float(row["Residue_Pos"]))
            except (TypeError, ValueError):
                continue
            key = (row["UniProt_ID"], pos)
            if sig in PATHOGENIC:
                calls[key].add(1)
            elif sig in BENIGN:
                calls[key].add(0)

    labels = {k: v.pop() for k, v in calls.items() if len(v) == 1}
    conflicting = sum(1 for v in calls.values() if len(v) > 1)
    if not labels:
        fatal("no residue carries an unambiguous classification; the values "
              "present were %s" % sorted(seen_values))
    log("clinical labels from %s" % os.path.basename(src))
    log("  residues labelled: %d  (pathogenic %d, benign %d)"
        % (len(labels), sum(labels.values()), len(labels) - sum(labels.values())))
    log("  residues dropped as conflicting: %d" % conflicting)

    scores = defaultdict(dict)
    with gzip.open(SCORES, "rt") as fh:
        for row in csv.DictReader(fh):
            if row["cohort"] != "data-human":
                continue
            key = (row["UniProt_ID"], int(float(row["i"])))
            if key in labels:
                scores[row["variant"]][key] = (float(row["rcs10"]),
                                               float(row["dcs10"]))
    missing = [v for v in VARIANTS if v not in scores]
    if missing:
        fatal("the score table carries no rows for %s" % ", ".join(missing))
    covered = set()
    for d in scores.values():
        covered |= set(d)
    if not covered:
        fatal("no labelled residue appears in %s" % SCORES)
    log("  labelled residues found in the score table: %d" % len(covered))

    rows = []
    for variant in VARIANTS:
        for name, idx in (("RCS", 0), ("DCS", 1)):
            pairs = [(vals[idx], labels[k]) for k, vals in scores[variant].items()]
            npos = sum(y for _, y in pairs)
            if npos == 0 or npos == len(pairs):
                fatal("variant %s leaves one class empty at %d residues"
                      % (variant, len(pairs)))
            a = auc(pairs)
            lo, hi = interval(pairs)
            rows.append({"variant": variant, "score": name,
                         "auc": "%.6f" % a, "ci_low": "%.6f" % lo,
                         "ci_high": "%.6f" % hi, "n_residues": len(pairs),
                         "n_pathogenic": npos, "n_benign": len(pairs) - npos})
            log("  %-16s %-4s AUC %.3f  95%% CI %.3f to %.3f  n=%d"
                % (variant, name, a, lo, hi, len(pairs)))


    shared = sorted(set(scores["A_as_published"]) & set(scores["C_gap_corrected"]))
    if not shared:
        fatal("the as-published and gap-free variants share no residue, so the "
              "paired difference cannot be formed")
    lab_shared = {k: labels[k] for k in shared}
    a_rcs = {k: scores["A_as_published"][k][0] for k in shared}
    c_rcs = {k: scores["C_gap_corrected"][k][0] for k in shared}
    d, dlo, dhi = paired_interval(lab_shared, a_rcs, c_rcs)
    excludes = "yes" if (dlo > 0 or dhi < 0) else "no"
    log("  paired difference, gap-free minus as published, over %d shared "
        "residues: %+.3f  95%% CI %+.3f to %+.3f  (excludes zero: %s)"
        % (len(shared), d, dlo, dhi, excludes))
    if dlo <= 0:
        fatal("the paired lower bound is %+.6f, so the manuscript sentence "
              "that it excludes zero is false" % dlo)


    sweep = [paired_interval(lab_shared, a_rcs, c_rcs,
                             boot=STABILITY_BOOTSTRAP, seed=sd)[1]
             for sd in range(1, STABILITY_SEEDS + 1)]
    log("  that lower bound at %d replicates across %d seeds: %+.3f to %+.3f "
        "(contains zero: %s), which is why the reported count is %d"
        % (STABILITY_BOOTSTRAP, STABILITY_SEEDS, min(sweep), max(sweep),
           "yes" if min(sweep) <= 0 <= max(sweep) else "no", BOOTSTRAP))
    if not min(sweep) <= 0 <= max(sweep):
        fatal("the sweep no longer straddles zero, so the Methods sentence "
              "justifying %d replicates no longer describes this run"
              % BOOTSTRAP)
    rows.append({"variant": "C_minus_A_paired", "score": "RCS",
                 "auc": "%.6f" % d, "ci_low": "%.6f" % dlo,
                 "ci_high": "%.6f" % dhi, "n_residues": len(shared),
                 "n_pathogenic": sum(lab_shared.values()),
                 "n_benign": len(shared) - sum(lab_shared.values())})

    with open(TABLE_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(LABELS_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["UniProt_ID", "residue_position", "clinical_label"])
        for (acc, pos), y in sorted(labels.items()):
            w.writerow([acc, pos, "pathogenic" if y else "benign"])
    with open(LOG_OUT, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    log("wrote %s, %s and %s"
        % tuple(os.path.basename(p) for p in (TABLE_OUT, LABELS_OUT, LOG_OUT)))


if __name__ == "__main__":
    main()
