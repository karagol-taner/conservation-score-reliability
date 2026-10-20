#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import partition as P


SUBFAMILIES = {
    "LPO": ("Lactoperoxidase",),
    "TPO": ("Thyroid peroxidase",),
    "EPX": ("Eosinophil peroxidase",),
    "MPO": ("Myeloperoxidase",),
}
OWN = "LPO"


MIN_SUBFAMILY_TOTAL = 17
MIN_OWN_TOTAL = 6


FUNCTIONAL = {
    225: "heme b, covalent ester", 226: "active site, proton acceptor",
    227: "Ca2+ ligand", 301: "Ca2+ ligand", 303: "Ca2+ ligand",
    305: "Ca2+ ligand", 307: "Ca2+ ligand",
    372: "transition state stabilizer", 375: "heme b, covalent ester",
    468: "heme b, axial Fe ligand",
}
DESIGNED = (144, 648, 205, 313)
HELD_BACK = (479, 169)
NATURAL = {414: "R->Q rs8178355", 514: "R->Q rs8178401", 700: "D->N rs8178412"}


DIAGNOSTIC_REQUIRED = 144
SEED = 20261007


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta",
                    default=P.local_data("an_peroxidase_PF03098.fasta"))
    ap.add_argument("--target", default=P.local_data("P22079.fasta"))
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    ap.add_argument("--n-perm", type=int, default=P.N_PERM)
    ap.add_argument("--sweep-n-perm", type=int, default=1000,
                    help="permutations for the all-position sweep, where the "
                         "cost is 712 columns rather than 19")
    a = ap.parse_args()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("01_positive_control_peroxidase.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Subfamily-explained fraction at the pI-code positions of human "
        "lactoperoxidase.")
    log("Statistic: mutual information between residue identity and subfamily "
        "label,")
    log("normalised by column entropy, chance-corrected against a label "
        "permutation null.")
    log("")
    log("thresholds: H_MIN=%.2f bits  SEF_HI=%.2f  SEF_LO=%.2f  P_MAX=%.3f"
        % (P.H_MIN, P.SEF_HI, P.SEF_LO, P.P_MAX))
    log("            MIN_SUBFAMILY_N=%d  MAX_GAP_FRAC=%.2f  n_perm=%d  seed=%d"
        % (P.MIN_SUBFAMILY_N, P.MAX_GAP_FRAC, a.n_perm, SEED))
    log("            inversion: family identity < %.2f and own-subfamily "
        "identity >= %.2f" % (P.INV_FAM_MAX, P.INV_OWN_MIN))


    fam = P.read_fasta(a.fasta)
    ref_header = next((k for k in fam if "PERL_HUMAN" in k), None)
    if ref_header is None:
        raise P.Fatal("PERL_HUMAN not found in %s" % a.fasta)
    ref_seq = fam[ref_header]

    tgt = P.read_fasta(a.target)
    design_input = next(iter(tgt.values()))
    if design_input != ref_seq:
        raise P.Fatal("PERL_HUMAN in the family FASTA differs from the target "
                      "FASTA; the two downloads disagree and the position "
                      "numbering cannot be trusted")
    log("\nhuman LPO %d residues, precursor numbering throughout" % len(ref_seq))
    log("family FASTA %d sequences" % len(fam))


    labels, unlabelled = P.assign_labels(fam, SUBFAMILIES)
    if ref_header not in labels or labels[ref_header] != OWN:
        raise P.Fatal("the reference did not label as %s; subfamily assignment "
                      "is wrong" % OWN)
    sizes = Counter(labels.values())
    log("\nsubfamily assignment over the mammalian-type peroxidases")
    for lab in sorted(sizes):
        log("  %-4s %2d sequences" % (lab, sizes[lab]))
    log("  %d sequences in the family carry no mammalian-type label and are "
        "excluded" % len(unlabelled))

    total = sum(sizes.values())
    if total < MIN_SUBFAMILY_TOTAL:
        raise P.Fatal("subfamily alignment has %d sequences, below the stated "
                      "minimum of %d" % (total, MIN_SUBFAMILY_TOTAL))
    if sizes[OWN] < MIN_OWN_TOTAL:
        raise P.Fatal("orthologue set has %d sequences, below the stated "
                      "minimum of %d" % (sizes[OWN], MIN_OWN_TOTAL))

    own_headers = [h for h, l in labels.items() if l == OWN]
    para_headers = [h for h, l in labels.items() if l != OWN]
    P.check_disjoint(own_headers, para_headers)
    log("  orthologue set %d, paralogue set %d, disjoint"
        % (len(own_headers), len(para_headers)))


    sub = {k: v for k, v in fam.items() if k in labels}
    log("\naligning %d sequences with FAMSA under a UPGMA guide tree" % len(sub))
    msa = P.align(sub)
    width = len(next(iter(msa.values())))
    log("  alignment width %d columns" % width)
    pos_to_col = P.map_positions(msa, ref_header, ref_seq)
    log("  gapped reference row reconstructs its input sequence exactly")

    rng = np.random.default_rng(SEED)

    def measure(pos, n_perm):
        col = pos_to_col[pos]
        wt = ref_seq[pos - 1]
        res, labs, gap_frac, dropped = P.column_set(msa, col, ref_header, labels)
        stat = P.column_partition(res, labs, rng, n_perm=n_perm)
        call = P.classify(stat)
        f_fam, n_fam, f_own, n_own = P.identities(
            msa, col, ref_header, wt, labels, OWN)
        row = dict(position=pos, wt=wt, gap_frac=round(gap_frac, 3),
                   dropped_subfamilies=";".join(dropped),
                   f_family=f_fam, n_family=n_fam,
                   f_own=f_own, n_own=n_own,
                   inverted=P.inverted(f_fam, f_own),
                   call=call, distribution=P.distribution(res))
        for k in ("n", "h_total", "mi", "sef", "sef_null", "sef_adj", "z",
                  "p_perm"):
            v = stat[k]
            row[k] = round(v, 4) if isinstance(v, float) else v
        return row

    def show(rows, header):
        log("\n" + header)
        log("  pos wt  H(X)    sef   sef_adj  p_perm   fam id   own id   "
            "call          distribution")
        for r in rows:
            log("  %-4d %s  %5s  %5s  %6s  %6s  %7s  %7s   %-13s %s"
                % (r["position"], r["wt"], r["h_total"], r["sef"],
                   r["sef_adj"], r["p_perm"], r["f_family"], r["f_own"],
                   r["call"], r["distribution"][:46]))


    control_rows = [measure(p, a.n_perm) for p in sorted(FUNCTIONAL)]
    show(control_rows, "positive control: the ten annotated catalytic and "
                       "cofactor residues")
    not_conserved = [r["position"] for r in control_rows
                     if r["call"] != "conserved" or r["f_own"] != 1.0]
    if not_conserved:
        raise P.Fatal(
            "positive control FAILED: positions %s are not invariant at every "
            "depth. The conservation measure is not working and no number from "
            "this run may be used." % not_conserved)
    called_diagnostic = [r["position"] for r in control_rows
                         if r["call"] == "diagnostic"]
    if called_diagnostic:
        raise P.Fatal(
            "positive control FAILED: invariant positions %s were called "
            "diagnostic. An invariant column carries no variation for the "
            "subfamily label to explain, so the statistic is finding structure "
            "in noise." % called_diagnostic)
    log("\n  control PASS: all ten invariant in the subfamily and in the "
        "orthologues,")
    log("  and none called diagnostic.")


    designed_rows = [measure(p, a.n_perm) for p in DESIGNED]
    show(designed_rows, "the four designed positions, in the order the design "
                        "applies them")
    d144 = next(r for r in designed_rows if r["position"] == DIAGNOSTIC_REQUIRED)
    if d144["call"] != "diagnostic":
        raise P.Fatal(
            "positive control FAILED: D%d came out %r rather than diagnostic "
            "(sef=%s, sef_adj=%s, p=%s). This is the position the "
            "lactoperoxidase result turns on, so the statistic does not "
            "reproduce the effect it was built to measure."
            % (DIAGNOSTIC_REQUIRED, d144["call"], d144["sef"],
               d144["sef_adj"], d144["p_perm"]))
    log("\n  D%d PASS: diagnostic at sef=%s, chance-corrected %s, p=%s, "
        "while the" % (DIAGNOSTIC_REQUIRED, d144["sef"], d144["sef_adj"],
                       d144["p_perm"]))
    log("  family-depth identity of %s would read as tolerance."
        % d144["f_family"])

    other_rows = ([measure(p, a.n_perm) for p in HELD_BACK]
                  + [measure(p, a.n_perm) for p in sorted(NATURAL)])
    show(other_rows, "held back by the three-paralogue test, and the natural "
                     "polymorphisms")


    log("\nsweeping all %d positions of the human sequence at n_perm=%d"
        % (len(ref_seq), a.sweep_n_perm))
    sweep = []
    for pos in range(1, len(ref_seq) + 1):
        r = measure(pos, a.sweep_n_perm)
        r["annotation"] = FUNCTIONAL.get(pos, NATURAL.get(pos, ""))
        r["set"] = ("functional" if pos in FUNCTIONAL else
                    "designed" if pos in DESIGNED else
                    "held_back" if pos in HELD_BACK else
                    "natural_variant" if pos in NATURAL else "")
        sweep.append(r)

    usable = [r for r in sweep if r["gap_frac"] <= P.MAX_GAP_FRAC]
    calls = Counter(r["call"] for r in usable)
    inv = [r for r in usable if r["inverted"]]
    log("  %d of %d positions pass the gap filter at MAX_GAP_FRAC=%.2f"
        % (len(usable), len(sweep), P.MAX_GAP_FRAC))
    for c in ("conserved", "diagnostic", "tolerant", "indeterminate"):
        log("    %-14s %4d  %5.1f%%"
            % (c, calls[c], 100.0 * calls[c] / len(usable) if usable else 0.0))
    log("  inversions (family identity < %.2f, own-subfamily identity >= %.2f): "
        "%d of %d usable, %.1f%%"
        % (P.INV_FAM_MAX, P.INV_OWN_MIN, len(inv), len(usable),
           100.0 * len(inv) / len(usable) if usable else 0.0))
    variable = [r for r in usable if r["call"] != "conserved"]
    if variable:
        d = sum(1 for r in variable if r["call"] == "diagnostic")
        log("  among the %d variable positions, %d are diagnostic, %.1f%%"
            % (len(variable), d, 100.0 * d / len(variable)))


    os.makedirs(a.outdir, exist_ok=True)
    reported = control_rows + designed_rows + other_rows
    cols = ["position", "wt", "set", "annotation", "n", "gap_frac",
            "dropped_subfamilies", "h_total", "mi", "sef", "sef_null",
            "sef_adj", "z", "p_perm", "n_perm", "f_family", "n_family",
            "f_own", "n_own", "inverted", "call", "distribution"]
    for r in reported:
        r.setdefault("annotation", FUNCTIONAL.get(r["position"],
                                                  NATURAL.get(r["position"], "")))
        r.setdefault("set", "functional" if r["position"] in FUNCTIONAL else
                     "designed" if r["position"] in DESIGNED else
                     "held_back" if r["position"] in HELD_BACK else
                     "natural_variant")

    def write(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        log("wrote %s (%d rows)" % (path, len(rows)))

    write(os.path.join(a.outdir, "positive_control_peroxidase.csv"), reported)
    write(os.path.join(a.outdir, "positive_control_sweep.csv"), sweep)

    logp = os.path.join(a.outdir, "positive_control_log.txt")
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        sys.exit("FATAL: %s" % exc)
