#!/usr/bin/env python3

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np

import partition as P

SUBFAMILIES = {
    "LPO": ("Lactoperoxidase",), "TPO": ("Thyroid peroxidase",),
    "EPX": ("Eosinophil peroxidase",), "MPO": ("Myeloperoxidase",),
}
OWN = "LPO"
CONTROLS = (225, 226, 227, 301, 303, 305, 307, 372, 375, 468)

checks = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = got is not None and abs(float(got) - float(want)) <= tol
    checks.append((name, ok))
    print("%-4s %-54s got=%-34s want=%s"
          % ("PASS" if ok else "FAIL", name, got, want))
    return ok


def h_bits(labels_or_residues):

    v = np.array(list(Counter(labels_or_residues).values()), dtype=float)
    if v.sum() <= 0:
        return 0.0
    p = v / v.sum()
    return float(-(p * np.log2(p)).sum())


def mi_bits(residues, labels):

    res = sorted(set(residues))
    lab = sorted(set(labels))
    tab = np.zeros((len(res), len(lab)), dtype=float)
    for r, l in zip(residues, labels):
        tab[res.index(r), lab.index(l)] += 1.0
    n = tab.sum()
    pxy = tab / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pxy * np.log2(pxy / (px * py))
    return float(np.nansum(np.where(pxy > 0, term, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta",
                    default=P.local_data("an_peroxidase_PF03098.fasta"))
    a = ap.parse_args()

    fam = P.read_fasta(a.fasta)
    check("family FASTA sequence count", len(fam), 80)

    ref = next(k for k in fam if "PERL_HUMAN" in k)
    ref_seq = fam[ref]
    check("human LPO length, precursor numbering", len(ref_seq), 712)

    labels, unlabelled = P.assign_labels(fam, SUBFAMILIES)
    sizes = Counter(labels.values())
    check("lactoperoxidase orthologues, human included", sizes["LPO"], 6)
    check("thyroid peroxidases", sizes["TPO"], 5)
    check("eosinophil peroxidases", sizes["EPX"], 4)
    check("myeloperoxidases", sizes["MPO"], 2)
    check("mammalian-type subfamily total", sum(sizes.values()), 17)
    check("PF03098 members with no mammalian-type label", len(unlabelled), 63)
    check("reference labels as its own subfamily", labels[ref], OWN)

    own = {h for h, l in labels.items() if l == OWN}
    para = {h for h, l in labels.items() if l != OWN}
    check("orthologue and paralogue sets disjoint", len(own & para), 0)

    sub = {k: v for k, v in fam.items() if k in labels}
    msa = P.align(sub)
    row = msa[ref]
    check("gapped reference row reconstructs its input",
          row.replace("-", "") == ref_seq, True)
    check("alignment width", len(row), 974)

    pos_to_col, pos = {}, 0
    for col, ch in enumerate(row):
        if ch != "-":
            pos += 1
            pos_to_col[pos] = col

    def column(p):
        col = pos_to_col[p]
        pairs = [(h, msa[h][col]) for h in msa if h != ref and msa[h][col] != "-"]
        sz = Counter(labels[h] for h, _ in pairs)
        pairs = [(h, r) for h, r in pairs if sz[labels[h]] >= P.MIN_SUBFAMILY_N]
        return [r for _, r in pairs], [labels[h] for h, _ in pairs]

    def sef(p):
        res, lab = column(p)
        h = h_bits(res)
        return (mi_bits(res, lab) / h) if h > 0 else None

    def ident(p, scope):
        col = pos_to_col[p]
        wt = ref_seq[p - 1]
        sel = [msa[h][col] for h in msa
               if h != ref and msa[h][col] != "-"
               and (scope == "fam" or labels.get(h) == OWN)]
        return sel.count(wt) / len(sel) if sel else None


    bad_fam = [p for p in CONTROLS if ident(p, "fam") != 1.0]
    bad_own = [p for p in CONTROLS if ident(p, "own") != 1.0]
    check("ten catalytic/cofactor residues, subfamily identity 1.0", bad_fam, [])
    check("ten catalytic/cofactor residues, orthologue identity 1.0", bad_own, [])
    check("ten catalytic/cofactor residues carry zero entropy",
          [p for p in CONTROLS if h_bits(column(p)[0]) != 0.0], [])


    res144, lab144 = column(144)
    check("D144 subfamily distribution", P.distribution(res144),
          "D:4 R:4 A:4 M:1 H:1 V:1 Y:1")
    check("D144 column entropy, bits", h_bits(res144), 2.5, tol=5e-4)
    check("D144 subfamily-depth identity", ident(144, "fam"), 0.25, tol=5e-4)
    check("D144 orthologue-depth identity", ident(144, "own"), 0.80, tol=5e-4)
    check("D144 subfamily-explained fraction", sef(144), 0.7695, tol=5e-4)

    check("E648 subfamily-explained fraction", sef(648), 1.0, tol=1e-9)
    check("E313 subfamily-explained fraction", sef(313), 1.0, tol=1e-9)
    check("E205 subfamily-explained fraction", sef(205), 0.7664, tol=5e-4)
    check("D479 subfamily-explained fraction", sef(479), 0.3846, tol=5e-4)


    usable = inv = 0
    for p in range(1, len(ref_seq) + 1):
        col = pos_to_col[p]
        raw = [msa[h][col] for h in msa if h != ref and h in labels]
        if sum(1 for r in raw if r == "-") / len(raw) > P.MAX_GAP_FRAC:
            continue
        usable += 1
        f_fam, f_own = ident(p, "fam"), ident(p, "own")
        if f_fam is not None and f_own is not None \
                and f_fam < P.INV_FAM_MAX and f_own >= P.INV_OWN_MIN:
            inv += 1
    check("positions passing the gap filter", usable, 703)
    check("inversions over the human sequence", inv, 132)

    failed = [n for n, ok in checks if not ok]
    print("\n%d checks, %d passed, %d failed"
          % (len(checks), len(checks) - len(failed), len(failed)))
    if failed:
        sys.exit("FAILED: %s" % failed)


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        sys.exit("FATAL: %s" % exc)
