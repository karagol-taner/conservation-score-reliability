#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import partition as P


HUMAN = "Homo sapiens"
MAMMALIAN_TYPE = ("Lactoperoxidase", "Thyroid peroxidase",
                  "Eosinophil peroxidase", "Myeloperoxidase")
SUBFAMILIES = {"LPO": ("Lactoperoxidase",), "TPO": ("Thyroid peroxidase",),
               "EPX": ("Eosinophil peroxidase",), "MPO": ("Myeloperoxidase",)}
OWN = "LPO"


MIN_COMPARATORS = {"D1_paralogues": 3, "D2_orthologues": 5,
                   "D3_subfamily": 16, "D4_family": 79}

FUNCTIONAL = (225, 226, 227, 301, 303, 305, 307, 372, 375, 468)
DESIGNED = (144, 648, 205, 313)
HELD_BACK = (479, 169)
NATURAL = (414, 514, 700)
SEED = 20261007

VARIABLE_MAX = P.INV_FAM_MAX
CONSERVED_MIN = P.INV_OWN_MIN


def select(fam, ref_header, which):

    out = []
    for h in fam:
        if h == ref_header:
            continue
        name, org = P.protein_name(h), P.organism(h)
        if which == "D1_paralogues":
            if org == HUMAN and any(p in name for p in MAMMALIAN_TYPE) \
                    and "Lactoperoxidase" not in name:
                out.append(h)
        elif which == "D2_orthologues":
            if "Lactoperoxidase" in name:
                out.append(h)
        elif which == "D3_subfamily":
            if any(p in name for p in MAMMALIAN_TYPE):
                out.append(h)
        elif which == "D4_family":
            out.append(h)
        else:
            raise P.Fatal("unknown depth %r" % which)
    return out


def depth_profile(fam, ref_header, ref_seq, comparators, log):

    seqs = {ref_header: ref_seq}
    seqs.update({h: fam[h] for h in comparators})
    msa = P.align(seqs)
    pos_to_col = P.map_positions(msa, ref_header, ref_seq)
    others = [h for h in msa if h != ref_header]
    prof = {}
    for pos, col in pos_to_col.items():
        wt = ref_seq[pos - 1]
        resid = [msa[h][col] for h in others]
        ungapped = [r for r in resid if r != P.GAP]
        prof[pos] = dict(
            n=len(ungapped),
            gap_frac=round(1.0 - len(ungapped) / len(resid), 4) if resid else 1.0,
            identity=(round(ungapped.count(wt) / len(ungapped), 4)
                      if ungapped else None))
    log("    aligned %d sequences, width %d columns, reference row reconstructs"
        % (len(seqs), len(next(iter(msa.values())))))
    return prof, msa, pos_to_col


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta",
                    default=P.local_data("an_peroxidase_PF03098.fasta"))
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("02_depth_ladder_peroxidase.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("One position, four sequence sets. The aligner, the guide tree and the")
    log("position mapping are held constant so that depth is the only variable.")
    log("")
    log("thresholds: variable at or below identity %.2f, conserved at or above "
        "%.2f" % (VARIABLE_MAX, CONSERVED_MIN))
    log("            gap filter %.2f, entropy floor %.2f bits, n_perm %d, "
        "seed %d" % (P.MAX_GAP_FRAC, P.H_MIN, a.n_perm, SEED))

    fam = P.read_fasta(a.fasta)
    ref_header = next((k for k in fam if "PERL_HUMAN" in k), None)
    if ref_header is None:
        raise P.Fatal("PERL_HUMAN not found in %s" % a.fasta)
    ref_seq = fam[ref_header]
    log("\nreference: human lactoperoxidase, %d residues, precursor numbering"
        % len(ref_seq))
    log("source: %d sequences in one file, so every depth draws the same copies"
        % len(fam))

    depths = list(MIN_COMPARATORS)
    sets = {}
    for d in depths:
        sets[d] = select(fam, ref_header, d)
        if len(sets[d]) < MIN_COMPARATORS[d]:
            raise P.Fatal("depth %s has %d comparators, below the stated "
                          "minimum of %d" % (d, len(sets[d]),
                                             MIN_COMPARATORS[d]))

    P.check_disjoint(sets["D2_orthologues"], sets["D1_paralogues"])
    log("\ncomparator sets")
    for d in depths:
        log("  %-16s %2d sequences" % (d, len(sets[d])))
    log("  D1 and D2 disjoint, so the paralogue-depth set holds no orthologue")

    prof = {}
    for d in depths:
        log("\n%s" % d)
        prof[d], msa_d, cols_d = depth_profile(fam, ref_header, ref_seq,
                                               sets[d], log)
        if d == "D3_subfamily":
            msa3, cols3 = msa_d, cols_d


    labels, _ = P.assign_labels(list(msa3), SUBFAMILIES)
    rng = np.random.default_rng(SEED)
    stat3 = {}
    for pos, col in cols3.items():
        res, labs, gap_frac, _ = P.column_set(msa3, col, ref_header, labels)
        s = P.column_partition(res, labs, rng, n_perm=a.n_perm)
        stat3[pos] = dict(sef=s["sef"], sef_adj=s["sef_adj"],
                          p_perm=s["p_perm"], h_total=s["h_total"],
                          call=P.classify(s))


    rows = []
    for pos in sorted(prof["D2_orthologues"]):
        wt = ref_seq[pos - 1]
        r = dict(position=pos, wt=wt,
                 set=("functional" if pos in FUNCTIONAL else
                      "designed" if pos in DESIGNED else
                      "held_back" if pos in HELD_BACK else
                      "natural_variant" if pos in NATURAL else ""))
        for d in depths:
            p = prof[d][pos]
            r["id_" + d] = p["identity"]
            r["gap_" + d] = p["gap_frac"]
            r["n_" + d] = p["n"]
        s = stat3[pos]
        r["h_total"] = round(s["h_total"], 4)
        r["sef"] = None if s["sef"] is None else round(s["sef"], 4)
        r["sef_adj"] = None if s["sef_adj"] is None else round(s["sef_adj"], 4)
        r["p_perm"] = s["p_perm"]
        r["call"] = s["call"]
        rows.append(r)

    def usable(r):
        return all(r["gap_" + d] <= P.MAX_GAP_FRAC for d in depths) \
               and all(r["id_" + d] is not None for d in depths)

    use = [r for r in rows if usable(r)]
    log("\n%d of %d positions pass the gap filter at every depth"
        % (len(use), len(rows)))


    log("\ninversion against orthologue depth: reads variable at the shallow")
    log("depth, conserved among the five orthologues")
    log("  depth             comparators  inverted  rate")
    summary = []
    for d in ("D1_paralogues", "D3_subfamily", "D4_family"):
        inv = [r for r in use
               if r["id_" + d] < VARIABLE_MAX
               and r["id_D2_orthologues"] >= CONSERVED_MIN]
        rate = len(inv) / len(use) if use else 0.0
        log("  %-16s %11d  %8d  %5.1f%%"
            % (d, len(sets[d]), len(inv), 100 * rate))
        summary.append(dict(measure="inversion_rate", depth=d,
                            comparators=len(sets[d]), n=len(inv),
                            denominator=len(use), value=round(rate, 4)))


    log("\nmean identity against alignment breadth, by the call at subfamily "
        "depth")
    log("  call           n     D1(3)   D2(5)   D3(16)  D4(79)")
    for call in ("conserved", "diagnostic", "tolerant", "indeterminate"):
        grp = [r for r in use if r["call"] == call]
        if not grp:
            continue
        means = [float(np.mean([r["id_" + d] for r in grp])) for d in depths]
        log("  %-13s %4d  %6.3f  %6.3f  %6.3f  %6.3f"
            % (call, len(grp), means[0], means[1], means[2], means[3]))
        for d, m in zip(depths, means):
            summary.append(dict(measure="mean_identity", depth=d, call=call,
                                n=len(grp), value=round(m, 4)))


    for call in ("diagnostic", "tolerant"):
        grp = [r for r in use if r["call"] == call]
        if not grp:
            continue
        drop = float(np.mean([r["id_D2_orthologues"] - r["id_D4_family"]
                              for r in grp]))
        log("  %s positions lose %.3f identity between orthologue and family "
            "depth" % (call, drop))
        summary.append(dict(measure="identity_drop_D2_to_D4", call=call,
                            n=len(grp), value=round(drop, 4)))


    def grades(key):
        vals = sorted((r[key], r["position"]) for r in use)
        g, n = {}, len(vals)
        for rank, (_, pos) in enumerate(vals):
            g[pos] = min(9, 1 + int(9 * rank / n))
        return g

    grade = {d: grades("id_" + d) for d in depths}
    for r in use:
        for d in depths:
            r["grade_" + d] = grade[d][r["position"]]

    log("\nthe same comparison in ConSurf's units: conservation grades 1 to 9 "
        "as")
    log("equal-count ranks within each depth, so divergence of the set cannot "
        "drive it")
    log("  depth             mean |shift|  shifted >=3  variable(<=3) at depth "
        "and conserved(>=7) at D2")
    for d in ("D1_paralogues", "D3_subfamily", "D4_family"):
        shifts = [abs(r["grade_" + d] - r["grade_D2_orthologues"]) for r in use]
        big = sum(1 for s in shifts if s >= 3)
        flip = sum(1 for r in use
                   if r["grade_" + d] <= 3 and r["grade_D2_orthologues"] >= 7)
        log("  %-16s %12.2f  %5d %5.1f%%  %5d %5.1f%%"
            % (d, float(np.mean(shifts)), big, 100 * big / len(use),
               flip, 100 * flip / len(use)))
        summary.append(dict(measure="mean_abs_grade_shift", depth=d,
                            n=len(use), value=round(float(np.mean(shifts)), 4)))
        summary.append(dict(measure="grade_shift_ge3_rate", depth=d, n=big,
                            denominator=len(use),
                            value=round(big / len(use), 4)))
        summary.append(dict(measure="grade_flip_variable_to_conserved", depth=d,
                            n=flip, denominator=len(use),
                            value=round(flip / len(use), 4)))


    var = [r for r in use if r["sef_adj"] is not None]
    if len(var) > 10:
        from scipy.stats import spearmanr
        x = [r["sef_adj"] for r in var]
        for d in ("D1_paralogues", "D3_subfamily", "D4_family"):
            y = [r["id_D2_orthologues"] - r["id_" + d] for r in var]
            rho, pv = spearmanr(x, y)
            log("\nSpearman(sef_adj, identity drop from orthologue depth to "
                "%s) = %.3f, p = %.3g, n = %d" % (d, rho, pv, len(var)))
            summary.append(dict(measure="spearman_sef_vs_drop", depth=d,
                                n=len(var), value=round(float(rho), 4),
                                p=float(pv)))


    log("\nthe four designed positions, the two held back, and the "
        "polymorphisms")
    log("  pos wt  set              D1(3)  D2(5)  D3(16) D4(79)  sef_adj  call")
    for pos in list(DESIGNED) + list(HELD_BACK) + list(NATURAL):
        r = next(x for x in rows if x["position"] == pos)
        log("  %-4d %s  %-15s %6s %6s %6s %6s  %7s  %s"
            % (r["position"], r["wt"], r["set"], r["id_D1_paralogues"],
               r["id_D2_orthologues"], r["id_D3_subfamily"],
               r["id_D4_family"], r["sef_adj"], r["call"]))

    variable_at_d1 = [p for p in DESIGNED
                   if next(x for x in rows if x["position"] == p)
                   ["id_D1_paralogues"] < CONSERVED_MIN]
    ortho_conserved = [p for p in DESIGNED
                       if next(x for x in rows if x["position"] == p)
                       ["id_D2_orthologues"] >= 0.8]
    log("\n  %d of the 4 designed positions read below %.2f identity at "
        "paralogue depth" % (len(variable_at_d1), CONSERVED_MIN))
    log("  %d of the 4 are at or above 0.80 identity among the orthologues"
        % len(ortho_conserved))
    if len(ortho_conserved) < 4:
        raise P.Fatal("only %d of the 4 designed positions are conserved among "
                      "the orthologues; the ladder is not reproducing the "
                      "conservation profile this control expects"
                      % len(ortho_conserved))


    os.makedirs(a.outdir, exist_ok=True)
    cols = (["position", "wt", "set"]
            + ["id_" + d for d in depths]
            + ["gap_" + d for d in depths]
            + ["n_" + d for d in depths]
            + ["grade_" + d for d in depths]
            + ["h_total", "sef", "sef_adj", "p_perm", "call"])
    out = os.path.join(a.outdir, "depth_ladder_peroxidase.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log("\nwrote %s (%d rows)" % (out, len(rows)))

    out2 = os.path.join(a.outdir, "depth_ladder_summary.csv")
    keys = ["measure", "depth", "call", "comparators", "n", "denominator",
            "value", "p"]
    with open(out2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    log("wrote %s (%d rows)" % (out2, len(summary)))

    logp = os.path.join(a.outdir, "depth_ladder_log.txt")
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        sys.exit("FATAL: %s" % exc)
